import os
import torch
import argparse
import datetime
import time
import torchvision
import wandb
import logging
import math
import shutil
import accelerate
import torch.utils.checkpoint
import diffusers
import torch.nn.functional as F
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import tensorflow.compat.v1 as tf
import os
import random
import torch.distributed as dist

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict
from omegaconf import OmegaConf
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, save, FullyShardedDataParallelPlugin
from accelerate.utils import TorchDynamoPlugin
from diffusers.models import AutoencoderKL
from copy import deepcopy
from einops import rearrange
from fit.utils.utils import (
    instantiate_from_config,
    default,
    get_obj_from_str,
)

from fit.utils.eval_utils import init_from_ckpt, calculate_inception_stats_imagenet
from fit.utils.evaluator import Evaluator
from fit.utils.utils import preprocess_raw_image, load_encoders
from fit.data.dataset import CustomDataset
from fit.data.in1k_latent_dataset import get_train_sampler

logger = get_logger(__name__, log_level="INFO")

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

# Redirect stdout and stderr for all processes, before any output
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def setup_terminal_logging(log_dir, process_idx=0):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"terminal_output_{process_idx}.txt")
    log_file = open(log_file_path, "a")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

@torch.no_grad()
def to_sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean #+ std * torch.randn_like(mean)
    #z = (z * latents_scale + latents_bias) 
    z = (z - latents_bias) * latents_scale
    return z 

@torch.no_grad()
def from_sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    z = (moments / latents_scale) + latents_bias
    return z

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def main(args):
    # set accelerator
    logging_dir = Path(args.logging.output_dir, args.logging.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.logging.output_dir, logging_dir=logging_dir
        )

    if args.compile.enabled:
        dynamo_plugin = TorchDynamoPlugin(
            backend=args.compile.backend,
            mode=args.compile.mode,
            fullgraph=args.compile.fullgraph,
            dynamic=args.compile.dynamic,
        )
    else:
        dynamo_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.optimization.gradient_accumulation_steps,
        mixed_precision=args.optimization.mixed_precision,
        log_with=args.logging.report_to,
        project_config=accelerator_project_config,
        dynamo_plugin=dynamo_plugin,
        #kwargs_handlers=[
        #DistributedDataParallelKwargs(find_unused_parameters=True)
        #]
    )

    save_dir = os.path.join(args.logging.output_dir, args.logging.exp_name)
    
    if accelerator.is_main_process:
        setup_terminal_logging(save_dir, int(os.environ.get("RANK", 0)))
        os.makedirs(args.logging.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(config=args, f=os.path.join(save_dir, "config.yaml"))

        # Create subfolders for checkpoints and logs
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    gen_results_dir = f"{save_dir}/gen_results"  # Stores generated samples

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(gen_results_dir, exist_ok=True)
        # Create logger
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.optimization.seed is not None:
        set_seed(args.optimization.seed + accelerator.process_index)

    # Create model:
    assert args.dataset.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.dataset.resolution // 8

    if args.loss.enc_type != "None":
        encoders, encoder_types, architectures = load_encoders(
            args.loss.enc_type, device, args.dataset.resolution
            )
    else:
        encoders, encoder_type, architectures = None, None, None
    z_dims = [encoder.embed_dim for encoder in encoders] if args.loss.enc_type != 'None' else [0]

    # Create model
    model = instantiate_from_config(args.network_config).to(device=device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad) 

    number_of_perflow = args.network_config.params.number_of_perflow
    sigmas = torch.linspace(0, 1, number_of_perflow+1).to(device)
    sigmas = sigmas.to(device=device)
    if accelerator.is_main_process:
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Fixed parameters: {fixed_params}")
        logger.info(f"Number of perflow: {number_of_perflow}")
        logger.info(f"Sigmas: {sigmas}")

    ema = deepcopy(model).to(device)
    vae_model = 'stabilityai/sd-vae-ft-ema'
    vae = AutoencoderKL.from_pretrained(vae_model, local_files_only=False).to(device=device)
    vae.eval() # important
    requires_grad(ema, False)

    # FID Evaluation pipeline
    if accelerator.is_main_process and args.evaluation.eval_fid:
        hf_config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        hf_config.gpu_options.allow_growth = True
        hf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
        evaluator = Evaluator(tf.Session(config=hf_config), batch_size=20)
        ref_acts = evaluator.read_activations_npz(args.evaluation.ref_path)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.evaluation.ref_path, ref_acts)
            
    torch.cuda.empty_cache()
    tf.reset_default_graph()

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.optimization.allow_tf32:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.capture_scalar_outputs = True
        torch.backends.cudnn.benchmark = True 
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.optimization.learning_rate,
        betas=(args.optimization.adam_beta1, args.optimization.adam_beta2),
        weight_decay=args.optimization.adam_weight_decay,
        eps=args.optimization.adam_epsilon,
    )    

    if accelerator.is_main_process:
        #tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=args.logging.name, 
            #config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.logging.exp_name}"}
            },
        )

    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    model, optimizer = accelerator.prepare(model, optimizer)
    ema = accelerator.prepare(ema)

    global_steps = 0
    if args.optimization.resume_step > 0:
        # Load checkpoint if resuming training
        dirs = os.listdir(checkpoint_dir)
        dirs = [d for d in dirs if d.isdigit()]
        dirs = sorted(dirs, key=lambda x: int(x))
        resume_from_path = dirs[-1] if len(dirs) > 0 else None
        if resume_from_path is None:
            if accelerator.is_main_process:
                logger.warning(f"No checkpoint found in {checkpoint_dir}. Starting from scratch.")
            global_steps = 0
        else:
            if accelerator.is_main_process:
                logger.info(f"Resuming from checkpoint: {resume_from_path}")
            accelerator.load_state(os.path.join(checkpoint_dir, resume_from_path))
            global_steps = int(resume_from_path)  # Use the last checkpoint as the global step
    
    total_batch_size = args.dataset.dataconfig.params.train.loader.batch_size * accelerator.num_processes * args.optimization.gradient_accumulation_steps
    # Setup data:
    train_dataset = CustomDataset(args.dataset.dataconfig.params.train.data_path)
    train_sampler = get_train_sampler(train_dataset, total_batch_size, args.optimization.max_train_steps, global_steps, args.optimization.seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.dataset.dataconfig.params.train.loader.batch_size, sampler=train_sampler, num_workers=args.dataset.dataconfig.params.train.loader.num_workers, pin_memory=True, drop_last=True)

    train_dataloader = accelerator.prepare(train_dataloader)

    progress_bar = tqdm(
        range(0, args.optimization.max_train_steps),
        initial=global_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 16
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    n = ys.size(0)
    patch_size = args.network_config.params.patch_size
    n_patch_h = args.network_config.params.n_patch_h
    n_patch_w = args.network_config.params.n_patch_w
    xT = torch.randn((n, n_patch_h*n_patch_w, (patch_size**2)*args.network_config.params.in_channels), device=device)
    H, W = n_patch_h * patch_size, n_patch_w * patch_size

    for epoch in range(args.optimization.epochs):
        for samples in train_dataloader:
            raw_x, x, y = samples
            raw_image = raw_x.to(device) / 255.
            # import pdb; pdb.set_trace()
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x = to_sample_posterior(x, latents_scale=vae.config.scaling_factor)
                x = x.reshape(x.shape[0], -1, latent_size//args.network_config.params.patch_size, args.network_config.params.patch_size, latent_size//args.network_config.params.patch_size, args.network_config.params.patch_size)
                x = rearrange(x, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                x = x.permute(0, 2, 1)  # (b, h, c)

            zs = []
            with accelerator.autocast():
                if args.loss.enc_type != "None":
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)
        
            loss = 0.0
            proj_loss = 0.0
            block_idx = 0
            per_block_idx = 0
            total_loss = 0.0
            total_proj_loss = 0.0
            
            #for layer_idx in range(number_of_perflow):
            x0 = torch.randn_like(x)
            #for layer_idx in range(number_of_perflow):
            for _ in range(1):
                layer_idx = random.randint(0, number_of_perflow-1)
                #layer_idx = global_steps % number_of_perflow
                sigma_next = sigmas[layer_idx + 1]
                sigma_current = sigmas[layer_idx]

                ratio_next = sigma_next.clone()
                while len(ratio_next.shape) < x0.ndim:
                    ratio_next = ratio_next.unsqueeze(-1)
                xt = x0 * (1-ratio_next) + x * ratio_next

                ratio = sigma_current.clone()
                while len(ratio.shape) < x0.ndim:
                    ratio = ratio.unsqueeze(-1)
                xt_input = x0 * (1-ratio) + x * ratio

                model_kwargs = dict(y=y, layer_idx=layer_idx)

                per_flow_ratio = torch.rand(x.shape[0]).to(device=device)
                t_input = sigma_current + per_flow_ratio.clone() * (sigma_next - sigma_current)
                while len(per_flow_ratio.shape) < x0.ndim:
                    per_flow_ratio = per_flow_ratio.unsqueeze(-1)
                x_input = xt_input * (1-per_flow_ratio) + xt * per_flow_ratio
                target = (xt - xt_input) / (sigma_next - sigma_current)
                weight = 1 #/ (sigma_next - sigma_current)
            
                with accelerator.autocast():
                    pred_model, representation_linear, loss_dummy = model(x_input, t_input, **model_kwargs)
                
                losses = mean_flat((((pred_model - target)) ** 2)) * weight
                loss_mean = losses.mean()

                if args.loss.enc_type != "None":
                    proj_loss_per = 0.0
                    repre_norm = F.normalize(representation_linear, dim=-1)
                    zs_norm = F.normalize(zs[0], dim=-1)
                    loss_per_token = -(zs_norm * repre_norm).sum(dim=-1)
                    proj_loss_mean = loss_per_token.mean()
                else:
                    proj_loss_mean = torch.tensor(0.0, device=device)

                # Backpropagate
                loss += loss_mean
                loss += 0.5 * proj_loss_mean
                loss += loss_dummy

                # total_loss += loss_mean
                # total_proj_loss += proj_loss_mean

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.optimization.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            accelerator.wait_for_everyone()

            # Gather the losses across all processes for logging (if we use distributed training).
            # total_loss = total_loss / number_of_perflow
            # total_proj_loss = total_proj_loss / number_of_perflow
            # avg_loss = accelerator.gather(total_loss.repeat(args.dataset.batch_size)).mean()
            # train_loss += avg_loss.item() / args.optimization.gradient_accumulation_steps
            # if args.enc_type is not None:
            #     avg_proj_loss = accelerator.gather(total_proj_loss.repeat(args.dataset.batch_size)).mean()

            # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
            if accelerator.sync_gradients: 
                update_ema(ema, model)
                    
                progress_bar.update(1)
                global_steps += 1

            if (global_steps % args.evaluation.checkpointing_steps == 0 and global_steps > 0):
                checkpoints = os.listdir(checkpoint_dir)
                checkpoints = [c for c in checkpoints if c.isdigit()]
                checkpoints = sorted(checkpoints, key=lambda x: int(x))
                if accelerator.is_main_process and len(checkpoints) >= args.evaluation.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.evaluation.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[:num_to_remove]
                    logger.info(f"Removing old checkpoints")
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = f"{checkpoint_dir}/{removing_checkpoint}"
                        shutil.rmtree(removing_checkpoint)

                checkpoint_path = f"{checkpoint_dir}/{global_steps:07d}"
                if accelerator.is_main_process:
                    os.makedirs(checkpoint_path, exist_ok=True)
                accelerator.wait_for_everyone()
                # Save the model state
                accelerator.save_state(checkpoint_path)
                if accelerator.is_main_process:
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                accelerator.wait_for_everyone()

                if global_steps in args.evaluation.checkpointing_steps_list:
                    checkpoint_path = os.path.join(checkpoint_dir, f"save-{global_steps:07d}")
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(checkpoint_path, exist_ok=True)
                    # Save the model state
                    accelerator.save_state(checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    accelerator.wait_for_everyone()
                
            if (global_steps == 1 or (global_steps % args.evaluation.sampling_steps == 0 and global_steps > 0)):
                model.eval()
                with torch.no_grad():
                    with accelerator.autocast():
                        
                        output_test = ema.forward_cfg(xT, 4, ys, number_of_step_perflow=6)
                        # else:
                        #     output_test = model.forward_cfg(xT, 4, y, number_of_step_perflow=6)

                        samples = output_test[:, : n_patch_h*n_patch_w]
                        # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        samples = ema.unpatchify(samples, (H, W))
                        # else:
                        #     samples = model.unpatchify(samples, (H, W))
                            
                        samples = from_sample_posterior(samples, vae.config.scaling_factor)
                        samples = vae.decode(samples.to(torch.bfloat16)).sample
                        samples = samples.clamp(-1, 1)
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    torchvision.utils.save_image(out_samples, f'{gen_results_dir}/{global_steps}.jpg', normalize=True, scale_each=True)
                    logging.info("Generating EMA samples done.")
                model.train()
                torch.cuda.empty_cache()
                    
            if args.evaluation.eval_fid and (global_steps % args.evaluation.evaluation_steps == 0 and global_steps > 0):
                all_images = []
                num_samples_per_process = args.evaluation.evaluation_number_samples // dist.get_world_size()
                with torch.no_grad():
                    number = 0
                    arr_list = []
                    test_fid_batch_size = args.evaluation.evaluation_batch_size

                    while num_samples_per_process > number:
                        latents = torch.randn((test_fid_batch_size, n_patch_h*n_patch_w, (patch_size**2)*args.network_config.params.in_channels), device=device)
                        y = torch.randint(0, 1000, (test_fid_batch_size,), device=device)

                        with accelerator.autocast():
                            # if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                            output_test = ema.forward_wo_cfg(latents, number_of_step_perflow=5, y=y)
                            # else:
                            #     output_test = ema_model.forward_wo_cfg(latents, t_test, 1, number_of_step_perflow=accelerate_cfg.test_nfe, y=y, representation_noise=latents)

                            samples = output_test[:, : n_patch_h*n_patch_w]
                            samples = ema.unpatchify(samples, (H, W))
                            # else:
                            #     samples = ema_model.unpatchify(samples, (H, W))
                            samples = from_sample_posterior(samples, vae.config.scaling_factor)
                            samples = vae.decode(samples.to(torch.bfloat16)).sample
                            samples = samples.clamp(-1, 1)
                            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                            number += samples.shape[0]

                        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
                        dist.all_gather(gathered_samples, samples)
                        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                    
                arr_list = np.concatenate(all_images, axis=0)
                arr_list = arr_list[: int(args.evaluation.evaluation_number_samples)]
                
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    sample_acts, sample_stats, sample_stats_spatial = calculate_inception_stats_imagenet(arr_list, evaluator)
                    inception_score = evaluator.compute_inception_score(sample_acts[0])
                    fid = sample_stats.frechet_distance(ref_stats)
                    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
                    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
                    logging.info("FID and Inception Score calculated.")
                    logger.info(f"Inception Score: {inception_score}")
                    logger.info(f"FID: {fid}")
                    logger.info(f"Spatial FID: {sfid}")
                    logger.info(f"Precision: {prec}")
                    logger.info(f"Recall: {recall}")
                    if args.logging.report_to == 'tensorboard' or args.logging.report_to == 'wandb':
                        accelerator.log({"inception_score": inception_score}, step=global_steps)
                        accelerator.log({"fid": fid}, step=global_steps)
                        accelerator.log({"sfid": sfid}, step=global_steps)
                        accelerator.log({"prec": prec}, step=global_steps)
                        accelerator.log({"recall": recall}, step=global_steps)
                torch.cuda.empty_cache()
                
            accelerator.wait_for_everyone()
            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
            }

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_steps)

        if global_steps >= args.optimization.max_train_steps and accelerator.is_main_process:
            logger.info(f'global steps ({global_steps}) >= max_train_steps ({args.optimization.max_train_steps}), stop training!!!')
            break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    return config

if __name__ == "__main__":
    args = parse_args()
    main(args)