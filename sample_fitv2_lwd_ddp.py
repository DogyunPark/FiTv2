import os
os.environ['TMPDIR'] = '/hub_data2/dogyun/tmpdir'
os.environ['TEMP'] = '/hub_data2/dogyun/tmpdir'
os.environ['TMP'] = '/hub_data2/dogyun/tmpdir'
os.environ['TORCH_HOME'] = '/hub_data2/dogyun/tmpdir'
import torch
import argparse
import datetime
import logging
import shutil
import accelerate
import torch.utils.checkpoint
import diffusers
import numpy as np
import torch.nn.functional as F

from omegaconf import OmegaConf
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, save, FullyShardedDataParallelPlugin
from tqdm.auto import tqdm
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from einops import rearrange
from fit.scheduler.transport import create_transport
from fit.utils.utils import (
    instantiate_from_config,
)
from fit.utils.eval_utils import init_from_ckpt
from fit.utils.lr_scheduler import get_scheduler
from fit.scheduler.transport.utils import get_flexible_mask_and_ratio
import time
import torchvision

from PIL import Image
logger = get_logger(__name__, log_level="INFO")

# For Omegaconf Tuple
def resolve_tuple(*args):
    return tuple(args)
OmegaConf.register_new_resolver("tuple", resolve_tuple)

def parse_args():
    parser = argparse.ArgumentParser(description="Argument.")
    parser.add_argument(
        "--project_name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="if setting, the logdir will be like: project_name",
    )
    parser.add_argument(
        "--main_project_name",
        type=str,
        default="Layer-Wise-Distillation",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="workdir",
        help="workdir",
    )
    parser.add_argument( # if resume, you change it none. i will load from the resumedir
        "--cfgdir",
        nargs="*",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--load_model_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be loaded from a pretrained model checkpoint."
            "Or you can set diffusion.pretrained_model_path in Config for loading!!!"
        ),
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="The decay rate for ema."
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=None,
        help="The path to the pretrained model checkpoint."
    )
    parser.add_argument(
        "--pretrain_ckpt2",
        type=str,
        default=None,
        help="The path to the pretrained model checkpoint."
    )
    parser.add_argument(
        "--solver_step",
        type=int,
        default=6,
        help="The number of steps to take in the solver."
    )
    parser.add_argument(
        "--use_elpips",
        action="store_true",
        default=False,
        help="Whether to use ELatentLPIPS."
    )
    parser.add_argument(
        "--number_of_perflow",
        type=int,
        default=36,
        help="The number of perflow."
    )
    parser.add_argument(
        "--number_of_layers_for_perflow",
        type=int,
        default=1,
        help="The number of layers for perflow."
    )
    parser.add_argument(
        "--random_perflow_step",
        action="store_true",
        default=False,
        help="Whether to use random perflow step."
    )
    parser.add_argument(
        "--reflow",
        action="store_true",
        default=False,
        help="Whether to use reflow."
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        default=False,
        help="Whether to use overlap."
    )
    parser.add_argument(
        "--consistency_loss",
        action="store_true",
        default=False,
        help="Whether to use consistency loss."
    )
    parser.add_argument(
        "--distillation",
        action="store_true",
        default=False,
        help="Whether to use distillation."
    )
    parser.add_argument(
        "--eval_fid",
        action="store_true",
        default=False,
        help="Whether to evaluate FID."
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default=None,
        help="The path to the reference image."
    )
    parser.add_argument(
        "--eval_fid_num_samples",
        type=int,
        default=5000,
        help="The number of samples to evaluate FID."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="The number of classes."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()
    
    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = None
    workdir = None
    workdirnow = None
    cfgdir = None
    ckptdir = None
    logging_dir = None
    imagedir = None
    
    if args.project_name:
        project_name = args.project_name
        if os.path.exists(os.path.join(args.workdir, project_name)): #open resume
            workdir=os.path.join(args.workdir, project_name)
        else: # new a workdir
            workdir = os.path.join(args.workdir, project_name)
            # if accelerator.is_main_process:
            os.makedirs(workdir, exist_ok=True)
        workdirnow = workdir

        cfgdir = os.path.join(workdirnow, "configs")
        ckptdir = os.path.join(workdirnow, "checkpoints")
        logging_dir = os.path.join(workdirnow, "logs")
        imagedir = os.path.join(workdirnow, "images")

        # if accelerator.is_main_process:
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(ckptdir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(imagedir, exist_ok=True)
    if args.cfgdir:
        load_cfgdir = args.cfgdir
    
    # setup config
    configs_list = load_cfgdir # read config from a config dir
    configs = [OmegaConf.load(cfg) for cfg in configs_list]
    config = OmegaConf.merge(*configs)
    accelerate_cfg = config.accelerate
    diffusion_cfg = config.diffusion
    data_cfg = config.data
    grad_accu_steps = accelerate_cfg.gradient_accumulation_steps
    
    train_strtg_cfg = getattr(config, 'training_strategy', None)
    if train_strtg_cfg != None:
        warp_pos_idx = hasattr(train_strtg_cfg, 'warp_pos_idx')
        if warp_pos_idx:
            warp_pos_idx_fn = partial(warp_pos_idx_from_grid, 
                shift=train_strtg_cfg.warp_pos_idx.shift, 
                scale=train_strtg_cfg.warp_pos_idx.scale,
                max_len=train_strtg_cfg.warp_pos_idx.max_len
            )

    accelerator_project_cfg = ProjectConfiguration(project_dir=workdirnow, logging_dir=logging_dir)
    
    if getattr(accelerate_cfg, 'fsdp_config', None) != None:
        import functools
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            BackwardPrefetch, CPUOffload, ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, ModuleWrapPolicy
        fsdp_cfg = accelerate_cfg.fsdp_config
        if accelerate_cfg.mixed_precision == "fp16":
            dtype = torch.float16
        elif accelerate_cfg.mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32   
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy = {
                'FULL_SHARD': ShardingStrategy.FULL_SHARD,
                'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
                'NO_SHARD': ShardingStrategy.NO_SHARD,
                'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
                'HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
            }[fsdp_cfg.sharding_strategy],
            backward_prefetch = {
                'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
                'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
            }[fsdp_cfg.backward_prefetch],
            # auto_wrap_policy = functools.partial(
            #     size_based_auto_wrap_policy, min_num_params=fsdp_cfg.min_num_params
            # ),
            auto_wrap_policy = ModuleWrapPolicy([FiTBlock, FinalLayer, PatchEmbedder, TimestepEmbedder, LabelEmbedder]),
            cpu_offload = CPUOffload(offload_params=fsdp_cfg.cpu_offload),
            state_dict_type = {
                'FULL_STATE_DICT': StateDictType.FULL_STATE_DICT,
                'LOCAL_STATE_DICT': StateDictType.LOCAL_STATE_DICT,
                'SHARDED_STATE_DICT': StateDictType.SHARDED_STATE_DICT
            }[fsdp_cfg.state_dict_type],
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            limit_all_gathers = fsdp_cfg.limit_all_gathers, # False
            use_orig_params = fsdp_cfg.use_orig_params, # True
            sync_module_states = fsdp_cfg.sync_module_states,   #True
            forward_prefetch = fsdp_cfg.forward_prefetch,   # False
            activation_checkpointing = fsdp_cfg.activation_checkpointing,   # False
        )
    else:
        fsdp_plugin = None
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accu_steps,
        mixed_precision=accelerate_cfg.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        log_with=getattr(accelerate_cfg, 'logger', 'wandb'),
        project_config=accelerator_project_cfg,
    )
    device = accelerator.device
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        File_handler = logging.FileHandler(os.path.join(logging_dir, project_name+"_"+datenow+".log"), encoding="utf-8")
        File_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        File_handler.setLevel(logging.INFO)
        logger.logger.addHandler(File_handler)
        
        diffusers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if args.seed is not None:
        # Add local rank to seed to differentiate between GPUs
        local_seed = args.seed + accelerator.process_index
        set_seed(local_seed)
        print(f"Setting seed to {local_seed} for process {accelerator.process_index}")

    if args.allow_tf32: # for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_grad_enabled(False)

    if args.scale_lr:
        learning_rate = (
            accelerate_cfg.learning_rate * 
            grad_accu_steps * 
            data_cfg.params.train.loader.batch_size *   # local batch size per device
            accelerator.num_processes / accelerate_cfg.learning_rate_base_batch_size    # global batch size
        )
    else:
        learning_rate = accelerate_cfg.learning_rate

    
    ema_model = instantiate_from_config(diffusion_cfg.distillation_network_config).to(device=device)
    trainable_params = sum(p.numel() for p in ema_model.parameters())
    logger.info(f"Total parameters: {trainable_params}")
    if args.pretrain_ckpt:
        init_from_ckpt(ema_model, checkpoint_dir=args.pretrain_ckpt, ignore_keys=None, verbose=True)

    number_of_perflow = args.number_of_perflow
    number_of_layers_for_perflow = args.number_of_layers_for_perflow
    assert number_of_perflow * number_of_layers_for_perflow == diffusion_cfg.distillation_network_config.params.depth, "The number of perflow and the number of layers for perflow must be equal to the depth of the distillation network."
    
    solver_step = args.solver_step
    logger.info(f"Solver step for perflow: {solver_step}")
    logger.info(f"Number of perflow: {number_of_perflow}")
    logger.info(f"Number of layers for perflow: {number_of_layers_for_perflow}")
    logger.info(f"Total sampling steps: {number_of_perflow * solver_step}")
    
    #scheduler = FlowMatchEulerDiscreteScheduler(invert_sigmas=True)
    #scheduler.set_timesteps(num_inference_steps, device=device)
    #timesteps = scheduler.timesteps
    #sigmas = scheduler.sigmas
    sigmas = torch.linspace(0, 1, number_of_perflow+1).to(device)
    logger.info(f"Sigmas: {sigmas}")

    ema_model = accelerator.prepare_model(ema_model, device_placement=False)

   
    vae_model = 'stabilityai/sd-vae-ft-ema'
    vae = AutoencoderKL.from_pretrained(vae_model, local_files_only=False).to(device)
    vae.eval() # important
    ema_model.eval()

    n_patch_h, n_patch_w = 16, 16
    patch_size = 2
    H, W = n_patch_h * patch_size, n_patch_w * patch_size
    with torch.no_grad():
        # Initialize variables for tracking and storing samples
        number = 0
        local_arr_list = []
        all_samples_collected = []  # For the main process to store all gathered samples
        gather_frequency = 1  # How often to perform gathering (every N iterations)
        
        iter_num = 0
        test_fid_batch_size = 64  # Restore the original batch size
        total_samples_needed = args.eval_fid_num_samples
        cfg_scale_test = 4 * torch.ones(1)
        cfg_scale_cond_test = cfg_scale_test.expand(test_fid_batch_size).to(device=device)
        t_test = torch.zeros_like(cfg_scale_cond_test)

        # Calculate samples per GPU
        samples_per_gpu = total_samples_needed // accelerator.num_processes
        print('samples_per_gpu: ', samples_per_gpu)
        if accelerator.is_main_process:
            samples_per_gpu += total_samples_needed % accelerator.num_processes

        while samples_per_gpu > number:
            latents = torch.randn((test_fid_batch_size, n_patch_h*n_patch_w, (patch_size**2)*diffusion_cfg.distillation_network_config.params.in_channels)).to(device=device)
            print(accelerator.process_index, latents[0,0,:3])
            y = torch.randint(0, args.num_classes, (test_fid_batch_size,), device=device)
            if accelerator.is_main_process:
                print('iter_num: ', iter_num)

            sampling_start = time.time()
            with accelerator.autocast():
                if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                    #output_test = ema_model.module.forward_maruyama(latents, t_test, 1.35, y=y, number_of_step_perflow=41, guidance_low=0.0, guidance_high=1.0, self_guidance=False)
                    output_test = ema_model.module.forward_maruyama_cfg(latents, t_test, 1.4, y=y, number_of_step_perflow=42)
                    #output_test = ema_model.module.forward_wo_cfg(latents, t_test, 1.35, y=y, number_of_step_perflow=41)
                else:
                    output_test = ema_model.forward_maruyama_cfg(latents, t_test, 2, y=y, number_of_step_perflow=41, guidance_low=0.35, guidance_high=1.0)
            
            sampling_time_2 = time.time() - sampling_start
            if accelerator.is_main_process:
                logger.info(f"Sampling time (NFE=250): {sampling_time_2:.4f}s")
            
            samples = output_test[:, : n_patch_h*n_patch_w]
            if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                samples = ema_model.module.unpatchify(samples, (H, W))
            else:
                samples = ema_model.unpatchify(samples, (H, W))
            
            samples = samples.to(torch.float32)
            samples = vae.decode(samples / vae.config.scaling_factor).sample
            # Save a sample image (only on main process)
            if accelerator.is_main_process and iter_num % 5 == 0:
                save_samples = samples.clone()
                save_samples = save_samples.clamp(-1, 1)
                torchvision.utils.save_image(save_samples, f"samples_iter_{iter_num}.png", normalize=True, scale_each=True)
            
            # Convert to uint8 numpy arrays
            samples = (samples + 1) / 2
            samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
            arr = samples.cpu().numpy()
            local_arr_list.append(arr)
            number += arr.shape[0]
            iter_num += 1
            
            # Perform gathering periodically to avoid accumulating too much memory
            if iter_num % gather_frequency == 0:
                # Only gather if we have new samples
                if local_arr_list:
                    logger.info(f"Rank {accelerator.process_index}: Gathering samples at iter {iter_num}")
                    
                    # Gather samples from all processes
                    gathered_batch = [None for _ in range(accelerator.num_processes)]
                    try:
                        torch.distributed.all_gather_object(gathered_batch, local_arr_list)
                    except Exception as e:
                        logger.error(f"Error during in-loop gathering: {e}")
                        # Continue with what we have
                        torch.distributed.barrier()
                    
                    # Only the main process needs to keep track of all samples
                    if accelerator.is_main_process:
                        for gpu_batch in gathered_batch:
                            if gpu_batch:  # Check if the process provided samples
                                all_samples_collected.extend(gpu_batch)
                        
                        total_so_far = sum(arr.shape[0] for arr in all_samples_collected)
                        logger.info(f"Collected {total_so_far} samples so far")
                        
                        # Optionally save intermediate results
                        if iter_num % 10 == 0:
                            # Save a subset or sample of what we have so far for preview
                            preview_samples = np.concatenate([all_samples_collected[0]] if all_samples_collected else [np.zeros((1, 256, 256, 3), dtype=np.uint8)])
                            np.savez(f"/hub_data2/dogyun/fitv2_samples_intermediate.npz", arr_0=preview_samples)
                    
                    # Clear local list after gathering to free memory
                    local_arr_list = []
                
                # Synchronize after gathering
                #torch.distributed.barrier()
        
        #torch.distributed.monitored_barrier(group=None, timeout=600000)

        # Final gathering of any remaining samples
        if local_arr_list:
            logger.info(f"Rank {accelerator.process_index}: Final gathering of remaining samples")
            gathered_final = [None for _ in range(accelerator.num_processes)]
            try:
                torch.distributed.all_gather_object(gathered_final, local_arr_list)
            except Exception as e:
                logger.error(f"Error during final gathering: {e}")
                if accelerator.is_main_process:
                    logger.warning("Proceeding with samples collected so far")
                # Continue with what we have
                torch.distributed.barrier()
            
            if accelerator.is_main_process:
                for gpu_batch in gathered_final:
                    if gpu_batch:
                        all_samples_collected.extend(gpu_batch)
        
        torch.distributed.barrier()
        # Only main process saves the final npz file
        if accelerator.is_main_process:
            # Concatenate all gathered arrays
            if all_samples_collected:
                all_samples = np.concatenate(all_samples_collected, axis=0)
                print('Total number of samples: ', all_samples.shape[0])
                
                # Ensure we have exactly the number of samples we want
                all_samples = all_samples[:args.eval_fid_num_samples]
                
                npz_path = '/hub_data2/dogyun/fitv2_sample_test_bfm_repa.npz'
                np.savez(npz_path, arr_0=all_samples)
                logger.info(f"Saved final {all_samples.shape[0]} samples to {npz_path}")
            else:
                logger.error("No samples were collected!")
        
        torch.distributed.barrier()
    accelerator.end_training()

if __name__ == "__main__":
    main()