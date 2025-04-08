import os
os.environ['TMPDIR'] = '/hub_data2/dogyun/tmpdir'
os.environ['TEMP'] = '/hub_data2/dogyun/tmpdir'
os.environ['TMP'] = '/hub_data2/dogyun/tmpdir'
os.environ['TORCH_HOME'] = '/hub_data2/dogyun/tmpdir'
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
import torch
import torch.utils.checkpoint
import diffusers
import numpy as np
import torch.nn.functional as F

from functools import partial
from torch.cuda import amp
from omegaconf import OmegaConf
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, save, FullyShardedDataParallelPlugin
from tqdm.auto import tqdm
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors import safe_open
from safetensors.torch import load_file
from copy import deepcopy
from einops import rearrange
from fit.scheduler.transport import create_transport
from fit.utils.utils import (
    instantiate_from_config,
    default,
    get_obj_from_str,
    update_ema,
    
)
from fit.utils.eval_utils import init_from_ckpt, calculate_inception_stats_cifar, compute_fid
from fit.utils.lr_scheduler import get_scheduler
from fit.model.fit_model import FiTBlock
from fit.model.modules import FinalLayer, PatchEmbedder, TimestepEmbedder, LabelEmbedder
from fit.scheduler.transport.utils import get_flexible_mask_and_ratio, mean_flat
from fit.data.cifar_dataset import create_cifar10_dataloader

from PIL import Image
import dnnlib
import pickle

from fvcore.nn import FlopCountAnalysis, flop_count_table

def cycle(dl):
    while True:
        for data in dl:
            yield data

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
        set_seed(args.seed)

    if args.allow_tf32: # for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    init_from_ckpt(ema_model, checkpoint_dir=args.pretrain_ckpt, ignore_keys=None, verbose=True)
    
    trainable_params = sum(p.numel() for p in ema_model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")

    number_of_perflow = args.number_of_perflow
    number_of_layers_for_perflow = args.number_of_layers_for_perflow
    assert number_of_perflow * number_of_layers_for_perflow == diffusion_cfg.distillation_network_config.params.depth, "The number of perflow and the number of layers for perflow must be equal to the depth of the distillation network."
    
    solver_step = args.solver_step
    logger.info(f"Solver step for perflow: {solver_step}")
    logger.info(f"Number of perflow: {number_of_perflow}")
    logger.info(f"Number of layers for perflow: {number_of_layers_for_perflow}")
    logger.info(f"Total sampling steps: {number_of_perflow * solver_step}")
    
    sigmas = torch.linspace(0, 1, number_of_perflow+1).to(device)
    logger.info(f"Sigmas: {sigmas}")

    train_dataloader = create_cifar10_dataloader(
        batch_size=data_cfg.params.train.loader.batch_size, num_workers=data_cfg.params.train.loader.num_workers, train=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)
    train_dataloader = cycle(train_dataloader)
    ema_model = accelerator.prepare_model(ema_model, device_placement=False)

    
    if accelerator.is_main_process and args.eval_fid:
        assert args.ref_path is not None, "Reference path is not provided."
        print('Loading Inception-v3 model...')
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        detector_kwargs = dict(return_features=True)
        feature_dim = 2048
        with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
            detector_net = pickle.load(f).to(device)
        with dnnlib.util.open_url(args.ref_path) as f:
            ref = dict(np.load(f))
        mu_ref = ref['mu']
        sigma_ref = ref['sigma']
    
    ema_model.eval()

    logger.info("Measuring model GFLOPs...")
    n_patch_h, n_patch_w = 16, 16
    patch_size = 2
    H, W = n_patch_h * patch_size, n_patch_w * patch_size
    
    # Create sample inputs for GFLOPs calculation
    sample_batch_size = 1
    sample_noise = torch.randn((sample_batch_size, diffusion_cfg.distillation_network_config.params.in_channels, H, W)).to(device=device)
    HX, WX = H, W
    for _ in range(2):
        HX = HX//2
        WX = WX//2
        sample_noise = torch.nn.functional.interpolate(sample_noise, size=(HX, WX), mode='bilinear') * 2
    sample_noise = sample_noise.reshape(sample_batch_size, -1, n_patch_h//4, patch_size, n_patch_w//4, patch_size)
    sample_noise = rearrange(sample_noise, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
    sample_noise = sample_noise.permute(0, 2, 1)
    sample_t = torch.zeros(sample_batch_size).to(device=device)
    sample_y = torch.zeros(sample_batch_size, dtype=torch.int).to(device=device)
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x, sample_t, torch.tensor(1.5).to(device), y=sample_y)
    
    # Define a proper wrapper class for sampling FLOPs counting
    class SamplingWrapper(torch.nn.Module):
        def __init__(self, model, nfe):
            super().__init__()
            self.model = model
            self.nfe = nfe
            
        def forward(self, x):
            return self.model.forward_cfg(x, sample_t, torch.tensor(1.5).to(device), y=sample_y, number_of_step_perflow=self.nfe)
        
    wrapper = ModelWrapper(ema_model)
    flops = FlopCountAnalysis(wrapper, sample_noise)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    total_flops = flops.total()
    logger.info(f"Single forward pass GFLOPs: {total_flops/1e9:.4f}")
    torch.cuda.empty_cache()

    for nfe in [1, 2]:
        with torch.no_grad():
            sampling_wrapper = SamplingWrapper(ema_model, nfe)
            sampling_flops = FlopCountAnalysis(sampling_wrapper, sample_noise)
            sampling_flops.unsupported_ops_warnings(False)
            sampling_flops.uncalled_modules_warnings(False)
            total_sampling_flops = sampling_flops.total()
        logger.info(f"Sampling GFLOPs (NFE={nfe}): {total_sampling_flops/1e9:.4f}")       
        torch.cuda.empty_cache()

    test_batch_size = 20
    n_patch_h, n_patch_w = 16, 16
    patch_size = 2
    H, W = n_patch_h * patch_size, n_patch_w * patch_size
    print('Generating images with resolution: ', H, 'x', W)
    y_test = torch.randint(0, args.num_classes, (test_batch_size,), device=device)
    noise_test = torch.randn((test_batch_size, diffusion_cfg.distillation_network_config.params.in_channels, H, W)).to(device=device)
    HX, WX = H, W
    for _ in range(2):
        HX = HX//2
        WX = WX//2
        noise_test = torch.nn.functional.interpolate(noise_test, size=(HX, WX), mode='bilinear') * 2
    noise_test = noise_test.reshape(test_batch_size, -1, n_patch_h//4, patch_size, n_patch_w//4, patch_size)
    noise_test = rearrange(noise_test, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
    noise_test = noise_test.permute(0, 2, 1)
    
    sigmas = torch.linspace(0, 1, number_of_perflow+1).to(device)

    with torch.no_grad():

        cfg_scale_test = 4 * torch.ones(1)
        cfg_scale_cond_test = cfg_scale_test.expand(noise_test.shape[0]).to(device=device)
        t_test = torch.zeros_like(cfg_scale_cond_test)

        for num_step_perflow in [1, 2, 6]:
            sampling_start = time.time()
            with accelerator.autocast():
                if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                    output_test = ema_model.module.forward_cfg(noise_test, t_test, 1.5, y=y_test, number_of_step_perflow=num_step_perflow)
                else:
                    output_test = ema_model.forward_cfg(noise_test, t_test, 1.5, y=y_test, number_of_step_perflow=num_step_perflow)
            
            sampling_time_1 = time.time() - sampling_start
            logger.info(f"Sampling time (NFE={num_step_perflow}): {sampling_time_1:.4f}s")

            samples = output_test[..., : n_patch_h*n_patch_w]
            if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                samples = ema_model.module.unpatchify(samples, (H, W))
            else:
                samples = ema_model.unpatchify(samples, (H, W))
            samples = samples.clamp(-1, 1)
            torchvision.utils.save_image(samples, os.path.join(f'{workdirnow}', f"images/fitv2_sample_test-{num_step_perflow}.jpg"), normalize=True, scale_each=True)
            torch.cuda.empty_cache()

    if 1:
        with torch.no_grad():
            number = 0
            arr_list = []
            test_fid_batch_size = 50

            while args.eval_fid_num_samples > number:
                latents = torch.randn((test_fid_batch_size, diffusion_cfg.distillation_network_config.params.in_channels, H, W)).to(device=device)
                HX, WX = H, W
                for _ in range(2):
                    HX = HX//2
                    WX = WX//2
                    latents = torch.nn.functional.interpolate(latents, size=(HX, WX), mode='bilinear') * 2
                latents = latents.reshape(test_fid_batch_size, -1, n_patch_h//4, patch_size, n_patch_w//4, patch_size)
                latents = rearrange(latents, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                latents = latents.permute(0, 2, 1)
                y = torch.randint(0, args.num_classes, (test_fid_batch_size,), device=device)

                #model_kwargs_fid = dict(y=y, grid=grid_test.long(), mask=mask_test, size=size_test)

                with accelerator.autocast():
                    if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                        output_test = ema_model.module.forward_cfg(latents, t_test, 2, y=y, number_of_step_perflow=6)
                    else:
                        output_test = ema_model.forward_cfg(latents, t_test, 1.5, y=y, number_of_step_perflow=2)

                samples = output_test[:, : n_patch_h*n_patch_w]
                if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                    samples = ema_model.module.unpatchify(samples, (H, W))
                else:
                    samples = ema_model.unpatchify(samples, (H, W))
                samples = samples.clamp(-1, 1)
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                arr = samples.cpu().numpy()
                arr_list.append(arr)
                number += arr.shape[0]
            
            arr_list = np.concatenate(arr_list, axis=0)
            mu, sigma = calculate_inception_stats_cifar(arr_list, detector_net=detector_net, detector_kwargs=detector_kwargs, device=device)
            fid = compute_fid(mu, sigma, ref_mu=mu_ref, ref_sigma=sigma_ref)
            print(f"FID: {fid}")
    accelerator.end_training()

if __name__ == "__main__":
    main()