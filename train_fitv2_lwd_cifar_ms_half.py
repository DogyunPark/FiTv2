import os
os.environ['TMPDIR'] = '/hub_data2/dogyun/tmpdir'
os.environ['TEMP'] = '/hub_data2/dogyun/tmpdir'
os.environ['TMP'] = '/hub_data2/dogyun/tmpdir'
os.environ['TORCH_HOME'] = '/hub_data2/dogyun/tmpdir'
import torch
import pickle
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
#from diffusers.models import AutoencoderKL
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
#from fit.utils.utils import bell_shaped_sample, discrete_lognormal_sample
from fit.utils.eval_utils import init_from_ckpt, compute_fid, calculate_inception_stats_cifar
from fit.utils.lr_scheduler import get_scheduler
from fit.model.fit_model_lwd import FiTBlock, RepresentationBlock
from fit.model.modules_lwd import FinalLayer, PatchEmbedder, TimestepEmbedder, LabelEmbedder
from fit.scheduler.transport.utils import get_flexible_mask_and_ratio, mean_flat
from fit.utils.utils import preprocess_raw_image, load_encoders

from PIL import Image
#from fit.scheduler.transport.utils import loss_func_huber
from fit.data.cifar_dataset import create_cifar10_dataloader
import dnnlib

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
        "--network_pkl",
        type=str,
        default=None,
        help="The path to the pretrained model checkpoint."
    )
    parser.add_argument(
        "--edm_sigmas",
        action="store_true",
        default=False,
        help="Whether to use EDM sigmas."
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
        "--double",
        action="store_true",
        default=False,
        help="Whether to use double."
    )
    parser.add_argument(
        "--enc_type",
        type=str,
        default=None,
        help="The type of encoder."
    )
    parser.add_argument(
        "--multi_scale",
        action="store_true",
        default=False,
        help="Whether to use multi-scale."
    )
    parser.add_argument(
        "--contrastive_loss",
        action="store_true",
        default=False,
        help="Whether to use contrastive loss."
    )
    parser.add_argument(
        "--structured_loss",
        action="store_true",
        default=False,
        help="Whether to use structured loss."
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
            auto_wrap_policy = ModuleWrapPolicy([FiTBlock, RepresentationBlock, FinalLayer, PatchEmbedder, TimestepEmbedder, LabelEmbedder]),
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
    
    #if args.seed is not None:
    #    set_seed(args.seed)

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

    if args.distillation:
        with dnnlib.util.open_url(args.network_pkl) as f:
            pretrained_model = pickle.load(f)['ema'].to(device)
        pretrained_model.eval()

        sigma_min = 0.002
        sigma_max = 80
        rho = 7.0

        # Time step discretization.
        step_indices = torch.arange(args.number_of_perflow, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (args.number_of_perflow - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([pretrained_model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    model = instantiate_from_config(diffusion_cfg.distillation_network_config).to(device=device)
    #init_from_ckpt(model, checkpoint_dir=args.pretrain_ckpt2, ignore_keys=None, verbose=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad) 
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Fixed parameters: {fixed_params}")

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
    if args.edm_sigmas:
        sigmas = t_steps
    else:
        sigmas = torch.linspace(0, 1, number_of_perflow+1).to(device)
        # weights = torch.arange(1, number_of_perflow+1)
        # total_weight = weights.sum()
        # segment_length = weights / total_weight
        # sigmas = torch.cat((torch.tensor([0.0]), torch.cumsum(segment_length, dim=0)))
        # sigmas = sigmas.to(device=device)
    logger.info(f"Sigmas: {sigmas}")

    # update ema
    if args.use_ema:
        # ema_dtype = torch.float32
        if hasattr(model, 'module'):
            ema_model = deepcopy(model.module).to(device=device)
        else:
            ema_model = deepcopy(model).to(device=device)
        if getattr(diffusion_cfg, 'pretrain_config', None) != None: # transfer to larger reolution
            if getattr(diffusion_cfg.pretrain_config, 'ema_ckpt', None) != None:
                init_from_ckpt(
                    ema_model, checkpoint_dir=diffusion_cfg.pretrain_config.ema_ckpt, 
                    ignore_keys=diffusion_cfg.pretrain_config.ignore_keys, verbose=True
                )
        for p in ema_model.parameters():
            p.requires_grad = False
    
    if args.use_ema:
        model = accelerator.prepare_model(model, device_placement=False)
        ema_model = accelerator.prepare_model(ema_model, device_placement=False)
    else:
        model = accelerator.prepare_model(model, device_placement=False)
        
    # In SiT, we use transport instead of diffusion
    transport = create_transport(**OmegaConf.to_container(diffusion_cfg.transport))  # default: velocity; 
    # schedule_sampler = create_named_schedule_sampler()

    # Setup Dataloader
    total_batch_size = data_cfg.params.train.loader.batch_size * accelerator.num_processes * grad_accu_steps
    global_steps = 0
    if args.resume_from_checkpoint:
        # normal read with safety check
        if args.resume_from_checkpoint != "latest":
            resume_from_path = os.path.basename(args.resume_from_checkpoint)
        else:   # Get the most recent checkpoint
            dirs = os.listdir(ckptdir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_from_path = dirs[-1] if len(dirs) > 0 else None

        if resume_from_path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            global_steps = int(resume_from_path.split("-")[1]) # gs not calculate the gradient_accumulation_steps
            logger.info(f"Resuming from steps: {global_steps}")

    train_dataloader = create_cifar10_dataloader(
        batch_size=data_cfg.params.train.loader.batch_size, num_workers=data_cfg.params.train.loader.num_workers, train=True
    )

    # Setup optimizer and lr_scheduler
    if getattr(diffusion_cfg, 'pretrain_config', None) != None: # transfer to larger reolution     
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = list(model.parameters())
    optimizer_cfg = default(
        accelerate_cfg.optimizer, {"target": "torch.optim.AdamW"}
    )
    optimizer = get_obj_from_str(optimizer_cfg["target"])(
        params, lr=learning_rate, **optimizer_cfg.get("params", dict())
    )
    #optimizer = CAME(params, lr=learning_rate, **accelerate_cfg.optimizer.params)
    lr_scheduler = get_scheduler(
        accelerate_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=accelerate_cfg.lr_warmup_steps,
        num_training_steps=accelerate_cfg.max_train_steps,
    )
    
    # Prepare Accelerate
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    train_dataloader = cycle(train_dataloader)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and getattr(accelerate_cfg, 'logger', 'wandb') != None:
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), workdirnow)
        accelerator.init_trackers(
            args.main_project_name, 
            #config=config, 
            init_kwargs={"wandb": {"group": args.project_name}}
        )

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

    if args.enc_type is not None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, 256
            )
        
        # encoders2, encoder_types2, architectures2 = load_encoders(
        #     'jepa-vit-h', device, 256
        #     )
    # Train!
    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {data_cfg.params.train.loader.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Learning rate = {learning_rate}")
    logger.info(f"  Gradient Accumulation steps = {grad_accu_steps}")
    logger.info(f"  Total optimization steps = {accelerate_cfg.max_train_steps}")
    logger.info(f"  Current optimization steps = {global_steps}")
    #logger.info(f"  Train dataloader length = {len(train_dataloader)} ")
    logger.info(f"  Training Mixed-Precision = {accelerate_cfg.mixed_precision}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # normal read with safety check
        error_times=0
        while(True):
            if error_times >= 100:
                raise
            try:
                logger.info(f"Resuming from checkpoint {resume_from_path}")
                accelerator.load_state(os.path.join(ckptdir, resume_from_path))
                break
            except (RuntimeError, Exception) as err:
                error_times+=1
                if accelerator.is_local_main_process:
                    logger.warning(err)
                    logger.warning(f"Failed to resume from checkpoint {resume_from_path}")
                    shutil.rmtree(os.path.join(ckptdir, resume_from_path))
                else:
                    time.sleep(2)
    
    # save config
    OmegaConf.save(config=config, f=os.path.join(cfgdir, "config.yaml"))
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, accelerate_cfg.max_train_steps), 
        disable = not accelerator.is_main_process
    )
    progress_bar.set_description("Optim Steps")
    progress_bar.update(global_steps)
    
    if args.use_ema:
        # ema_model = ema_model.to(ema_dtype)
        ema_model.eval()
    # Training Loop
    model.train()
    train_loss = 0.0

    test_batch_size = accelerate_cfg.test_batch_size
    n_patch_h, n_patch_w = diffusion_cfg.distillation_network_config.params.n_patch_h, diffusion_cfg.distillation_network_config.params.n_patch_w
    patch_size = diffusion_cfg.distillation_network_config.params.patch_size
    H, W = n_patch_h * patch_size, n_patch_w * patch_size
    print('Generating images with resolution: ', H, 'x', W)
    y_test = torch.randint(0, data_cfg.class_num, (test_batch_size,), device=device)
    print('Class: ', y_test)
    if diffusion_cfg.distillation_network_config.params.multi_scale:
        print('Multi-scale: ', diffusion_cfg.distillation_network_config.params.multi_scale)
        print('Down-scale factor: ', diffusion_cfg.distillation_network_config.params.down_scale_factor)
        down_scale_factor = diffusion_cfg.distillation_network_config.params.down_scale_factor
        noise_test = torch.randn((test_batch_size, diffusion_cfg.distillation_network_config.params.in_channels, H, W)).to(device=device)
        HX, WX = H, W
        for _ in range(down_scale_factor):
            HX = HX//2
            WX = WX//2
            noise_test = torch.nn.functional.interpolate(noise_test, size=(HX, WX), mode='bilinear') * 2
        noise_test = noise_test.reshape(test_batch_size, -1, n_patch_h//(2**down_scale_factor), patch_size, n_patch_w//(2**down_scale_factor), patch_size)
        noise_test = rearrange(noise_test, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
        noise_test = noise_test.permute(0, 2, 1)
        multi_scale_index_list = [int(number_of_perflow//3), int(2*(number_of_perflow//3))]
        print('Multi scale index:', multi_scale_index_list)
    else:
        noise_test = torch.randn((test_batch_size, n_patch_h*n_patch_w, (patch_size**2)*diffusion_cfg.distillation_network_config.params.in_channels)).to(device=device)
    
    noise_test_list = [torch.randn((test_batch_size, n_patch_h*n_patch_w, (patch_size**2)*diffusion_cfg.distillation_network_config.params.in_channels)).to(device=device) for _ in range(number_of_perflow-1)]

    for step in range(global_steps, accelerate_cfg.max_train_steps):
        batch = next(train_dataloader)
        x, y = batch[0], batch[1]
        x_data = x * 2 - 1

        # prepare other parameters
        y = y.to(torch.int)
        #mask = torch.ones(data_cfg.params.train.loader.batch_size, n_patch_h*n_patch_w).to(device=device, dtype=torch.bfloat16)

        cfg_scale = torch.randint(1, 2, (1,))
        cfg_scale_cond = cfg_scale.expand(x.shape[0]).to(device=device)

        
        x_noise = torch.randn_like(x_data)
        loss = 0.0
        proj_loss = 0.0

        if args.enc_type is not None:
            # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            #     raw_x = model.module.unpatchify(x, (H, W))
            # else:
            #     raw_x = model.unpatchify(x, (H, W))
            raw_x = x_data
            with torch.no_grad():
                raw_x = (raw_x + 1)/2.
                raw_x = preprocess_raw_image(raw_x, args.enc_type)
                raw_x = raw_x.to(torch.float32)
                with accelerator.autocast():
                    raw_z_data = encoders[0].forward_features(raw_x)
                    if 'dinov2' in args.enc_type:
                        raw_z_cls = raw_z_data['x_norm_clstoken']
                        raw_z_data = raw_z_data['x_norm_patchtokens']
                        #import pdb; pdb.set_trace()
                
                #raw_z2 = encoders2[0].forward_features(raw_x)

        with accelerator.accumulate(model):
            for layer_idx in range(number_of_perflow):
            #for _ in range(4):
                #layer_idx = torch.randint(0, number_of_perflow, (1,)).item()
                #layer_idx = bell_shaped_sample(0, number_of_perflow, 5, 1, 5)[0]
                #layer_idx = discrete_lognormal_sample(0, number_of_perflow, 1, 1, 1)[0] - 1
                mod_index = layer_idx % (number_of_perflow // 3) / 3
                mod_index_next = (layer_idx + 1) % (number_of_perflow // 3) / 3

                if args.multi_scale:
                    if layer_idx < number_of_perflow//3:
                        x_past = None

                        x0 = x_noise.clone()
                        x = x_data.clone()
                        HX, WX = H, W
                        for _ in range(2):
                            HX = HX//2
                            WX = WX//2
                            x0 = torch.nn.functional.interpolate(x0, size=(HX, WX), mode='bilinear') * 2
                            x = torch.nn.functional.interpolate(x, size=(HX, WX), mode='bilinear')
                        x = x.reshape(x.shape[0], -1, n_patch_h//4, patch_size, n_patch_w//4, patch_size)
                        x = rearrange(x, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x = x.permute(0, 2, 1)
                        x0 = x0.reshape(x0.shape[0], -1, n_patch_h//4, patch_size, n_patch_w//4, patch_size)
                        x0 = rearrange(x0, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x0 = x0.permute(0, 2, 1)

                        start_idx = 0
                        end_idx = int(number_of_perflow//3)
                        start_sigma = sigmas[start_idx]
                        end_sigma = sigmas[end_idx]

                        x_start = x0.clone()
                        ratio_end = end_sigma.clone()
                        while len(ratio_end.shape) < x0.ndim:
                            ratio_end = ratio_end.unsqueeze(-1)
                        x_end = x0 * (1-ratio_end) + x * ratio_end

                        if args.enc_type is not None:
                            raw_z = rearrange(raw_z_data, 'b (h w) c -> b c h w', h=n_patch_h, w=n_patch_w)
                            #raw_z = rearrange(raw_z, 'b h w c -> b c h w')
                            raw_z = torch.nn.functional.interpolate(raw_z, size=(n_patch_h//4, n_patch_w//4), mode='bilinear')
                            raw_z = rearrange(raw_z, 'b c h w -> b (h w) c', h=n_patch_h//4, w=n_patch_w//4)

                    elif number_of_perflow//3 <= layer_idx < 2*number_of_perflow//3:
                    #if layer_idx < number_of_perflow//2:
                        x_past = x_data.clone()
                        HX, WX = H, W
                        for _ in range(2):
                            HX = HX//2
                            WX = WX//2
                            x_past = torch.nn.functional.interpolate(x_past, size=(HX, WX), mode='bilinear')
                        x_past = torch.nn.functional.interpolate(x_past, size=(H//2, W//2), mode='nearest')
                        x_past = x_past.reshape(x_past.shape[0], -1, n_patch_h//2, patch_size, n_patch_w//2, patch_size)
                        x_past = rearrange(x_past, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x_past = x_past.permute(0, 2, 1)
                        #x_past = None
                        
                        x = torch.nn.functional.interpolate(x_data, size=(H//2, W//2), mode='bilinear')
                        x = x.reshape(x.shape[0], -1, n_patch_h//2, patch_size, n_patch_w//2, patch_size)
                        x = rearrange(x, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x = x.permute(0, 2, 1)

                        x0 = torch.nn.functional.interpolate(x_noise, size=(H//2, W//2), mode='bilinear') * 2
                        x0 = x0.reshape(x_noise.shape[0], -1, n_patch_h//2, patch_size, n_patch_w//2, patch_size)
                        x0 = rearrange(x0, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x0 = x0.permute(0, 2, 1)

                        start_idx = int(number_of_perflow//3)
                        end_idx = int(2*(number_of_perflow//3))
                        start_sigma = sigmas[start_idx]
                        ori_sigma = start_sigma
                        gamma = 1/3
                        corrected_sigma = (1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)) * ori_sigma
                        start_sigma = corrected_sigma

                        end_sigma = sigmas[end_idx]

                        ratio_start = start_sigma.clone()
                        while len(ratio_start.shape) < x0.ndim:
                            ratio_start = ratio_start.unsqueeze(-1)
                        x_start = x0 * (1-ratio_start) + x_past * ratio_start

                        ratio_end = end_sigma.clone()
                        while len(ratio_end.shape) < x0.ndim:
                            ratio_end = ratio_end.unsqueeze(-1)
                        x_end = x0 * (1-ratio_end) + x * ratio_end
                            

                        if args.enc_type is not None:
                            raw_z = rearrange(raw_z_data, 'b (h w) c -> b c h w', h=n_patch_h, w=n_patch_w)
                            #raw_z = rearrange(raw_z, 'b h w c -> b c h w')
                            raw_z = torch.nn.functional.interpolate(raw_z, size=(n_patch_h//2, n_patch_w//2), mode='bilinear')
                            raw_z = rearrange(raw_z, 'b c h w -> b (h w) c', h=n_patch_h//2, w=n_patch_w//2)

                    else:
                        x_past = torch.nn.functional.interpolate(x_data, size=(H//2, W//2), mode='bilinear')
                        x_past = torch.nn.functional.interpolate(x_past, size=(H, W), mode='nearest')
                        x_past = x_past.reshape(x_past.shape[0], -1, n_patch_h, patch_size, n_patch_w, patch_size)
                        x_past = rearrange(x_past, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x_past = x_past.permute(0, 2, 1)

                        x = x_data.reshape(x_data.shape[0], -1, n_patch_h, patch_size, n_patch_w, patch_size)
                        x = rearrange(x, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x = x.permute(0, 2, 1)
                        x0 = x_noise.reshape(x_noise.shape[0], -1, n_patch_h, patch_size, n_patch_w, patch_size)
                        x0 = rearrange(x0, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                        x0 = x0.permute(0, 2, 1)

                        start_idx = int(2*(number_of_perflow//3))
                        start_sigma = sigmas[start_idx]
                        ori_sigma = start_sigma
                        gamma = 1/3
                        corrected_sigma = (1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)) * ori_sigma
                        start_sigma = corrected_sigma
                        
                        ratio_start = start_sigma.clone()
                        while len(ratio_start.shape) < x0.ndim:
                            ratio_start = ratio_start.unsqueeze(-1)
                        x_start = x0 * (1-ratio_start) + x_past * ratio_start
                        x_end = x.clone()

                        if args.enc_type is not None:
                            raw_z = raw_z_data

                # ratio_start = mod_index.clone()
                # while len(ratio_start.shape) < x0.ndim:
                #     ratio_start = ratio_start.unsqueeze(-1)
                # xt = x_start * (1-ratio_start) + x_end * ratio_start
                # ratio_end = mod_index_next.clone()
                # while len(ratio_end.shape) < x0.ndim:
                #     ratio_end = ratio_end.unsqueeze(-1)
                # xt_input = x_start * (1-ratio_end) + x_end * ratio_end
                
                model_kwargs = dict(y=y, target_layer_start=layer_idx * number_of_layers_for_perflow, target_layer_end=layer_idx * number_of_layers_for_perflow + number_of_layers_for_perflow)


                sigma_next = sigmas[layer_idx + 1]
                #sigma_next = sigmas[-1]
                if args.overlap:
                    if layer_idx == 0:
                        sigma_current = sigmas[layer_idx]
                    else:
                        sigma_current = sigmas[layer_idx] - 1/(number_of_perflow*2)
                else:
                    sigma_current = sigmas[layer_idx]
                    if args.multi_scale and layer_idx in multi_scale_index_list:
                        ori_sigma = sigma_current
                        gamma = 1/3
                        corrected_sigma = (1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)) * ori_sigma
                        sigma_current = corrected_sigma
                    else:
                        sigma_current = sigmas[layer_idx]


                if args.random_perflow_step:
                    perflow_solver_step = torch.randint(1, solver_step+1, (1,)).item()
                else:
                    perflow_solver_step = solver_step
                
                sigma_list = torch.linspace(sigma_current.item(), sigma_next.item(), perflow_solver_step+1)
                
                ratio = sigma_current.clone()
                while len(ratio.shape) < x0.ndim:
                    ratio = ratio.unsqueeze(-1)

                ratio_next = sigma_next.clone()
                while len(ratio_next.shape) < x0.ndim:
                    ratio_next = ratio_next.unsqueeze(-1)
                xt = x0 * (1-ratio_next) + x * ratio_next

                if args.reflow:
                    if args.edm_sigmas:
                        xt_input = x + sigmas[layer_idx] * x0
                    else:
                        if args.multi_scale:
                            if x_past is not None and layer_idx in multi_scale_index_list:
                                xt_input = x0 * (1-ratio) + x_past * ratio
                            else:
                                xt_input = x0 * (1-ratio) + x * ratio
                        else:
                            xt_input = x0 * (1-ratio) + x * ratio
                    per_flow_ratio = torch.randint(0, 1000, (x.shape[0],)) / 1000
                    per_flow_ratio = per_flow_ratio.to(device=device)
                    #per_flow_ratio = torch.rand(x.shape[0]).to(device=device)
                    t_input = sigma_current + per_flow_ratio.clone() * (sigma_next - sigma_current)
                    while len(per_flow_ratio.shape) < x0.ndim:
                        per_flow_ratio = per_flow_ratio.unsqueeze(-1)
                    x_input = xt_input * (1-per_flow_ratio) + xt * per_flow_ratio
                    target = (xt - xt_input) / (sigma_next - sigma_current)
                    weight = 1 #/ (sigma_next - sigma_current)

                # forward model and compute loss
                with accelerator.autocast():
                    #_, _ = get_flexible_mask_and_ratio(model_kwargs, x_input)
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        pred_model, representation_linear, representation_linear_cls = model.module.forward_run_layer(x_input, t_input, cfg_scale_cond, **model_kwargs, t_next=None, representation_noise=x0)
                    else:
                        pred_model, representation_linear, representation_linear_cls = model.forward_run_layer(x_input, t_input, cfg_scale_cond, **model_kwargs, t_next=None, representation_noise=x0)
                    
                    #pred_model, representation_linear, representation_linear_cls = model(x_input, t_input, cfg_scale_cond, **model_kwargs, t_next=None, representation_noise=x0)
                    
                    
                losses = mean_flat(((pred_model - target)**2)) * weight
                loss += losses.mean()
                if args.enc_type is not None:
                    proj_loss_per = 0.0
                    # representation_linear [B N C]
                    # raw_z [B N C]
                    for j, (repre_j, raw_z_j) in enumerate(zip(representation_linear, raw_z)):
                        raw_z_j = torch.nn.functional.normalize(raw_z_j, dim=-1) 
                        repre_j = torch.nn.functional.normalize(repre_j, dim=-1) 
                        proj_loss_per += mean_flat(-(raw_z_j * repre_j).sum(dim=-1))

                        if args.contrastive_loss:
                            # Calculate similarity matrix between all pairs in the batch
                            sim_matrix = torch.matmul(repre_j, repre_j.transpose(-2, -1))  # [B, N, N]
                            
                            # Create labels - diagonal elements are positives
                            labels = torch.arange(repre_j.shape[0], device=repre_j.device)
                            labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
                            
                            # Temperature parameter for scaling
                            temperature = 0.1
                            
                            # InfoNCE loss
                            sim_matrix = sim_matrix / temperature
                            exp_sim = torch.exp(sim_matrix)
                            
                            # Mask out self-similarity
                            mask = torch.eye(repre_j.shape[0], device=repre_j.device)
                            exp_sim = exp_sim * (1 - mask)
                            
                            # Calculate positive and negative terms
                            positive_sim = sim_matrix[labels]
                            negative_sim = torch.log(exp_sim.sum(dim=1))
                            
                            # Compute contrastive loss
                            contrastive_loss = (-positive_sim + negative_sim).mean()
                            
                            # Add to projection loss with a weight factor
                            proj_loss_per += 0.1 * contrastive_loss
                        
                        if args.structured_loss:
                            # Add structure loss
                            # Calculate similarity matrices for both representations
                            raw_sim_matrix = torch.matmul(raw_z_j, raw_z_j.transpose(-2, -1))  # [B, N, N]
                            rep_sim_matrix = torch.matmul(repre_j, repre_j.transpose(-2, -1))  # [B, N, N]
                            
                            # Remove diagonal elements (self-similarity) as they're trivial
                            eye_mask = 1.0 - torch.eye(repre_j.shape[0], device=repre_j.device)
                            raw_sim_matrix = raw_sim_matrix * eye_mask
                            rep_sim_matrix = rep_sim_matrix * eye_mask
                            
                            # Structure loss: MSE between similarity matrices
                            # This preserves the relative similarities between samples
                            #structure_loss = torch.nn.functional.mse_loss(rep_sim_matrix, raw_sim_matrix)
                            structure_loss = torch.norm(rep_sim_matrix - raw_sim_matrix, p='fro') / (rep_sim_matrix.size(0) * rep_sim_matrix.size(1))
                            # Alternative: KL divergence between similarity distributions
                            # This can be used instead of or in addition to MSE
                            # First convert similarities to probabilities via softmax
                            # raw_sim_prob = torch.nn.functional.softmax(raw_sim_matrix / 0.1, dim=-1)
                            # rep_sim_prob = torch.nn.functional.softmax(rep_sim_matrix / 0.1, dim=-1)
                            
                            # # KL divergence
                            # kl_loss = torch.nn.functional.kl_div(
                            #     torch.log(rep_sim_prob + 1e-8),  # Add small epsilon to avoid log(0)
                            #     raw_sim_prob,
                            #     reduction='batchmean'
                            # )
                            
                            # Add structure losses to projection loss with weight factors
                            proj_loss_per += structure_loss  # MSE structure loss
                            #proj_loss_per += 0.5 * kl_loss         # KL structure loss
                        
                    proj_loss += proj_loss_per / raw_z.shape[0]

            # Backpropagate
            loss = loss / (number_of_perflow)
            proj_loss = proj_loss / number_of_perflow
            loss += proj_loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients and accelerate_cfg.max_grad_norm > 0.:
                all_norm = accelerator.clip_grad_norm_(
                    model.parameters(), accelerate_cfg.max_grad_norm
                )
            optimizer.step()
            lr_scheduler.step()
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(data_cfg.params.train.loader.batch_size)).mean()
            train_loss += avg_loss.item() / grad_accu_steps
            if args.enc_type is not None:
                avg_proj_loss = accelerator.gather(proj_loss.repeat(data_cfg.params.train.loader.batch_size)).mean()
            # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
        if accelerator.sync_gradients: 
            if args.use_ema:
                # update_ema(ema_model, deepcopy(model).type(ema_dtype), args.ema_decay)
                update_ema(ema_model, model, args.ema_decay)
                
            progress_bar.update(1)
            global_steps += 1
            if getattr(accelerate_cfg, 'logger', 'wandb') != None:
                accelerator.log({"train_loss": train_loss}, step=global_steps)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_steps)
                if args.enc_type is not None:
                    accelerator.log({"proj_loss": proj_loss}, step=global_steps)
                if accelerate_cfg.max_grad_norm != 0.0:
                    accelerator.log({"grad_norm": all_norm.item()}, step=global_steps)
            train_loss = 0.0
            if global_steps % accelerate_cfg.checkpointing_steps == 0:
                if accelerate_cfg.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(ckptdir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if accelerator.is_main_process and len(checkpoints) >= accelerate_cfg.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - accelerate_cfg.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(ckptdir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(ckptdir, f"checkpoint-{global_steps}")
                if accelerator.is_main_process:
                    os.makedirs(save_path)
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()
                
            if global_steps in accelerate_cfg.checkpointing_steps_list:
                save_path = os.path.join(ckptdir, f"save-checkpoint-{global_steps}")
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()
            
            if global_steps % accelerate_cfg.evaluation_steps == 0:
                ema_model.eval()
                with torch.no_grad():
                    cfg_scale_test = torch.ones(1)
                    cfg_scale_cond_test = cfg_scale_test.expand(noise_test.shape[0]).to(device=device)
                    t_test = torch.zeros_like(cfg_scale_cond_test)

                    with accelerator.autocast():
                        output_test = ema_model(noise_test, t_test, cfg_scale_cond_test, y=y_test, noise=noise_test_list, representation_noise=noise_test)

                    samples = output_test[:, : n_patch_h*n_patch_w]
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        samples = model.module.unpatchify(samples, (H, W))
                    else:
                        samples = model.unpatchify(samples, (H, W))
                    samples = samples.clamp(-1, 1)     
                    
                    if accelerator.is_main_process:
                        torchvision.utils.save_image(samples, os.path.join(f'{workdirnow}', f"images/fitv2_sample_{global_steps}.jpg"), value_range=(-1, 1), normalize=True, scale_each=True)
                    
                    for nfe in [2, 6]:
                        with accelerator.autocast():
                            if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                                output_test = ema_model.module.forward_cfg(noise_test, t_test, 2, number_of_step_perflow=nfe, y=y_test, noise=noise_test_list, representation_noise=noise_test)
                            else:
                                output_test = ema_model.forward_cfg(noise_test, t_test, 2, number_of_step_perflow=nfe, y=y_test, noise=noise_test_list, representation_noise=noise_test)

                        samples = output_test[:, : n_patch_h*n_patch_w]
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            samples = model.module.unpatchify(samples, (H, W))
                        else:
                            samples = model.unpatchify(samples, (H, W))
                        samples = samples.clamp(-1, 1)     

                        if accelerator.is_main_process:
                            torchvision.utils.save_image(samples, os.path.join(f'{workdirnow}', f"images/fitv2_sample_{global_steps}-NFE{nfe}.jpg"), value_range=(-1, 1), normalize=True, scale_each=True)

            if args.eval_fid and global_steps % accelerate_cfg.eval_fid_steps == 0 and global_steps > 0:
                with torch.no_grad():
                    number = 0
                    ema_model.eval()
                    arr_list = []
                    test_fid_batch_size = accelerate_cfg.test_fid_batch_size
                    while args.eval_fid_num_samples > number:
                        if args.multi_scale:
                            latents = torch.randn((test_fid_batch_size, diffusion_cfg.distillation_network_config.params.in_channels, H, W)).to(device=device)
                            HX, WX = H, W
                            for _ in range(2):
                                HX = HX//2
                                WX = WX//2
                                latents = torch.nn.functional.interpolate(latents, size=(HX, WX), mode='bilinear') * 2
                            latents = latents.reshape(test_fid_batch_size, -1, n_patch_h//4, patch_size, n_patch_w//4, patch_size)
                            latents = rearrange(latents, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
                            latents = latents.permute(0, 2, 1)
                        else:
                            latents = torch.randn((test_fid_batch_size, n_patch_h*n_patch_w, (patch_size**2)*diffusion_cfg.distillation_network_config.params.in_channels)).to(device=device)
                        y = torch.randint(0, 10, (test_fid_batch_size,), device=device)

                        with accelerator.autocast():
                            if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                                output_test = ema_model.module.forward_cfg(latents, t_test, accelerate_cfg.test_cfg_scale, number_of_step_perflow=accelerate_cfg.test_nfe, y=y, representation_noise=latents)
                            else:
                                output_test = ema_model.forward_cfg(latents, t_test, accelerate_cfg.test_cfg_scale, number_of_step_perflow=accelerate_cfg.test_nfe, y=y, representation_noise=latents)

                        samples = output_test[:, : n_patch_h*n_patch_w]
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            samples = model.module.unpatchify(samples, (H, W))
                        else:
                            samples = model.unpatchify(samples, (H, W))
                        samples = samples.clamp(-1, 1)     
                        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                        arr = samples.cpu().numpy()
                        arr_list.append(arr)
                        number += arr.shape[0]
                    
                    arr_list = np.concatenate(arr_list, axis=0)
                    if accelerator.is_main_process:
                        mu, sigma = calculate_inception_stats_cifar(arr_list, detector_net=detector_net, detector_kwargs=detector_kwargs, device=device)
                        fid = compute_fid(mu, sigma, ref_mu=mu_ref, ref_sigma=sigma_ref)
                        logger.info(f"FID: {fid}")
                        if getattr(accelerate_cfg, 'logger', 'wandb') != None:
                            accelerator.log({"fid": fid}, step=global_steps)
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        logs = {"step_loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        if global_steps % accelerate_cfg.logging_steps == 0:
            if accelerator.is_main_process:
                logger.info("step="+str(global_steps)+" / total_step="+str(accelerate_cfg.max_train_steps)+", step_loss="+str(logs["step_loss"])+', lr='+str(logs["lr"]))

        if global_steps >= accelerate_cfg.max_train_steps:
            logger.info(f'global step ({global_steps}) >= max_train_steps ({accelerate_cfg.max_train_steps}), stop training!!!')
            break
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()