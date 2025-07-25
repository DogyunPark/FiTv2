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
    linear_decrease_division,
    configure_optimizer_with_different_lr,
)

from fit.utils.eval_utils import init_from_ckpt, calculate_inception_stats_imagenet
from fit.utils.lr_scheduler import get_scheduler
from fit.model.fit_model_lwd import FiTBlock
from fit.model.modules_lwd import FinalLayer, PatchEmbedder, TimestepEmbedder, LabelEmbedder, RepresentationBlock
from fit.scheduler.transport.utils import get_flexible_mask_and_ratio, mean_flat

#from came_pytorch import CAME
from PIL import Image
#from elatentlpips import ELatentLPIPS
from fit.scheduler.transport.utils import loss_func_huber
from fit.utils.utils import bell_shaped_sample, symmetric_segment_division, linear_increase_division, sample_posterior
from fit.utils.evaluator import Evaluator
from fit.utils.utils import preprocess_raw_image, load_encoders
from fit.data.dataset import CustomDataset
from fit.data.in1k_latent_dataset import get_train_sampler
import tensorflow.compat.v1 as tf


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
        default=0.9999,
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
        "--enc_type",
        type=str,
        default=None,
        help="The type of encoder to use."
    )
    parser.add_argument(
        "--multi_scale",
        action="store_true",
        default=False,
        help="Whether to use multi-scale."
    )
    parser.add_argument(
        "--structured_loss",
        action="store_true",
        default=False,
        help="Whether to use structured loss."
    )
    parser.add_argument(
        "--number_of_representation_blocks",
        type=int,
        default=None,
        help="The number of representation blocks."
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
            auto_wrap_policy = ModuleWrapPolicy([FiTBlock, RepresentationBlock]),
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
    
    # if args.seed is not None:
    #     set_seed(args.seed)

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
        pretrained_model = instantiate_from_config(diffusion_cfg.network_config).to(device=device)
        init_from_ckpt(pretrained_model, checkpoint_dir=args.pretrain_ckpt, ignore_keys=None, verbose=True)
        pretrained_model.eval()

    model = instantiate_from_config(diffusion_cfg.distillation_network_config).to(device=device)
    #init_from_ckpt(model, checkpoint_dir=args.pretrain_ckpt2, ignore_keys='y_embedder', verbose=True)
    # if torch.__version__ >= '2.0.0':
    #     logger.info("Using torch.compile to optimize the model")
    #     compile_mode = "max-autotune"  # Try this first for faster startup
    #     for block in model.blocks:
    #         block.attn.forward = torch.compile(
    #             block.attn.forward,
    #             backend="inductor", 
    #             mode=compile_mode
    #         )

    # if torch.__version__ >= '2.0.0':
    #     logger.info("Using torch.compile to optimize the model")
    #     # You can choose different backends: 'inductor' (default), 'aot_eager', 'cudagraphs'
    #     # mode options: 'default', 'reduce-overhead', 'max-autotune'
    #     compile_kwargs = {
    #         "backend": "inductor",
    #         "mode": "reduce-overhead",
    #         "fullgraph": False,  # Set to True for full graph optimization if your model supports it
    #     }
    #     model = torch.compile(model)
    #     logger.info(f"Model compiled with settings: {compile_kwargs}")
    # else:
    #     logger.info("PyTorch version < 2.0, torch.compile not available")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad) 
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Fixed parameters: {fixed_params}")

    number_of_perflow = args.number_of_perflow
    number_of_layers_for_perflow = args.number_of_layers_for_perflow
    number_of_representation_blocks = args.number_of_representation_blocks
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
    #sigmas = torch.linspace(0, 1, 3+1).to(device)
    sigmas = torch.linspace(0, 1, number_of_perflow+1).to(device)
    #sigmas = linear_decrease_division(number_of_perflow)
    #sigmas = symmetric_segment_division(number_of_perflow)
    sigmas = sigmas.to(device=device)
    logger.info(f"Sigmas: {sigmas}")

    # update ema
    if args.use_ema:
        # ema_dtype = torch.float32
        if hasattr(model, 'module'):
            ema_model = deepcopy(model.module).to(device=device)
        else:
            ema_model = deepcopy(model).to(device=device)
        
        # if torch.__version__ >= '2.0.0':
        #     ema_model = torch.compile(ema_model, **compile_kwargs)
        #     logger.info("EMA model compiled")

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

    # get_train_dataloader = instantiate_from_config(data_cfg)
    # train_len = get_train_dataloader.train_len()
    # train_dataloader = get_train_dataloader.train_dataloader(
    #     global_batch_size=total_batch_size, max_steps=accelerate_cfg.max_train_steps, 
    #     resume_step=global_steps, seed=args.seed
    # )
    train_dataset = CustomDataset(data_cfg.params.train.data_path)
    train_sampler = get_train_sampler(train_dataset, total_batch_size, accelerate_cfg.max_train_steps, global_steps, args.seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=data_cfg.params.train.loader.batch_size, sampler=train_sampler, num_workers=data_cfg.params.train.loader.num_workers, pin_memory=True, drop_last=True)
    # train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset, 
    #         batch_size=data_cfg.params.train.loader.batch_size, 
    #         sampler=train_sampler, 
    #         num_workers=min(os.cpu_count() - 1, 8),  # Optimize worker count
    #         pin_memory=True,
    #         persistent_workers=True,  # Keep workers alive between epochs
    #         prefetch_factor=2,  # Prefetch batches
    #         drop_last=True
    #     )

    # Setup optimizer and lr_scheduler
    # if accelerator.is_main_process:
    #     for name, param in model.named_parameters():
    #         print(name, param.requires_grad)
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

    #optimizer = configure_optimizer_with_different_lr(model, base_lr=learning_rate, rep_lr_factor=1.0, blocks_lr_factor=2.0, **optimizer_cfg.get("params", dict()))
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
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and getattr(accelerate_cfg, 'logger', 'wandb') != None:
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), workdirnow)
        accelerator.init_trackers(
            args.main_project_name, 
            #config=config, 
            init_kwargs={"wandb": {"group": args.project_name}}
        )

   
    vae_model = 'stabilityai/sd-vae-ft-ema'
    vae = AutoencoderKL.from_pretrained(vae_model, local_files_only=False).to(device)
    vae.eval() # important

    if args.use_elpips:
        elatentlpips = ELatentLPIPS(encoder="sd15", augment='bg').to(device).eval()
    
    if accelerator.is_main_process and args.eval_fid:
        hf_config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        hf_config.gpu_options.allow_growth = True
        hf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
        evaluator = Evaluator(tf.Session(config=hf_config), batch_size=20)
        ref_acts = evaluator.read_activations_npz(args.ref_path)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_path, ref_acts)
            
    torch.cuda.empty_cache()
    tf.reset_default_graph()

    if args.enc_type is not None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, 256
            )
        # encoders2, encoder_types2, architectures2 = load_encoders(
        #     'jepa-vit-h', device, 256
        #     )
    # Train!
    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {train_len}")
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
                #accelerator.load_state(os.path.join(ckptdir, ' LWD/fitv2_perflow_9/checkpoints/checkpoint-55000'))
                break
            except (RuntimeError, Exception) as err:
                error_times+=1
                if accelerator.is_local_main_process:
                    logger.warning(err)
                    logger.warning(f"Failed to resume from checkpoint {resume_from_path}")
                    #shutil.rmtree(os.path.join(ckptdir, resume_from_path))
                else:
                    time.sleep(2)
    
    #del model.module.representation_norm

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
    
    noise_test = torch.randn((test_batch_size, n_patch_h*n_patch_w, (patch_size**2)*diffusion_cfg.distillation_network_config.params.in_channels)).to(device=device)
    noise_test_list = [torch.randn((test_batch_size, n_patch_h*n_patch_w, (patch_size**2)*diffusion_cfg.distillation_network_config.params.in_channels)).to(device=device) for _ in range(number_of_perflow-1)]
    
    for step, batch in enumerate(train_dataloader, start=global_steps):
        raw_x, x, y = batch
        raw_x = raw_x.to(device)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x_data = sample_posterior(x, vae.config.scaling_factor)
            x = x_data.reshape(x_data.shape[0], -1, n_patch_h, patch_size, n_patch_w, patch_size)
            x = rearrange(x, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
            x = x.permute(0, 2, 1)
        #y = y.to(torch.int)

        cfg_scale = torch.tensor([4.0])
        cfg_scale_cond = cfg_scale.expand(x.shape[0]).to(device=device)

        if args.enc_type is not None:
            with torch.no_grad():
                raw_x = raw_x / 255.
                raw_x = preprocess_raw_image(raw_x, args.enc_type)
                raw_z = encoders[0].forward_features(raw_x)
                if 'dinov2' in args.enc_type:
                    raw_z_cls = raw_z['x_norm_clstoken']
                    raw_z_data = raw_z['x_norm_patchtokens']
        
        loss = 0.0
        proj_loss = 0.0
        block_idx = 0
        per_block_idx = 0
        total_loss = 0.0
        total_proj_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for layer_idx in range(number_of_perflow):
            #print(f'layer_idx: {layer_idx}')
            x0 = torch.randn_like(x)
            #optimizer.zero_grad()
            loss = 0.0
            proj_loss = 0.0
            #layer_idx = torch.randint(0, number_of_perflow, (1,))
            #layer_idx = step % number_of_perflow
            sigma_next = sigmas[layer_idx + 1]
            sigma_current = sigmas[layer_idx]

            #sigma_list = torch.linspace(sigma_current.item(), sigma_next.item(), perflow_solver_step+1)
            
            if args.enc_type is not None:
                raw_z = raw_z_data

            ratio_next = sigma_next.clone()
            while len(ratio_next.shape) < x0.ndim:
                ratio_next = ratio_next.unsqueeze(-1)
            xt = x0 * (1-ratio_next) + x * ratio_next

            ratio = sigma_current.clone()
            while len(ratio.shape) < x0.ndim:
                ratio = ratio.unsqueeze(-1)
            xt_input = x0 * (1-ratio) + x * ratio

            model_kwargs = dict(y=y, target_layer_start=layer_idx * number_of_layers_for_perflow, target_layer_end=(layer_idx+1) * number_of_layers_for_perflow)

            if args.reflow:
                #per_flow_ratio = torch.randint(0, 1000, (x.shape[0],)) / 1000
                per_flow_ratio = torch.rand(x.shape[0]).to(device=device)
                #per_flow_ratio = per_flow_ratio.to(device=device)
                #per_flow_ratio = torch.rand(x.shape[0]).to(device=device)
                t_input = sigma_current + per_flow_ratio.clone() * (sigma_next - sigma_current)
                while len(per_flow_ratio.shape) < x0.ndim:
                    per_flow_ratio = per_flow_ratio.unsqueeze(-1)
                x_input = xt_input * (1-per_flow_ratio) + xt * per_flow_ratio
                target = (xt - xt_input) / (sigma_next - sigma_current)
                weight = 1 #/ (sigma_next - sigma_current)
        
            # save memory for x, grid, mask
            # forward model and compute loss
            with accelerator.autocast():
                #_, _ = get_flexible_mask_and_ratio(model_kwargs, x)
                #if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                pred_model, representation_linear, dummy_loss = model(x_input, t_input, cfg_scale_cond, **model_kwargs)
            
            losses = mean_flat((((pred_model - target)) ** 2)) * weight
            loss_mean = losses.mean()

            if args.enc_type is not None:
                proj_loss_per = 0.0
                #if layer_idx <= int(number_of_perflow/2):
                for j, (repre_j, raw_z_j) in enumerate(zip(representation_linear, raw_z)):
                    raw_z_j = torch.nn.functional.normalize(raw_z_j, dim=-1) 
                    repre_j = torch.nn.functional.normalize(repre_j, dim=-1) 
                    proj_loss_per += mean_flat(-(raw_z_j * repre_j).sum(dim=-1))
                proj_loss_mean = proj_loss_per / raw_z.shape[0]
                #proj_loss += proj_loss_mean

            # Backpropagate
            loss += loss_mean / number_of_perflow + dummy_loss
            proj_loss += proj_loss_mean / number_of_perflow
            #loss += loss_mean
            total_loss += loss_mean
            loss += 0.05 * proj_loss
            total_proj_loss += proj_loss_mean
            accelerator.backward(loss)

        #loss = loss / number_of_perflow
        #proj_loss = proj_loss / number_of_perflow
        total_loss = total_loss / number_of_perflow
        total_proj_loss = total_proj_loss / number_of_perflow
        if accelerator.sync_gradients and accelerate_cfg.max_grad_norm > 0.:
            all_norm = accelerator.clip_grad_norm_(
                model.parameters(), accelerate_cfg.max_grad_norm
            )
        optimizer.step()
        # Gather the losses across all processes for logging (if we use distributed training).
        #total_loss = total_loss / number_of_perflow
        #total_proj_loss = total_proj_loss / number_of_perflow
        avg_loss = accelerator.gather(total_loss.repeat(data_cfg.params.train.loader.batch_size)).mean()
        train_loss += avg_loss.item() / grad_accu_steps
        if args.enc_type is not None:
            avg_proj_loss = accelerator.gather(total_proj_loss.repeat(data_cfg.params.train.loader.batch_size)).mean()

        # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
        if accelerator.sync_gradients: 
            if args.use_ema:
                update_ema(ema_model, model, args.ema_decay)
                
            progress_bar.update(1)
            global_steps += 1
            if getattr(accelerate_cfg, 'logger', 'wandb') != None:
                accelerator.log({"train_loss": train_loss}, step=global_steps)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_steps)
                if args.enc_type is not None:
                    accelerator.log({"proj_loss": avg_proj_loss}, step=global_steps)
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
                model.eval()
                with torch.no_grad():
                    # prepare for x
                    cfg_scale_test = torch.ones(1)
                    cfg_scale_cond_test = cfg_scale_test.expand(noise_test.shape[0]).to(device=device)
                    t_test = torch.zeros_like(cfg_scale_cond_test)

                    # with accelerator.autocast():
                    #     output_test = ema_model(noise_test, t_test, cfg_scale_cond_test, y=y_test, noise=noise_test_list, representation_noise=noise_test)

                    # samples = output_test[..., : n_patch_h*n_patch_w]
                    # if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                    #     samples = ema_model.module.unpatchify(samples, (H, W))
                    # else:
                    #     samples = ema_model.unpatchify(samples, (H, W))
                    # samples = samples.to(torch.bfloat16)     
                    # samples = vae.decode(samples / vae.config.scaling_factor).sample
                    # samples = samples.clamp(-1, 1)
                    # if accelerator.is_main_process:
                    #     torchvision.utils.save_image(samples, os.path.join(f'{workdirnow}', f"images/fitv2_sample_{global_steps}.jpg"), normalize=True, scale_each=True)
                    
                    for nfe in [6]:
                        with accelerator.autocast():
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                                output_test = model.module.forward_cfg(noise_test, t_test, 2, number_of_step_perflow=nfe, y=y_test, noise=noise_test_list, representation_noise=noise_test)
                            else:
                                output_test = model.forward_cfg(noise_test, t_test, 2, number_of_step_perflow=nfe, y=y_test, noise=noise_test_list, representation_noise=noise_test)

                        samples = output_test[:, : n_patch_h*n_patch_w]
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            samples = model.module.unpatchify(samples, (H, W))
                        else:
                            samples = model.unpatchify(samples, (H, W))
                            
                        #samples = samples.to(torch.bfloat16)     
                        samples = vae.decode(samples / vae.config.scaling_factor).sample
                        samples = samples.clamp(-1, 1)
                        if accelerator.is_main_process:
                            torchvision.utils.save_image(samples, os.path.join(f'{workdirnow}', f"images/fitv2_sample_{global_steps}.jpg"), normalize=True, scale_each=True)

                    for nfe in [6]:
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
                            
                        #samples = samples.to(torch.bfloat16)     
                        samples = vae.decode(samples / vae.config.scaling_factor).sample
                        samples = samples.clamp(-1, 1)
                        if accelerator.is_main_process:
                            torchvision.utils.save_image(samples, os.path.join(f'{workdirnow}', f"images/fitv2_sample_{global_steps}-NFE{nfe}.jpg"), normalize=True, scale_each=True)
                torch.cuda.empty_cache()
                model.train()
                
            if args.eval_fid and global_steps % accelerate_cfg.eval_fid_steps == 0 and global_steps > 0:
                with torch.no_grad():
                    number = 0
                    arr_list = []
                    test_fid_batch_size = accelerate_cfg.test_fid_batch_size

                    while args.eval_fid_num_samples > number:
                        
                        latents = torch.randn((test_fid_batch_size, n_patch_h*n_patch_w, (patch_size**2)*diffusion_cfg.distillation_network_config.params.in_channels)).to(device=device)
                        y = torch.randint(0, 1000, (test_fid_batch_size,), device=device)

                        with accelerator.autocast():
                            if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                                output_test = ema_model.module.forward_wo_cfg(latents, t_test, accelerate_cfg.test_cfg_scale, number_of_step_perflow=accelerate_cfg.test_nfe, y=y, representation_noise=latents)
                            else:
                                output_test = ema_model.forward_wo_cfg(latents, t_test, accelerate_cfg.test_cfg_scale, number_of_step_perflow=accelerate_cfg.test_nfe, y=y, representation_noise=latents)

                        samples = output_test[:, : n_patch_h*n_patch_w]
                        if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                            samples = ema_model.module.unpatchify(samples, (H, W))
                        else:
                            samples = ema_model.unpatchify(samples, (H, W))
                        #samples = samples.to(torch.bfloat16)     
                        samples = vae.decode(samples / vae.config.scaling_factor).sample
                        samples = samples.clamp(-1, 1)
                        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                        arr = samples.cpu().numpy()
                        arr_list.append(arr)
                        number += arr.shape[0]
                    
                    arr_list = np.concatenate(arr_list, axis=0)
                    if accelerator.is_main_process:
                        sample_acts, sample_stats, sample_stats_spatial = calculate_inception_stats_imagenet(arr_list, evaluator)
                        inception_score = evaluator.compute_inception_score(sample_acts[0])
                        fid = sample_stats.frechet_distance(ref_stats)
                        sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
                        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
                        logger.info(f"Inception Score: {inception_score}")
                        logger.info(f"FID: {fid}")
                        logger.info(f"Spatial FID: {sfid}")
                        logger.info(f"Precision: {prec}")
                        logger.info(f"Recall: {recall}")
                        if getattr(accelerate_cfg, 'logger', 'wandb') != None:
                            accelerator.log({"inception_score": inception_score}, step=global_steps)
                            accelerator.log({"fid": fid}, step=global_steps)
                            accelerator.log({"sfid": sfid}, step=global_steps)
                            accelerator.log({"prec": prec}, step=global_steps)
                            accelerator.log({"recall": recall}, step=global_steps)
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        logs = {"step_loss": total_loss.detach().item(), 
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