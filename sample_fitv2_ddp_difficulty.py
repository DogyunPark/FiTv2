# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import os
import sys
import math
import torch
import time
import argparse
import numpy as np
import torch.distributed as dist
import re
import torchvision
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from diffusers.models import AutoencoderKL
#from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from flow_match_scheduler import FlowMatchEulerDiscreteScheduler
from fit.scheduler.transport import create_transport, Sampler
from fit.utils.eval_utils import create_npz_from_sample_folder, init_from_ckpt
from fit.utils.utils import instantiate_from_config
from fit.utils.sit_eval_utils import parse_sde_args, parse_ode_args
from fvcore.nn import FlopCountAnalysis, flop_count_table
from fit.utils.measure import compute_spectral_entropy, compute_ssim, compute_pixelwise_variance, compute_gradient_magnitude, compute_mutual_information, high_frequency_ratio

def ntk_scaled_init(head_dim, base=10000, alpha=8):
    #The method is just these two lines
    dim_h = head_dim // 2 # for x and y
    base = base * alpha ** (dim_h / (dim_h-2)) #Base change formula
    return base

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.mixed == "fp32":
        weight_dtype = torch.float32
    elif args.mixed == "bf16":
        weight_dtype = torch.bfloat16
    
    if args.cfgdir == "":
        args.cfgdir = os.path.join(args.ckpt.split("/")[0], args.ckpt.split("/")[1], "configs/config.yaml")
    print("config dir: ",args.cfgdir)
    config = OmegaConf.load(args.cfgdir)
    config_diffusion = config.diffusion
    
    
    H, W = args.image_height // 8, args.image_width // 8
    patch_size = config_diffusion.network_config.params.patch_size
    n_patch_h, n_patch_w = H // patch_size, W // patch_size
    
    if args.interpolation != 'no':
        if args.interpolation == 'linear':    # 这个就是positional index interpolation，原来叫normal，现在叫linear
            config_diffusion.network_config.params['custom_freqs'] = 'linear'
        elif args.interpolation == 'dynntk':    # 这个就是ntk-aware
            config_diffusion.network_config.params['custom_freqs'] = 'ntk-aware'
        elif args.interpolation == 'ntkpro1':
            config_diffusion.network_config.params['custom_freqs'] = 'ntk-aware-pro1'
        elif args.interpolation == 'ntkpro2':
            config_diffusion.network_config.params['custom_freqs'] = 'ntk-aware-pro2'
        elif args.interpolation == 'partntk':   # 这个就是ntk-by-parts
            config_diffusion.network_config.params['custom_freqs'] = 'ntk-by-parts'
        elif args.interpolation == 'yarn':
            config_diffusion.network_config.params['custom_freqs'] = 'yarn'
        else:
            raise NotImplementedError
        config_diffusion.network_config.params['max_pe_len_h'] = n_patch_h
        config_diffusion.network_config.params['max_pe_len_w'] = n_patch_w
        config_diffusion.network_config.params['decouple'] = args.decouple
        config_diffusion.network_config.params['ori_max_pe_len'] = int(args.ori_max_pe_len)
        
        config_diffusion.network_config.params['online_rope'] = False
        
    else:   # there is no need to do interpolation!
        config_diffusion.network_config.params['custom_freqs'] = 'normal'
        config_diffusion.network_config.params['online_rope'] = False
        
        
    
    model = instantiate_from_config(config_diffusion.network_config).to(device, dtype=weight_dtype)
    init_from_ckpt(model, checkpoint_dir=args.ckpt, ignore_keys=None, verbose=True)
    model.eval() # important
    
    # prepare first stage model
    if args.vae_decoder == 'sd-ft-mse':
        vae_model = 'stabilityai/sd-vae-ft-mse'
    elif args.vae_decoder == 'sd-ft-ema':
        vae_model = 'stabilityai/sd-vae-ft-ema'
    vae = AutoencoderKL.from_pretrained(vae_model, local_files_only=False).to(device, dtype=weight_dtype)
    vae.eval() # important
    
    scheduler = FlowMatchEulerDiscreteScheduler(invert_sigmas=True)
    
    # prepare transport
    transport = create_transport(**OmegaConf.to_container(config_diffusion.transport))  # default: velocity; 
    sampler = Sampler(transport)
    sampler_mode = args.sampler_mode
    if sampler_mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.ode_sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.ode_sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
    elif sampler_mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sde_sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    else:
        raise NotImplementedError
    

    # Calculate GFLOPs before starting sampling
    if rank == 0:
        print("Measuring model GFLOPs...")
        # Create sample inputs for GFLOPs calculation
        sample_batch_size = 1
        sample_z = torch.randn(
            (sample_batch_size, n_patch_h*n_patch_w, (patch_size**2)*model.in_channels)
        ).to(device=device, dtype=weight_dtype)
        sample_y = torch.randint(0, args.num_classes, (sample_batch_size,), device=device)
        
        # Prepare grid, mask, size
        grid_h = torch.arange(n_patch_h, dtype=torch.long)
        grid_w = torch.arange(n_patch_w, dtype=torch.long)
        grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
        grid = torch.cat(
            [grid[0].reshape(1,-1), grid[1].reshape(1,-1)], dim=0
        ).repeat(sample_batch_size,1,1).to(device=device, dtype=torch.long)
        mask = torch.ones(sample_batch_size, n_patch_h*n_patch_w).to(device=device, dtype=weight_dtype)
        size = torch.tensor((n_patch_h, n_patch_w)).repeat(sample_batch_size,1).to(device=device, dtype=torch.long)
        size = size[:, None, :]
        
        # Define model wrapper for single forward pass GFLOPs
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                timestep = torch.zeros(x.shape[0], device=x.device)
                return self.model(x, timestep, y=sample_y, grid=grid, mask=mask, size=size)
        
        # Define sampling wrapper for total GFLOPs
        class SamplingWrapper(torch.nn.Module):
            def __init__(self, model, num_steps):
                super().__init__()
                self.model = model
                self.num_steps = num_steps
                
            def forward(self, x):
                z = x
                for idx in range(self.num_steps):
                    timestep = torch.zeros(z.shape[0], device=z.device)
                    noise_pred = self.model(z, timestep, y=sample_y, grid=grid, mask=mask, size=size)
                    z = z + (1/self.num_steps) * noise_pred  # Simplified update rule for FLOPs calculation
                return z
        
        # Calculate single forward pass GFLOPs
        wrapper = ModelWrapper(model)
        flops = FlopCountAnalysis(wrapper, sample_z)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total_flops = flops.total()
        print(f"Single forward pass GFLOPs: {total_flops/1e9:.4f}")
        torch.cuda.empty_cache()

        for steps in [2, 4]:
            sampling_wrapper = SamplingWrapper(model, steps)
            sampling_flops = FlopCountAnalysis(sampling_wrapper, sample_z)
            sampling_flops.unsupported_ops_warnings(False)
            sampling_flops.uncalled_modules_warnings(False)
            total_sampling_flops = sampling_flops.total()
            print(f"Sampling GFLOPs (Steps={steps}): {total_sampling_flops/1e9:.4f}")
            torch.cuda.empty_cache()

    workdir_name = 'official_fit'
    folder_name = f'{args.ckpt.split("/")[-1].split(".")[0]}'

    
    # sample_folder_dir = f"{args.sample_dir}/{workdir_name}/{folder_name}"
    # if rank == 0:
    #     os.makedirs(os.path.join(args.sample_dir, workdir_name), exist_ok=True)
    #     os.makedirs(sample_folder_dir, exist_ok=True)
    #     print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()
    args.cfg_scale = float(args.cfg_scale)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    index = 0
    all_images = []
    times = 0
    #for sampling_steps in [1, 2, 12, 24]:
    sampling_steps = args.num_sampling_steps
    spectral_entropies = []
    hf_ratios = []
    timesteps_list = []
    if 1:
        scheduler.set_timesteps(sampling_steps, device=device)
        timesteps = scheduler.timesteps
        print("timesteps: ", timesteps, flush=True)
        print(device, "device: ", index, flush=True)

        index+=1
        # Sample inputs:
        #import pdb; pdb.set_trace()
        z = torch.randn(
            (n, n_patch_h*n_patch_w, (patch_size**2)*model.in_channels)
        ).to(device=device, dtype=weight_dtype)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        
        # prepare for x
        grid_h = torch.arange(n_patch_h, dtype=torch.long)
        grid_w = torch.arange(n_patch_w, dtype=torch.long)
        grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
        grid = torch.cat(
            [grid[0].reshape(1,-1), grid[1].reshape(1,-1)], dim=0
        ).repeat(n,1,1).to(device=device, dtype=torch.long)
        mask = torch.ones(n, n_patch_h*n_patch_w).to(device=device, dtype=weight_dtype)
        size = torch.tensor((n_patch_h, n_patch_w)).repeat(n,1).to(device=device, dtype=torch.long)
        size = size[:, None, :]
        # Setup classifier-free guidance:
        if using_cfg:
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)       # (B,) -> (2B, )
            grid = torch.cat([grid, grid], 0)   # (B, 2, N) -> (2B, 2, N)
            mask = torch.cat([mask, mask], 0)   # (B, N) -> (2B, N)
            size = torch.cat([size, size], 0)
            #model_kwargs = dict(y=y, grid=grid, mask=mask, size=size, cfg_scale=args.cfg_scale, scale_pow=args.scale_pow)
            model_kwargs = dict(y=y, grid=grid, mask=mask, size=size)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y, grid=grid, mask=mask, size=size)
            model_fn = model.forward

        sigmas = torch.linspace(0, 1, sampling_steps+1).to(device)

        # Sample images:
        # timesteps = reversed(timesteps)
        #for idx, t in enumerate(sigmas):
        #sampling_start = time.time()
        for idx in range(len(sigmas)-1):
            if using_cfg:
                z_input = torch.cat([z, z], 0)            # (B, N, patch_size**2 * C) -> (2B, N, patch_size**2 * C)
            else:
                z_input = z


            sigma_current = sigmas[idx]
            sigma_next = sigmas[idx+1]
            timestep = sigma_current.expand(z_input.shape[0]).to(z_input.device)
            #print("timestep: ", timestep, flush=True)
            noise_pred = model(z_input, timestep, **model_kwargs)
            #noise_pred, rest = noise_pred[:, :, :3*patch_size*patch_size:], noise_pred[:, :, 3*patch_size*patch_size:]
            if using_cfg:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred_cond - noise_pred_uncond)
            #noise_pred = torch.cat([noise_pred, rest], dim=2)
            z = z + (sigma_next - sigma_current) * noise_pred

            if idx in [0, 20, 40, 60, 80, 99]:
                samples = z[..., : n_patch_h*n_patch_w]
                samples = model.unpatchify(samples, (H, W))        
                samples = vae.decode(samples / vae.config.scaling_factor).sample
                #samples = samples.clamp(-1, 1) # B C H W
                torchvision.utils.save_image(samples, f'/hub_data2/dogyun/noisy_images/noisy_image_t{idx}.png', normalize=True, scale_each=True)

                entropy = compute_spectral_entropy(samples.cpu())  # (B,)
                mean_entropy = entropy.mean().item()
                spectral_entropies.append(mean_entropy)

                hf_ratio = high_frequency_ratio(samples.cpu())
                hf_ratios.append(hf_ratio)

                timesteps_list.append(sigma_current.cpu().item())

        samples = z[..., : n_patch_h*n_patch_w]
        samples = model.unpatchify(samples, (H, W))        
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        samples = samples.clamp(-1, 1) # B C H W
        torchvision.utils.save_image(samples, f"/hub_data2/dogyun/samples-{sampling_steps}.jpg", normalize=True, scale_each=True)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        times += 1

        # gather samples
        # gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in [samples]])
        
        entropy_change_rate = []
        for i in range(1, len(spectral_entropies)):
            dt = -(timesteps_list[i] - timesteps_list[i-1])
            de = spectral_entropies[i] - spectral_entropies[i-1]
            entropy_change_rate.append(de / dt)
        # Add a value for the first point (can use forward difference)
        if len(spectral_entropies) > 1:
            entropy_change_rate.insert(0, entropy_change_rate[0])
        else:
            entropy_change_rate.append(0)  # Handle edge case with only one point
        entropy_change_rate = np.array(entropy_change_rate)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot spectral entropy and its change rate
        ax1.plot(timesteps_list, spectral_entropies, 'b-o', linewidth=2, label='Spectral Entropy')
        ax1.set_xlabel('Timestep (t)', fontsize=12)
        ax1.set_ylabel('Spectral Entropy', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Spectral Entropy vs Timestep', fontsize=14)

        # Add entropy change rate on secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(timesteps_list, entropy_change_rate, 'r--s', linewidth=2, label='Change Rate')
        ax1_twin.set_ylabel('Change Rate of Entropy', color='r', fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor='r')

        # Add legend for both plots on the first subplot
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        # Plot high frequency ratio
        ax2.plot(timesteps_list, hf_ratios, 'm-o', linewidth=2)
        ax2.set_xlabel('Timestep (t)', fontsize=12)
        ax2.set_ylabel('High Frequency Ratio', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('High Frequency Ratio vs Timestep', fontsize=14)

        plt.tight_layout()
        plt.savefig('/hub_data2/dogyun/spectral_entropy_vs_timesteps.png', dpi=300, bbox_inches='tight')
        plt.close()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgdir",  type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--sample-dir", type=str, default="workdir/eval")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae-decoder", type=str, choices=['sd-ft-mse', 'sd-ft-ema'], default='sd-ft-ema')
    parser.add_argument("--cfg-scale",  type=str, default='1.0')
    parser.add_argument("--scale-pow",  type=float, default=0.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--interpolation", type=str, choices=['no', 'linear', 'yarn', 'dynntk', 'partntk', 'ntkpro1', 'ntkpro2'], default='no') # interpolation
    parser.add_argument("--ori-max-pe-len", default=None, type=int)
    parser.add_argument("--decouple", default=False, action="store_true") # interpolation
    parser.add_argument("--sampler-mode", default='SDE', choices=['SDE', 'ODE'])
    parser.add_argument("--tf32", action='store_true', default=True)
    parser.add_argument("--mixed", type=str, default="fp32")
    parser.add_argument("--save-images", action='store_true', default=False)
    parse_ode_args(parser)
    parse_sde_args(parser)
    args = parser.parse_args()
    main(args)
