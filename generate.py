# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from omegaconf import OmegaConf
from fit.utils.eval_utils import init_from_ckpt
from fit.utils.utils import instantiate_from_config

@torch.no_grad()
def from_sample_posterior(moments, latents_scale, latents_bias):
    device = moments.device
    z = (moments / latents_scale) + latents_bias
    return z

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.precision.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    model = instantiate_from_config(args.network_config).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = f"{args.sample_dir}/checkpoints/save-0{args.iteration}/model_1.safetensors"
    # ckpt_path = args.ckpt
    init_from_ckpt(model, ckpt_path)
    model.eval()  # important!

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", local_files_only=False).to(device, dtype=torch.bfloat16)
    vae.eval() # important
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    ckpt_string_name = os.path.basename(ckpt_path).replace(".safetensors", "") if ckpt_path else "pretrained"
    folder_name = f"{ckpt_string_name}-size-{args.resolution}-vae-" \
                  f"cfg-{args.cfg_scale}-seed-{args.seed.global_seed}-{args.sampling.mode}-{args.iteration}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    patch_size = args.network_config.params.patch_size
    n_patch_h = args.network_config.params.n_patch_h
    n_patch_w = args.network_config.params.n_patch_w
    H, W = n_patch_h * patch_size, n_patch_w * patch_size
    n = args.sampling.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.sampling.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        latents = torch.randn((n, n_patch_h*n_patch_w, (patch_size**2)*args.network_config.params.in_channels), device=device)
        y = torch.randint(0, 1000, (n,), device=device)

        with torch.inference_mode():
            output_test = model.forward_wo_cfg(latents, number_of_step_perflow=41, y=y)
            samples = model.unpatchify(output_test, (H, W))
            samples = from_sample_posterior(samples, vae.config.scaling_factor, vae.config.shift_factor)
            samples = vae.decode(samples.to(torch.bfloat16)).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.sampling.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.empty_cache()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--config', type=str)
    parser.add_argument('--sample_dir', type=str)
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.sample_dir = args.sample_dir
    config.iteration = args.iteration
    config.cfg_scale = args.cfg_scale
    return config

if __name__ == "__main__":
    args = parse_args()
    main(args)
