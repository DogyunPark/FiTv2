from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import re
import os
import scipy
from safetensors.torch import load_file


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    imgs = sorted(os.listdir(sample_dir), key=lambda x: int(x.split('.')[0]))
    print(len(imgs))
    assert len(imgs) >= num
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{imgs[i]}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def init_from_ckpt(
    model, checkpoint_dir, ignore_keys=None, verbose=False
) -> None: 
    if checkpoint_dir.endswith(".safetensors"):
        try:
            model_state_dict=load_file(checkpoint_dir)
        except: # 历史遗留问题，千万别删
            model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    else:
        model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    model_new_ckpt=dict()
    for i in model_state_dict.keys():
        model_new_ckpt[i] = model_state_dict[i]
    keys = list(model_new_ckpt.keys())
    for k in keys:
        if ignore_keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print("Deleting key {} from state_dict.".format(k))
                    del model_new_ckpt[k]
    missing, unexpected = model.load_state_dict(model_new_ckpt, strict=False)
    if verbose:
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    if verbose:
        print("")


def calculate_inception_stats_cifar(arr, detector_net=None, detector_kwargs=None, batch_size=100, device='cpu'):
    num_samples = arr.shape[0]
    count = 0
    mu = torch.zeros([2048], dtype=torch.float64, device=device)
    sigma = torch.zeros([2048, 2048], dtype=torch.float64, device=device)

    for k in range((arr.shape[0] - 1) // batch_size + 1):
        mic_img = arr[k * batch_size: (k + 1) * batch_size]
        mic_img = torch.tensor(mic_img).permute(0, 3, 1, 2).to(device)
        features = detector_net(mic_img, **detector_kwargs).to(torch.float64)
        if count + mic_img.shape[0] > num_samples:
            remaining_num_samples = num_samples - count
        else:
            remaining_num_samples = mic_img.shape[0]
        mu += features[:remaining_num_samples].sum(0)
        sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
        count = count + remaining_num_samples
    assert count == num_samples
    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    return mu, sigma

def calculate_inception_stats_imagenet(arr, evaluator, batch_size=100, device='cpu'):
    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(arr)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
    return sample_acts, sample_stats, sample_stats_spatial
    

def compute_fid(mu, sigma, ref_mu=None, ref_sigma=None):
    m = np.square(mu - ref_mu).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
    fid = m + np.trace(sigma + ref_sigma - s * 2)
    fid = float(np.real(fid))
    return fid