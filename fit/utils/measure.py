import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from sklearn.feature_selection import mutual_info_regression
from scipy.ndimage import gaussian_filter

def high_frequency_ratio(image_tensor, threshold_ratio=0.5):
    """
    Compute the fraction of energy in the high frequency band of the image.
    
    Parameters:
    - image: 2D numpy array (grayscale image).
    - threshold_ratio: Relative threshold (0 to 1) that defines the high frequency cutoff.
    
    Returns:
    - hf_ratio: Ratio of high frequency energy to total energy.
    """
    # Convert image to grayscale if needed
    if image_tensor.shape[1] == 3:
        image_tensor = 0.2989 * image_tensor[:,0,:,:] + \
                       0.5870 * image_tensor[:,1,:,:] + \
                       0.1140 * image_tensor[:,2,:,:]  # (B, H, W)
    else:
        image_tensor = image_tensor[:,0,:,:]  # (B, H, W)

    # Compute the 2D Fourier transform and shift zero frequency to the center.
    fft = torch.fft.fft2(image_tensor)  # (B, H, W), complex tensor
    fft_shifted = torch.fft.fftshift(fft)  # Shift zero-frequency to center

    # Compute Power Spectral Density (PSD)
    psd = torch.abs(fft_shifted) ** 2  # (B, H, W)

    # Create a frequency mask: pixels further than threshold_ratio * (max distance) are considered high frequency.
    rows, cols = image_tensor.shape[1:]
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate grids
    y_grid = torch.arange(rows, device=image_tensor.device).view(-1, 1).repeat(1, cols)
    x_grid = torch.arange(cols, device=image_tensor.device).view(1, -1).repeat(rows, 1)
    
    # Calculate distance from center
    distance = torch.sqrt((y_grid - crow)**2 + (x_grid - ccol)**2)
    max_distance = torch.sqrt(torch.tensor(crow**2 + ccol**2, device=image_tensor.device))
    mask = distance > (threshold_ratio * max_distance)
    
    # Compute energy in high frequency components and the total energy
    # Expand mask to match batch dimension
    batch_mask = mask.unsqueeze(0).expand_as(psd)
    hf_energy = torch.sum(psd[batch_mask])
    total_energy = torch.sum(psd)
    hf_ratio = hf_energy / total_energy
    return hf_ratio

def compute_spectral_entropy(image_tensor, eps=1e-8):
    """
    Compute the Spectral Entropy of an image batch.

    Args:
        image_tensor: PyTorch Tensor of shape (B, C, H, W), pixel values in [-1, 1] or [0, 1].
        eps: Small epsilon to prevent log(0).

    Returns:
        entropy: PyTorch Tensor of shape (B,) representing the spectral entropy per image.
    """
    # Convert image to grayscale if needed
    if image_tensor.shape[1] == 3:
        image_tensor = 0.2989 * image_tensor[:,0,:,:] + \
                       0.5870 * image_tensor[:,1,:,:] + \
                       0.1140 * image_tensor[:,2,:,:]  # (B, H, W)
    else:
        image_tensor = image_tensor[:,0,:,:]  # (B, H, W)

    # Compute 2D FFT (frequency domain)
    fft = torch.fft.fft2(image_tensor)  # (B, H, W), complex tensor
    fft_shifted = torch.fft.fftshift(fft)  # Shift zero-frequency to center

    # Compute Power Spectral Density (PSD)
    psd = torch.abs(fft_shifted) ** 2  # (B, H, W)

    # Flatten PSD to compute probability distribution
    psd_flat = psd.view(psd.shape[0], -1)  # (B, H*W)

    # Normalize PSD to form a probability distribution
    psd_sum = psd_flat.sum(dim=-1, keepdim=True)  # (B, 1)
    psd_norm = psd_flat / (psd_sum + eps)

    # Compute spectral entropy
    entropy = -torch.sum(psd_norm * torch.log(psd_norm + eps), dim=-1)  # (B,)

    return entropy  # higher entropy indicates more randomness/difficulty

# SSIM computation (using skimage)
def compute_ssim(img1, img2):
    img1_np = img1.permute(1,2,0).cpu().numpy()
    img2_np = img2.permute(1,2,0).cpu().numpy()
    ssim_val = ssim_metric(img1_np, img2_np, channel_axis=-1, data_range=img1_np.max() - img1_np.min())
    return ssim_val

# --- Gradient Magnitude ---
def compute_gradient_magnitude(images):
    gray_images = 0.2989 * images[:,0,:,:] + \
                  0.5870 * images[:,1,:,:] + \
                  0.1140 * images[:,2,:,:]

    sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).to(images.device)
    sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).to(images.device)

    # Convert sobel filters to float type to match input tensor type
    sobel_x_float = sobel_x.float()
    sobel_y_float = sobel_y.float()
    
    grad_x = F.conv2d(gray_images.unsqueeze(1), sobel_x_float, padding=1)
    grad_y = F.conv2d(gray_images.unsqueeze(1), sobel_y_float, padding=1)

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2).mean().item()
    return grad_mag

# --- Mutual Information ---
def compute_mutual_information(x_clean, x_noisy):
    B = x_clean.size(0)
    x_clean_flat = x_clean.view(B, -1).cpu().numpy()
    x_noisy_flat = x_noisy.view(B, -1).cpu().numpy()

    mi_vals = []
    for i in range(B):
        mi = mutual_info_regression(x_noisy_flat[i].reshape(-1,1), x_clean_flat[i])
        mi_vals.append(np.mean(mi))
    return np.mean(mi_vals)

# --- Pixel-wise Variance ---
def compute_pixelwise_variance(images):
    variance = images.var(dim=0).mean().item()
    return variance