# Example: Measure Spectral Entropy over forward diffusion steps
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from fit.utils.measure import compute_spectral_entropy, compute_ssim, compute_pixelwise_variance, compute_gradient_magnitude, compute_mutual_information, high_frequency_ratio
from fit.data.cifar_dataset import create_cifar10_dataloader
from fit.data.dataset import CustomDataset
from fit.data.in1k_latent_dataset import get_train_sampler

# Assume you have a function that generates noisy images at timestep t
# e.g., noisy_images = forward_diffusion(x0, t)

# Dummy example to illustrate:
def cycle(dl):
    while True:
        for data in dl:
            yield data

batch_size = 50
train_dataset = CustomDataset("datasets2/imagenet_256/")
train_sampler = get_train_sampler(train_dataset, batch_size, 2000000, 0, 42)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)
train_dataloader = cycle(train_dataloader)

batch = next(train_dataloader)
x0 = batch[0] / 255.
x0 = x0 * 2 - 1
#x0 = x0.reshape(x0.shape[0], -1, 32, 32)
#x0 = x0.permute(0, 2, 3, 1)


B, C, H, W = x0.shape
timesteps = torch.linspace(0, 1, steps=10)  # 10 timesteps from clean (0) to noisy (1)
spectral_entropies = []
ssim_changes = []
pixel_variances = []
gradient_magnitudes = []
mutual_infos = []
mutual_changes = []
hf_ratios = []

prev_images = torch.randn(B, C, H, W)
for t in timesteps:
    # Replace this with your diffusion forward function: noisy_images = forward_diffusion(x0, t)
    noisy_images = x0 * (t) + torch.randn(B, C, H, W) * (1-t)

    entropy = compute_spectral_entropy(noisy_images)  # (B,)
    mean_entropy = entropy.mean().item()
    spectral_entropies.append(mean_entropy)

    hf_ratio = high_frequency_ratio(noisy_images)
    hf_ratios.append(hf_ratio)

    # ssim_vals = []
    # ssim_vals = [compute_ssim(noisy_images[i], prev_images[i]) for i in range(batch_size)]
    # ssim_changes.append(1 - np.mean(ssim_vals))

    # Pixel-wise variance across batch
    # variance = compute_pixelwise_variance(noisy_images)
    # pixel_variances.append(variance)
    # gradient_magnitudes.append(compute_gradient_magnitude(noisy_images))
    # mutual_infos.append(compute_mutual_information(x0, noisy_images))
    # mutual_changes.append(compute_mutual_information(prev_images, noisy_images))
    prev_images = noisy_images.clone()

    # Save intermediate noisy images at specific timesteps
    torchvision.utils.save_image(noisy_images, f'/hub_data2/dogyun/noisy_images/noisy_image_t{t:.2f}.png', normalize=True, scale_each=True)

# Compute discrete derivative (change rate)
# Calculate change rate manually
entropy_change_rate = []
for i in range(1, len(spectral_entropies)):
    dt = -(timesteps[i].item() - timesteps[i-1].item())
    de = spectral_entropies[i] - spectral_entropies[i-1]
    entropy_change_rate.append(de / dt)
# Add a value for the first point (can use forward difference)
if len(spectral_entropies) > 1:
    entropy_change_rate.insert(0, entropy_change_rate[0])
else:
    entropy_change_rate.append(0)  # Handle edge case with only one point
entropy_change_rate = np.array(entropy_change_rate)


# # Plot entropy vs timesteps
# fig, ax1 = plt.subplots(figsize=(10,6))
# ax1.plot(timesteps.numpy(), spectral_entropies, 'b-o', label='Spectral Entropy')
# ax1.set_xlabel('Timestep (t)', fontsize=12)
# ax1.set_ylabel('Spectral Entropy', color='b', fontsize=12)
# ax1.tick_params(axis='y', labelcolor='b')
# ax1.grid(True)

# # Plot Change Rate on secondary axis
# ax2 = ax1.twinx()
# ax2.plot(timesteps.numpy(), entropy_change_rate, 'r--s', label='Change Rate')
# ax2.set_ylabel('Change Rate of Entropy', color='r', fontsize=12)
# ax2.tick_params(axis='y', labelcolor='r')

# # Add legend
# fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85), fontsize=12)

# plt.title('Spectral Entropy and Its Change Rate across Diffusion Timesteps', fontsize=14)
# plt.tight_layout()
# plt.savefig('/hub_data2/dogyun/spectral_entropy_vs_timesteps.png', dpi=300, bbox_inches='tight')
# plt.close()

# The subplot size (15, 18) is good for a 3x2 grid with detailed plots
# It provides enough space for each subplot and their labels/legends
#Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot spectral entropy and its change rate
ax1.plot(timesteps.numpy(), spectral_entropies, 'b-o', linewidth=2, label='Spectral Entropy')
ax1.set_xlabel('Timestep (t)', fontsize=12)
ax1.set_ylabel('Spectral Entropy', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)
ax1.set_title('Spectral Entropy vs Timestep', fontsize=14)

# Add entropy change rate on secondary axis
ax1_twin = ax1.twinx()
ax1_twin.plot(timesteps.numpy(), entropy_change_rate, 'r--s', linewidth=2, label='Change Rate')
ax1_twin.set_ylabel('Change Rate of Entropy', color='r', fontsize=12)
ax1_twin.tick_params(axis='y', labelcolor='r')

# Add legend for both plots on the first subplot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# Plot high frequency ratio
ax2.plot(timesteps.numpy(), hf_ratios, 'm-o', linewidth=2)
ax2.set_xlabel('Timestep (t)', fontsize=12)
ax2.set_ylabel('High Frequency Ratio', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_title('High Frequency Ratio vs Timestep', fontsize=14)

plt.tight_layout()
plt.savefig('/hub_data2/dogyun/spectral_entropy_vs_timesteps.png', dpi=300, bbox_inches='tight')
plt.close()