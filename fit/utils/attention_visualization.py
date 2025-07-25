import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import os

def reshape_attention_maps(attn_maps, grid_size, img_size, patch_size=2):
    """
    Reshape attention maps to match the image grid for visualization.
    
    Args:
        attn_maps (list): List of attention maps from the model [B, heads, seq_len, seq_len]
        grid_size (tuple): The size of the grid (h//patch_size, w//patch_size)
        img_size (tuple): The size of the original image (h, w)
        patch_size (int): The patch size used in the model
        
    Returns:
        list: Reshaped attention maps for visualization
    """
    h, w = img_size
    h_patches, w_patches = grid_size  # Number of patches in height and width
    
    reshaped_maps = []
    for layer_idx, attn in enumerate(attn_maps):
        # Take the first batch item
        if len(attn.shape) == 4:  # [B, heads, seq_len, seq_len]
            attn = attn[0]  # [heads, seq_len, seq_len]
        
        num_heads = attn.shape[0]
        
        # Average across heads if needed
        avg_attn = attn.mean(0)  # [seq_len, seq_len]
        
        # Reshape to grid dimensions
        try:
            grid_attn = avg_attn.reshape(h_patches, w_patches, h_patches, w_patches)
        except:
            print(f"Cannot reshape attention map of shape {avg_attn.shape} to {h_patches}x{w_patches}x{h_patches}x{w_patches}")
            continue
            
        reshaped_maps.append({
            'layer': layer_idx,
            'raw': attn,
            'avg': avg_attn,
            'grid': grid_attn
        })
    
    return reshaped_maps

def plot_attention_map(attn_map, save_path=None, title=None, figsize=(10, 10)):
    """
    Plot an attention map from a specific layer and position.
    
    Args:
        attn_map (torch.Tensor): Attention map to visualize [h_patches, w_patches]
        save_path (str, optional): Path to save the visualization
        title (str, optional): Title for the plot
        figsize (tuple, optional): Figure size
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.cpu().numpy()
    
    # Create a custom colormap going from blue to white to red
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list('bwr', colors, N=100)
    
    plt.figure(figsize=figsize)
    plt.imshow(attn_map, cmap=cmap)
    plt.colorbar(label='Attention Weight')
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_attention_for_position(reshaped_maps, position, grid_size, save_dir=None):
    """
    Visualize attention maps for a specific position across all layers.
    
    Args:
        reshaped_maps (list): List of reshaped attention maps
        position (tuple): Position (i, j) for which to visualize attention
        grid_size (tuple): Size of the grid (h_patches, w_patches)
        save_dir (str, optional): Directory to save visualizations
    """
    i, j = position
    h_patches, w_patches = grid_size
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for layer_data in reshaped_maps:
        layer_idx = layer_data['layer']
        grid_attn = layer_data['grid']
        
        # Get attention weights from position (i, j) to all other positions
        attn_from_pos = grid_attn[i, j]
        
        # Reshape to 2D grid for visualization
        attn_map = attn_from_pos.reshape(h_patches, w_patches)
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"layer_{layer_idx}_position_{i}_{j}.png")
        
        title = f"Layer {layer_idx} - Attention from position ({i}, {j})"
        plot_attention_map(attn_map, save_path, title)

def visualize_attention_heatmap_on_image(image, attn_map, position, patch_size=2, alpha=0.7, save_path=None):
    """
    Overlay attention heatmap on the original image.
    
    Args:
        image (np.ndarray): Original image [H, W, C]
        attn_map (torch.Tensor): Attention map [h_patches, w_patches]
        position (tuple): Position (i, j) to highlight
        patch_size (int): Patch size used in the model
        alpha (float): Transparency level for the overlay
        save_path (str, optional): Path to save the visualization
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.cpu().numpy()
    
    # Convert attention map to heatmap
    attn_min, attn_max = attn_map.min(), attn_map.max()
    attn_map = (attn_map - attn_min) / (attn_max - attn_min)  # Normalize to [0, 1]
    
    # Resize attention map to match image dimensions
    h, w = image.shape[:2]
    attn_map_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap((attn_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Highlight the source position
    i, j = position
    patch_h, patch_w = h // (attn_map.shape[0] * patch_size), w // (attn_map.shape[1] * patch_size)
    start_h, start_w = i * patch_h, j * patch_w
    end_h, end_w = start_h + patch_h, start_w + patch_w
    
    overlay = image.copy()
    cv2.rectangle(overlay, (start_w, start_h), (end_w, end_h), (0, 255, 0), 2)
    
    # Blend the heatmap and the image
    blended = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    
    # Add the source position highlight
    result = cv2.addWeighted(blended, 0.8, overlay, 0.2, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Attention from position ({i}, {j})")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_attention_visualization_example(model, image, t, y, grid, mask, size=None, patch_size=2):
    """
    Create attention visualizations for an example input.
    
    Args:
        model: The FiT model
        image: Input image tensor
        t: Timestep tensor
        y: Class label tensor
        grid: Grid tensor
        mask: Mask tensor
        size: Size tensor
        patch_size: Patch size
        
    Returns:
        tuple: Output and attention maps
    """
    # Enable attention visualization
    model.enable_attention_visualization()
    
    # Run forward pass
    with torch.no_grad():
        output = model(image, t, y, grid, mask, size)
    
    # Get attention maps
    attention_maps = model.get_attention_maps()
    
    # Disable attention visualization to save memory
    model.disable_attention_visualization()
    
    return output, attention_maps

def create_attention_rollout(attention_maps, grid_shape, discard_ratio=0.9):
    """
    Create an attention rollout visualization.
    
    This implementation is based on the paper "Quantifying Attention Flow in Transformers"
    and helps to visualize how attention propagates through the network.
    
    Args:
        attention_maps: List of attention maps
        grid_shape: Shape of the grid (h_patches, w_patches)
        discard_ratio: Ratio of lowest attention weights to discard
        
    Returns:
        torch.Tensor: Attention rollout map
    """
    h_patches, w_patches = grid_shape
    num_patches = h_patches * w_patches
    
    # Create rollout matrix initialized with identity
    rollout = torch.eye(num_patches, device=attention_maps[0].device)
    
    # Process each layer's attention map
    for attn_map in attention_maps:
        # Take average over heads
        layer_attn = attn_map.mean(dim=1)  # [B, seq_len, seq_len]
        
        # Take first batch for simplicity
        layer_attn = layer_attn[0]  # [seq_len, seq_len]
        
        if layer_attn.shape[0] != num_patches:
            continue
            
        # Discard lowest attention weights
        if discard_ratio > 0:
            # Flatten to find kth percentile
            flat_attn = layer_attn.view(-1)
            k = int(flat_attn.shape[0] * discard_ratio)
            threshold = torch.kthvalue(flat_attn, k).values
            
            # Create binary mask for attention above threshold
            mask = (layer_attn > threshold).float()
            
            # Apply mask to attention
            layer_attn = layer_attn * mask
        
        # Dot product with rollout matrix from previous layer
        rollout = torch.matmul(layer_attn, rollout)
    
    # Reshape to grid
    rollout = rollout.reshape(h_patches, w_patches, h_patches, w_patches)
    
    return rollout 