import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image
from torchvision import transforms

from fit.model.fit_model import FiT
from fit.model.utils import get_grid, create_masks
from fit.utils.attention_visualization import (
    create_attention_visualization_example,
    reshape_attention_maps,
    visualize_attention_for_position,
    visualize_attention_heatmap_on_image,
    create_attention_rollout
)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize attention maps in FiT model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='attention_visualizations', help='Directory to save visualizations')
    parser.add_argument('--timestep', type=float, default=0.5, help='Diffusion timestep (between 0 and 1)')
    parser.add_argument('--class_label', type=int, default=0, help='Class label for conditioning')
    parser.add_argument('--position_i', type=int, default=None, help='Row index for attention visualization')
    parser.add_argument('--position_j', type=int, default=None, help='Column index for attention visualization')
    parser.add_argument('--layer', type=int, default=None, help='Layer to visualize (if None, visualize all layers)')
    return parser.parse_args()

def load_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image for the model."""
    image = Image.open(image_path).convert('RGB')
    
    # Save the original image for visualization
    original_image = np.array(image.resize(target_size))
    
    # Preprocess for the model
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, original_image

def load_model(model_path, patch_size=2, save_attention=True):
    """Load the FiT model from checkpoint."""
    # Adjust these parameters to match your model
    model = FiT(
        patch_size=patch_size,
        in_channels=3,  # RGB image
        hidden_size=1152,  # Adjust as needed
        depth=28,  # Adjust as needed
        num_heads=16,  # Adjust as needed
        num_classes=1000,  # Adjust as needed
        save_attention=save_attention,
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    return model

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image and model
    image_tensor, original_image = load_image(args.image_path)
    model = load_model(args.model_path)
    
    # Get image dimensions
    _, _, h, w = image_tensor.shape
    h_patches, w_patches = h // model.patch_size, w // model.patch_size
    
    # Prepare model inputs
    t = torch.tensor([args.timestep]).unsqueeze(0)  # [1, 1]
    y = torch.tensor([args.class_label]).unsqueeze(0)  # [1, 1]
    
    # Create grid and mask
    grid = get_grid(h_patches, w_patches).unsqueeze(0)  # [1, 2, N]
    mask = create_masks(1, h_patches * w_patches)  # [1, N]
    
    # Run model with attention visualization
    output, attention_maps = create_attention_visualization_example(
        model, image_tensor, t, y, grid, mask
    )
    
    # Reshape attention maps for visualization
    reshaped_maps = reshape_attention_maps(
        attention_maps, 
        grid_size=(h_patches, w_patches),
        img_size=(h, w),
        patch_size=model.patch_size
    )
    
    # Set default position if not provided
    if args.position_i is None or args.position_j is None:
        args.position_i = h_patches // 2
        args.position_j = w_patches // 2
    
    # Visualize attention maps for the specified position
    position = (args.position_i, args.position_j)
    
    # If a specific layer is requested, visualize only that layer
    if args.layer is not None:
        if 0 <= args.layer < len(reshaped_maps):
            layer_data = reshaped_maps[args.layer]
            grid_attn = layer_data['grid']
            attn_from_pos = grid_attn[args.position_i, args.position_j]
            attn_map = attn_from_pos.reshape(h_patches, w_patches)
            
            # Visualize as heatmap on the original image
            save_path = os.path.join(args.output_dir, f"layer_{args.layer}_position_{args.position_i}_{args.position_j}_overlay.png")
            visualize_attention_heatmap_on_image(
                original_image, 
                attn_map, 
                position, 
                patch_size=model.patch_size,
                save_path=save_path
            )
            print(f"Saved visualization to {save_path}")
    else:
        # Visualize all layers
        visualize_attention_for_position(
            reshaped_maps, 
            position, 
            (h_patches, w_patches),
            save_dir=args.output_dir
        )
        
        # Create attention rollout visualization
        rollout = create_attention_rollout(attention_maps, (h_patches, w_patches))
        
        # Get attention from the selected position
        rollout_from_pos = rollout[args.position_i, args.position_j]
        rollout_map = rollout_from_pos.reshape(h_patches, w_patches)
        
        # Visualize rollout as heatmap
        save_path = os.path.join(args.output_dir, f"rollout_position_{args.position_i}_{args.position_j}_overlay.png")
        visualize_attention_heatmap_on_image(
            original_image, 
            rollout_map, 
            position, 
            patch_size=model.patch_size,
            save_path=save_path
        )
        print(f"Saved rollout visualization to {save_path}")
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 