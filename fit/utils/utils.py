import importlib
import torch
import math
#from came_pytorch import CAME
from collections import OrderedDict
from torchvision.datasets.utils import download_url
import numpy as np
from math import exp
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(ema_model, 'module'):
        ema_model = ema_model.module
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def exists(x):
    if x is None:
        return False
    return True

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def bell_shaped_sample(low=0, high=20, peak=10, size=10, sigma=3.0):
    """
    Samples from a (truncated) discrete approximation to a normal distribution
    centered around 'peak' with standard deviation 'sigma'.
    """
    x = np.arange(low, high+1)
    
    # Compute approximate normal PDF values, centered at 'peak'
    # (We’ll just use the formula and then normalize.)
    weights = np.array([
        exp(-0.5 * ((val - peak)/sigma)**2) for val in x
    ], dtype=float)
    
    weights /= weights.sum()  # normalize
    
    samples = np.random.choice(x, size=size, p=weights)
    return samples

def symmetric_segment_division(N):
    indices = torch.arange(1, N + 1)
    center = (N + 1) / 2.0
    
    # Define weights so that they are highest at the ends and lowest at the center.
    # This is done by taking the absolute difference from the center and adding 1.
    # For N=7, weights = [|1-4|+1, |2-4|+1, |3-4|+1, |4-4|+1, |5-4|+1, |6-4|+1, |7-4|+1]
    #            = [4, 3, 2, 1, 2, 3, 4]
    weights = torch.abs(indices - center) + 1.0
    
    # Compute normalized segment lengths so they sum to 1.
    segment_lengths = weights / weights.sum()
    
    # Compute endpoints by taking the cumulative sum of the segment lengths,
    # starting from 0.
    endpoints = torch.cat((torch.tensor([0.0]), torch.cumsum(segment_lengths, dim=0)))
    return endpoints

def linear_increase_division(N):
    weights = torch.arange(1, N+1)
    total_weight = weights.sum()
    segment_length = weights / total_weight
    sigmas = torch.cat((torch.tensor([0.0]), torch.cumsum(segment_length, dim=0)))
    return sigmas


@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        
        elif 'dinov1' == encoder_type:
            import timm
            from models import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(f"/hub_data4/dogyun/checkpoints/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from fit.encoders.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            #with open(f"/hub_data4/dogyun/checkpoints/jepa/IN1K-vit.h.14-300e.pth.tar", "rb") as f:
            state_dict = torch.load("/hub_data4/dogyun/checkpoints/jepa/IN1K-vit.h.14-300e.pth.tar", map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def preprocess_raw_image(x, enc_type):
    #resolution = x.shape[-1]
    resolution = 256
    if 'clip' in enc_type:
        #x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        #x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        #x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        #x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        #x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x