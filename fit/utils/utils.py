import importlib
import torch
#from came_pytorch import CAME
from collections import OrderedDict
import numpy as np
from math import exp

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