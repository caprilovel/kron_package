import torch 
import torch.nn as nn
from models.KronLinear import KronLinear

def get_kronlinear_sparsity(model:nn.Module, threshold=1e-3):
    """get the sparsity of a model which have KronLinear layers, only calculate the sparsity of the s parameter

    Args:
        model (nn.Module): The model to calculate the sparsity
        threshold (float, optional): The threshold to determine the sparsity. Defaults to 1e-3.
    Returns:
        sparsity: The sparsity of the model. Calculated as the number of s parameters that are less than the threshold divided by the total number of s parameters
    """
    sparsity = 0
    total = 0
    for name, module in model._modules.items():
        if isinstance(module, KronLinear):
            if module.s is not None:
                sparsity += torch.sum(torch.abs(module.s) < threshold).item()
                total += module.s.numel()
    return sparsity/total
