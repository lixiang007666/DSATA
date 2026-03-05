import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F

import torch.nn as nn


fra_features = {
    'low_dim': [],
    'high_dim': [],
    'attention_maps_low': [],
    'attention_maps_high': [],
}


def clear_fra_features():
    global fra_features
    fra_features = {
        'low_dim': [],
        'high_dim': [],
        'attention_maps_low': [],
        'attention_maps_high': [],
    }


class FRAInjectedLinear(nn.Module):
    """Feature Refinement Adapter (FRA) - Injected Linear Layer"""
    def __init__(self, in_features, out_features, bias=False, r=4, r2=64):
        super().__init__()

        self.linear_fra = nn.Linear(in_features, out_features, bias)
        self.fra_down = nn.Linear(in_features, r, bias=False)
        self.fra_up = nn.Linear(r, out_features, bias=False)
        self.fra_down2 = nn.Linear(in_features, r2, bias=False)
        self.fra_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale1 = nn.Parameter(torch.tensor(1e-2))
        self.scale2 = nn.Parameter(torch.tensor(1e-2))

        nn.init.normal_(self.fra_down.weight, std=1 / r**2)
        nn.init.zeros_(self.fra_up.weight)

        nn.init.normal_(self.fra_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.fra_up2.weight)

    def forward(self, input):
        return self.linear_fra(input) + self.fra_up(self.fra_down(input)) * self.scale1 + self.fra_up2(self.fra_down2(input)) * self.scale2


class FRAInjectedConv2d(nn.Module):
    """Feature Refinement Adapter (FRA) - Injected Conv2d Layer for CNN backbones like Res2Net"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=False, r=4, r2=64):
        super().__init__()
        
        self.conv_fra = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                   padding, dilation, groups, bias)
        
        # Low-rank adaptation using 1x1 convs
        self.fra_down = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)
        self.fra_up = nn.Conv2d(r, out_channels, kernel_size=1, bias=False)
        self.fra_down2 = nn.Conv2d(in_channels, r2, kernel_size=1, bias=False)
        self.fra_up2 = nn.Conv2d(r2, out_channels, kernel_size=1, bias=False)
        
        self.scale1 = nn.Parameter(torch.tensor(1e-2))
        self.scale2 = nn.Parameter(torch.tensor(1e-2))
        
        nn.init.normal_(self.fra_down.weight, std=1 / r**2)
        nn.init.zeros_(self.fra_up.weight)
        nn.init.normal_(self.fra_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.fra_up2.weight)

    def forward(self, input):
        base_output = self.conv_fra(input)
        
        # Adaptation path
        adapt1 = self.fra_up(self.fra_down(input)) * self.scale1
        adapt2 = self.fra_up2(self.fra_down2(input)) * self.scale2
        
        # Resize adaptation to match base output if needed
        if adapt1.shape[-2:] != base_output.shape[-2:]:
            adapt1 = F.interpolate(adapt1, size=base_output.shape[-2:], mode='bilinear', align_corners=False)
            adapt2 = F.interpolate(adapt2, size=base_output.shape[-2:], mode='bilinear', align_corners=False)
        
        return base_output + adapt1 + adapt2


def inject_trainable_fra(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 4,
    r2: int = 16,
):
    """Inject FRA (Feature Refinement Adapter) into model, and returns FRA parameter groups."""

    require_grad_params = []
    names = []
    device = next(model.parameters()).device

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            injected_count = 0

            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":
                    if injected_count >= 3:
                        break

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = FRAInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2,
                    )
                    _tmp = _tmp.to(device)
                    _tmp.linear_fra.weight = weight
                    if bias is not None:
                        _tmp.linear_fra.bias = bias

                    _module._modules[name] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[name].fra_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].fra_down.parameters())
                    )
                    _module._modules[name].fra_up.weight.requires_grad = True
                    _module._modules[name].fra_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[name].fra_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].fra_down2.parameters())
                    )
                    _module._modules[name].fra_up2.weight.requires_grad = True
                    _module._modules[name].fra_down2.weight.requires_grad = True                    
                    names.append(name)
                    injected_count += 1

    return require_grad_params, names


def inject_trainable_fra_conv(
    model: nn.Module,
    target_replace_module: List[str] = ["Bottle2neck", "BasicConv2d"],
    r: int = 4,
    r2: int = 16,
    max_inject_per_module: int = 2,
):
    """Inject FRA (Feature Refinement Adapter) into Conv2d layers for CNN backbones."""

    require_grad_params = []
    names = []
    device = next(model.parameters()).device

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            injected_count = 0

            for name, _child_module in _module.named_modules():
                if isinstance(_child_module, nn.Conv2d) and _child_module.kernel_size[0] > 1:
                    if injected_count >= max_inject_per_module:
                        break

                    weight = _child_module.weight
                    bias = _child_module.bias
                    
                    _tmp = FRAInjectedConv2d(
                        _child_module.in_channels,
                        _child_module.out_channels,
                        _child_module.kernel_size,
                        _child_module.stride,
                        _child_module.padding,
                        _child_module.dilation,
                        _child_module.groups,
                        _child_module.bias is not None,
                        r,
                        r2,
                    )
                    _tmp = _tmp.to(device)
                    _tmp.conv_fra.weight = weight
                    if bias is not None:
                        _tmp.conv_fra.bias = bias

                    _module._modules[name] = _tmp

                    require_grad_params.extend(list(_tmp.fra_up.parameters()))
                    require_grad_params.extend(list(_tmp.fra_down.parameters()))
                    require_grad_params.extend(list(_tmp.fra_up2.parameters()))
                    require_grad_params.extend(list(_tmp.fra_down2.parameters()))
                    
                    _tmp.fra_up.weight.requires_grad = True
                    _tmp.fra_down.weight.requires_grad = True
                    _tmp.fra_up2.weight.requires_grad = True
                    _tmp.fra_down2.weight.requires_grad = True
                    
                    names.append(name)
                    injected_count += 1

    return require_grad_params, names


def get_fra_features():
    global fra_features
    return fra_features
