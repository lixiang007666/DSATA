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


class FRAInjectedLinearWithHooks(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2=64, collect_features=False):
        super().__init__()

        self.linear_fra = nn.Linear(in_features, out_features, bias)
        self.fra_down = nn.Linear(in_features, r, bias=False)
        self.fra_up = nn.Linear(r, out_features, bias=False)
        self.fra_down2 = nn.Linear(in_features, r2, bias=False)
        self.fra_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale1 = nn.Parameter(torch.tensor(1e-5))
        self.scale2 = nn.Parameter(torch.tensor(1e-5))
        
        self.collect_features = collect_features
        self.in_features = in_features
        self.out_features = out_features

        nn.init.normal_(self.fra_down.weight, std=1 / r**2)
        nn.init.zeros_(self.fra_up.weight)

        nn.init.normal_(self.fra_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.fra_up2.weight)

    def forward(self, input):
        global fra_features
        
        low_dim_feat = self.fra_down(input)
        low_dim_output = self.fra_up(low_dim_feat) * self.scale1
        high_dim_feat = self.fra_down2(input)
        high_dim_output = self.fra_up2(high_dim_feat) * self.scale2
        
        if self.collect_features:
            fra_features['low_dim'].append(low_dim_feat.detach().cpu())
            fra_features['high_dim'].append(high_dim_feat.detach().cpu())
            low_attn = torch.norm(low_dim_output, dim=-1, keepdim=True)
            fra_features['attention_maps_low'].append(low_attn.detach().cpu())
            high_attn = torch.norm(high_dim_output, dim=-1, keepdim=True)
            fra_features['attention_maps_high'].append(high_attn.detach().cpu())
        
        return self.linear_fra(input) + low_dim_output + high_dim_output


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


def inject_trainable_fra_with_hooks(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 4,
    r2: int = 16,
    collect_features: bool = True,
):
    """
    Inject FRA with hooks into model for feature visualization.
    """

    require_grad_params = []
    names = []
    fra_modules = []
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
                    _tmp = FRAInjectedLinearWithHooks(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2,
                        collect_features=collect_features,
                    )
                    _tmp = _tmp.to(device)
                    _tmp.linear_fra.weight = weight
                    if bias is not None:
                        _tmp.linear_fra.bias = bias

                    _module._modules[name] = _tmp
                    fra_modules.append(_tmp)

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

    return require_grad_params, names, fra_modules


def set_fra_collect_features(fra_modules: List[nn.Module], collect: bool):
    for module in fra_modules:
        if hasattr(module, 'collect_features'):
            module.collect_features = collect


def get_fra_features():
    global fra_features
    return fra_features
