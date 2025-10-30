"""
modules.py

This file contains the layer-by-layer implementation of the ConvNeXt model
architecture, adapted for the ADNI dataset classification task.

Based on the official implementation from:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# --- LayerNorm Implementation ---
class LayerNorm(nn.Module):
    """
    Custom LayerNorm that supports channels_first and channels_last data formats.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        Initialises the LayerNorm module.

        Args:
            normalized_shape (int or tuple): Input shape from an expected set of dimensions.
            eps (float): Epsilon for numerical stability.
            data_format (str): "channels_last" (default) or "channels_first".
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        """Forward pass for LayerNorm."""
        if self.data_format == "channels_last":
            # Use PyTorch's native F.layer_norm for channels_last
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Manually implement LayerNorm for channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# --- ConvNeXt Block ---
class Block(nn.Module):
    """
    The core building block of the ConvNeXt architecture.
    Implements a depthwise conv, layer norm, and a 2-layer MLP (bottleneck).
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        """
        Initialises the ConvNeXt Block.

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate.
            layer_scale_init_value (float): Initial value for LayerScale (gamma).
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # LayerScale (gamma)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        
        # Stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass for the Block, including residual connection."""
        input_tensor = x
        
        # Depthwise Conv
        x = self.dwconv(x)
        
        # Permute for LayerNorm (channels_last)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        
        # 2-layer MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # LayerScale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back to channels_first
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # Add DropPath and residual connection
        x = input_tensor + self.drop_path(x)
        return x

# --- Main ConvNeXt Model --- 
class ConvNeXt(nn.Module):
    """
    ConvNeXt model built from scratch.
    Paper: `A ConvNet for the 2020s` (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, in_chans=1, num_classes=2, 
                depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                drop_path_rate=0., layer_scale_init_value=1e-6, 
                head_init_scale=1.):
        """
        Initialises the ConvNeXt model.

        Args:
            in_chans (int): Number of input image channels.
            num_classes (int): Number of output classes for the classifier.
            depths (list[int]): Number of blocks at each stage.
            dims (list[int]): Feature dimension at each stage.
            drop_path_rate (float): Stochastic depth rate.
            layer_scale_init_value (float): Initial value for LayerScale (gamma).
            head_init_scale (float): Scaling factor for classifier head weights.
        """
        super().__init__()

        # --- Stem and Downsampling Layers ---
        self.downsample_layers = nn.ModuleList()
        
        # 1. Stem: 4x4 convolution with stride 4
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # 2. Downsampling blocks (3 of them)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # --- Main Stages ---
        self.stages = nn.ModuleList()
        
        # Create a list of drop path rates for stochastic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        # Build the 4 stages
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # --- Classifier Head ---
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # Final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        # --- Weight Initialisation ---
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        """Initialises weights for Conv2d and Linear layers."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Passes input through the stem and 4 stages to extract features."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # Global average pooling (mean over H and W)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        """Full forward pass: features -> classifier head."""
        x = self.forward_features(x)
        x = self.head(x)
        return x