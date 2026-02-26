"""
Neural Network Models for Flow and Score Matching

This module provides neural network components for generative modeling:

Basic Models (Lab01-02):
- build_mlp: Simple MLP builder
- MLPVectorField: Vector field network u_t^θ(x) for 2D data
- MLPScore: Score network s_t^θ(x) for 2D data
- ScoreFromVectorField: Compute score from vector field for Gaussian paths

Conditional Models (Lab03):
- ConditionalVectorField: Abstract interface for conditional vector fields
- FourierEncoder: Time embedding using Fourier features
- ResidualLayer: Residual block with FiLM conditioning
- Encoder/Decoder/Midcoder: U-Net components
- MNISTUNet: U-Net architecture for conditional image generation

All models expect time t as an additional input dimension.
"""

from typing import List, Type
from abc import ABC, abstractmethod
import math
import torch
from einops import repeat


def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU) -> torch.nn.Sequential:
    """
    Build a simple MLP with given layer sizes.
    
    Args:
        dims (List[int]): Layer dimensions, e.g., [3, 64, 64, 2]
        activation (torch.nn.Module): Activation class for hidden layers
    
    Returns:
        torch.nn.Sequential: MLP model
    """
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    return torch.nn.Sequential(*mlp)


def _format_time(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Format time tensor to shape (batch_size, 1).
    
    Args:
        t (torch.Tensor): Time tensor (scalar or vector).
        batch_size (int): Batch size to match.
    
    Returns:
        torch.Tensor: Formatted time tensor.
            Shape: (batch_size, 1)
    """
    if t.ndim == 0:
        t = repeat(t, ' -> b 1', b=batch_size)
    elif t.ndim == 1:
        t = t.view(-1, 1)
        if t.shape[0] == 1:
            t = t.expand(batch_size, 1)
    return t


class MLPVectorField(torch.nn.Module):
    """
    MLP-based vector field model u_t^θ(x).
    
    The model takes (x, t) as input and outputs a vector field of same dimension as x:
        u_t^θ(x) = MLP([x, t])
    
    Args:
        dim (int): Data dimension
        hiddens (List[int]): Hidden layer sizes
    """
    
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input positions.
                Shape: (batch_size, dim)
            t (torch.Tensor): Time values.
                Shape: () or (batch_size,) or (batch_size, 1)
        
        Returns:
            torch.Tensor: Vector field output.
                Shape: (batch_size, dim)
        """
        t = _format_time(t, x.shape[0])
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class MLPScore(torch.nn.Module):
    """
    MLP-based score model s_t^θ(x).
    
    The model takes (x, t) as input and outputs the score:
        s_t^θ(x) ≈ ∇_x log p_t(x)
    
    Args:
        dim (int): Data dimension
        hiddens (List[int]): Hidden layer sizes
    """
    
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input positions.
                Shape: (batch_size, dim)
            t (torch.Tensor): Time values.
                Shape: () or (batch_size,) or (batch_size, 1)
        
        Returns:
            torch.Tensor: Score output.
                Shape: (batch_size, dim)
        """
        t = _format_time(t, x.shape[0])
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class ConditionalVectorField(torch.nn.Module, ABC):
    """
    Abstract conditional vector field u_t^theta(x|y).
    Used by classifier-free guidance models.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: shape (bs, c, h, w) or (bs, dim)
            t: shape (bs, 1, 1, 1) or (bs, 1)
            y: shape (bs,)
        Returns:
            torch.Tensor: vector field with same shape as x
        """
        pass


class ScoreFromVectorField(torch.nn.Module):
    r"""
    Compute score function from a learned vector field for Gaussian paths.
    
    Given a Gaussian conditional path with schedules α_t and β_t, the score can be
    derived from the vector field u_t(x):
        s_t(x) = (α_t u_t(x) - \dot{α}_t x) / (β_t^2 \dot{α}_t - α_t \dot{β}_t β_t)
    
    Args:
        vector_field (torch.nn.Module): Learned vector field u_t^θ
        alpha: Alpha schedule
        beta: Beta schedule
    """
    
    def __init__(self, vector_field: torch.nn.Module, alpha, beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute score from vector field.
        
        Args:
            x (torch.Tensor): Input positions.
                Shape: (batch_size, dim)
            t (torch.Tensor): Time values.
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Score estimate.
                Shape: (batch_size, dim)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)
        u_t = self.vector_field(x, t)
        num = alpha_t * u_t - dt_alpha_t * x
        denom = beta_t ** 2 * dt_alpha_t - alpha_t * dt_beta_t * beta_t
        return num / denom


# ============================================
# U-Net Architecture for Conditional Image Generation
# ============================================


class FourierEncoder(torch.nn.Module):
    """
    Fourier feature time embedding.
    
    Encodes scalar time t ∈ [0,1] into high-dimensional Fourier features
    using learnable frequency weights:
        embedding = [sin(2π·t·w), cos(2π·t·w)] · √2
    
    This provides multi-scale temporal information to the network.
    
    Based on: https://github.com/lucidrains/denoising-diffusion-pytorch
    
    Args:
        dim (int): Embedding dimension (must be even).
    """
    
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.half_dim = dim // 2
        # Use register_buffer for fixed Fourier features (not trainable)
        # This ensures weights automatically move with the model to the correct device
        self.register_buffer('weights', torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time into Fourier features.
        
        Args:
            t (torch.Tensor): Time values.
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Fourier embeddings.
                Shape: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class ResidualLayer(torch.nn.Module):
    """
    Residual block with Feature-wise Linear Modulation (FiLM).
    
    Architecture:
        x → Conv → +time_embed → +label_embed → Conv → +residual → out
    
    The time and label embeddings are injected via addition after being
    adapted to match the channel dimension, implementing FiLM conditioning.
    
    Args:
        channels (int): Number of feature channels.
        time_embed_dim (int): Time embedding dimension.
        y_embed_dim (int): Label embedding dimension.
    """
    
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(channels),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(channels),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Time adapter: (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = torch.nn.Sequential(
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, channels)
        )
        # Label adapter: (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = torch.nn.Sequential(
            torch.nn.Linear(y_embed_dim, y_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time and label conditioning.
        
        Args:
            x (torch.Tensor): Feature maps.
                Shape: (bs, channels, h, w)
            t_embed (torch.Tensor): Time embedding.
                Shape: (bs, time_embed_dim)
            y_embed (torch.Tensor): Label embedding.
                Shape: (bs, y_embed_dim)
        
        Returns:
            torch.Tensor: Output features.
                Shape: (bs, channels, h, w)
        """
        res = x.clone()  # Save for residual connection
        
        # First conv block
        x = self.block1(x)
        
        # Add time conditioning (FiLM)
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + t_embed
        
        # Add label conditioning (FiLM)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + y_embed
        
        # Second conv block
        x = self.block2(x)
        
        # Residual connection
        x = x + res
        
        return x


class Encoder(torch.nn.Module):
    """
    U-Net encoder module.
    
    Applies residual blocks followed by downsampling to extract
    hierarchical features at reduced spatial resolution.
    
    Args:
        channels_in (int): Input channels.
        channels_out (int): Output channels.
        num_residual_layers (int): Number of residual blocks.
        t_embed_dim (int): Time embedding dimension.
        y_embed_dim (int): Label embedding dimension.
    """
    
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, 
                 t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = torch.nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) 
            for _ in range(num_residual_layers)
        ])
        self.downsample = torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features.
                Shape: (bs, channels_in, h, w)
            t_embed (torch.Tensor): Time embedding.
                Shape: (bs, t_embed_dim)
            y_embed (torch.Tensor): Label embedding.
                Shape: (bs, y_embed_dim)
        
        Returns:
            torch.Tensor: Downsampled features.
                Shape: (bs, channels_out, h//2, w//2)
        """
        # Apply residual blocks (preserve resolution)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        
        # Downsample (reduce resolution, increase channels)
        x = self.downsample(x)
        
        return x


class Midcoder(torch.nn.Module):
    """
    U-Net bottleneck module.
    
    Processes features at the lowest spatial resolution without
    changing the resolution. This layer has the largest receptive
    field and captures global context.
    
    Args:
        channels (int): Feature channels.
        num_residual_layers (int): Number of residual blocks.
        t_embed_dim (int): Time embedding dimension.
        y_embed_dim (int): Label embedding dimension.
    """
    
    def __init__(self, channels: int, num_residual_layers: int, 
                 t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = torch.nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) 
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features.
                Shape: (bs, channels, h, w)
            t_embed (torch.Tensor): Time embedding.
                Shape: (bs, t_embed_dim)
            y_embed (torch.Tensor): Label embedding.
                Shape: (bs, y_embed_dim)
        
        Returns:
            torch.Tensor: Output features (same shape).
                Shape: (bs, channels, h, w)
        """
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        return x


class Decoder(torch.nn.Module):
    """
    U-Net decoder module.
    
    Applies upsampling followed by residual blocks to reconstruct
    spatial resolution while decreasing feature channels.
    
    Args:
        channels_in (int): Input channels.
        channels_out (int): Output channels.
        num_residual_layers (int): Number of residual blocks.
        t_embed_dim (int): Time embedding dimension.
        y_embed_dim (int): Label embedding dimension.
    """
    
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, 
                 t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        )
        self.res_blocks = torch.nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) 
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features.
                Shape: (bs, channels_in, h, w)
            t_embed (torch.Tensor): Time embedding.
                Shape: (bs, t_embed_dim)
            y_embed (torch.Tensor): Label embedding.
                Shape: (bs, y_embed_dim)
        
        Returns:
            torch.Tensor: Upsampled features.
                Shape: (bs, channels_out, 2*h, 2*w)
        """
        # Upsample (increase resolution, decrease channels)
        x = self.upsample(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
        
        return x


class MNISTUNet(ConditionalVectorField):
    """
    U-Net for conditional MNIST generation.
    
    Implements the conditional vector field u_t^θ(x|y) using a U-Net
    architecture with skip connections. The network takes noisy images,
    time, and class labels as input and predicts the denoising direction.
    
    Architecture:
        x → init_conv → [Encoders + skip connections] → Midcoder → 
        → [Decoders + skip connections] → final_conv → output
    
    Time and label information are injected into every residual block
    via Feature-wise Linear Modulation (FiLM).
    
    Args:
        channels (List[int]): Channel progression, e.g., [32, 64, 128].
            Defines the depth and capacity of the U-Net.
        num_residual_layers (int): Residual blocks per encoder/decoder.
        t_embed_dim (int): Time embedding dimension.
        y_embed_dim (int): Label embedding dimension.
    
    Example:
        >>> unet = MNISTUNet(
        ...     channels=[32, 64, 128],
        ...     num_residual_layers=2,
        ...     t_embed_dim=40,
        ...     y_embed_dim=40
        ... )
        >>> x = torch.randn(4, 1, 32, 32)  # Noisy images
        >>> t = torch.rand(4, 1, 1, 1)      # Time
        >>> y = torch.randint(0, 10, (4,))  # Labels
        >>> u = unet(x, t, y)               # Predicted vector field
        >>> u.shape  # (4, 1, 32, 32)
    """
    
    def __init__(self, channels: List[int], num_residual_layers: int, 
                 t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        
        # Initial convolution: (bs, 1, 32, 32) -> (bs, channels[0], 32, 32)
        self.init_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(channels[0]),
            torch.nn.SiLU()
        )
        
        # Time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)
        
        # Label embedder: 11 labels (0-9 + null label ∅=10)
        self.y_embedder = torch.nn.Embedding(num_embeddings=11, embedding_dim=y_embed_dim)
        
        # Build encoder-decoder pairs
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(reversed(decoders))
        
        # Bottleneck
        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
        
        # Final convolution: (bs, channels[0], 32, 32) -> (bs, 1, 32, 32)
        self.final_conv = torch.nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Noisy images.
                Shape: (bs, 1, 32, 32)
            t (torch.Tensor): Time values.
                Shape: (bs, 1, 1, 1)
            y (torch.Tensor): Class labels.
                Shape: (bs,) - integers in {0, 1, ..., 9, 10}
        
        Returns:
            torch.Tensor: Predicted vector field u_t^θ(x|y).
                Shape: (bs, 1, 32, 32)
        """
        # Embed time and labels
        t_embed = self.time_embedder(t)  # (bs, t_embed_dim)
        y_embed = self.y_embedder(y)     # (bs, y_embed_dim)
        
        # Initial convolution
        x = self.init_conv(x)  # (bs, channels[0], 32, 32)
        
        # Encoding path with skip connections
        residuals = []
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            residuals.append(x.clone())
        
        # Bottleneck
        x = self.midcoder(x, t_embed, y_embed)
        
        # Decoding path with skip connections
        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res  # Add skip connection
            x = decoder(x, t_embed, y_embed)
        
        # Final convolution
        x = self.final_conv(x)  # (bs, 1, 32, 32)
        
        return x
