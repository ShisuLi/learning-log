"""
Neural Network Models for Flow and Score Matching

This module provides neural network components used in lab02:
- build_mlp: Simple MLP builder
- MLPVectorField: Vector field network u_t^θ(x)
- MLPScore: Score network s_t^θ(x)
- ScoreFromVectorField: Compute score from vector field for Gaussian paths

All models expect time t as an additional input dimension.
"""

from typing import List, Type
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


class ScoreFromVectorField(torch.nn.Module):
    """
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
