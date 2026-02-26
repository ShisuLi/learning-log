"""
Time schedules for conditional probability paths - Production Version

Provides α_t and β_t schedules for interpolating between distributions.
"""

from abc import ABC, abstractmethod
import torch
from torch.func import vmap, jacrev


class Alpha(ABC):
    """
    Alpha schedule: α: [0,1] → [0,1] with α_0 = 0, α_1 = 1
    Controls weight on target in interpolation.
    """
    
    def __init__(self):
        # Verify boundary conditions
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)), \
            "Alpha schedule must satisfy α_0 = 0"
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)), \
            "Alpha schedule must satisfy α_1 = 1"
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute α_t."""
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Compute dα_t/dt using autodiff."""
        t_in = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t_in)
        return dt.view_as(t)


class Beta(ABC):
    """
    Beta schedule: β: [0,1] → [0,1] with β_0 = 1, β_1 = 0
    Controls weight on source (noise) in interpolation.
    """
    
    def __init__(self):
        # Verify boundary conditions
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)), \
            "Beta schedule must satisfy β_0 = 1"
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)), \
            "Beta schedule must satisfy β_1 = 0"
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute β_t."""
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Compute dβ_t/dt using autodiff."""
        t_in = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t_in)
        return dt.view_as(t)


class LinearAlpha(Alpha):
    """Linear schedule: α_t = t"""
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class LinearBeta(Beta):
    """Linear schedule: β_t = 1 - t"""
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


class SquareRootBeta(Beta):
    """Square root schedule: β_t = √(1 - t)"""
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - t)
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 / torch.sqrt(1.0 - t)
