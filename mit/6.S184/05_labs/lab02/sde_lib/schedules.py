"""
Alpha and Beta Schedules for Conditional Probability Paths

This module provides schedule functions α_t and β_t that control the interpolation
between source and target distributions in conditional probability paths:

- Alpha: Controls the weight on the target (data) component
- Beta: Controls the weight on the source (noise) component

For Gaussian conditional paths: X_t = α_t·z + β_t·X_0

Boundary conditions:
- α_0 = 0, α_1 = 1 (start from source, end at target)
- β_0 = 1, β_1 = 0 (start with full noise, end with none)

"""

from abc import ABC, abstractmethod
import torch
from einops import rearrange
from torch.func import vmap, jacrev


class Alpha(ABC):
    """
    Abstract base class for alpha schedules in probability paths.
    
    Mathematical Definition:
        α: [0,1] → [0,1], where α_0 = 0 and α_1 = 1
    
    The alpha schedule controls the weight of the target (conditioning variable z)
    in the interpolation:
        X_t = α_t·z + β_t·X_0
    
    Properties:
        - Monotonically increasing from 0 to 1
        - α_0 = 0: Start with no target influence
        - α_1 = 1: End at exactly the target
    
    Subclasses must implement:
        - __call__: Compute α_t
        - dt (optional): Compute dα_t/dt (has default autodiff implementation)
    
    Example:
        >>> alpha = LinearAlpha()
        >>> t = torch.tensor([[0.5]])
        >>> alpha(t)  # Returns 0.5
        >>> alpha.dt(t)  # Returns 1.0 (derivative)
    """
    
    def __init__(self):
        """
        Initialize and verify boundary conditions.
        
        Raises:
            AssertionError: If α_0 ≠ 0 or α_1 ≠ 1
        """
        assert torch.allclose(self(torch.zeros(1, 1)), torch.zeros(1, 1)), \
            "Alpha schedule must satisfy α_0 = 0"
        assert torch.allclose(self(torch.ones(1, 1)), torch.ones(1, 1)), \
            "Alpha schedule must satisfy α_1 = 1"
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute α_t at given time points.
        
        Args:
            t (torch.Tensor): Time values in [0, 1].
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Alpha values at each time.
                Shape: (batch_size, 1)
        """
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dα_t/dt using autodiff.
        
        Args:
            t (torch.Tensor): Time values in [0, 1].
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Derivative values at each time.
                Shape: (batch_size, 1)
        
        Note:
            Uses vmap and jacrev for vectorized autodiff.
            Subclasses can override with analytical derivatives for efficiency.
        """
        t = rearrange(t, 'b d -> b 1 d')  # (batch_size, 1, 1)
        dt = vmap(jacrev(self))(t)  # (batch_size, 1, 1, 1, 1)
        return rearrange(dt, 'b ... d -> b d')


class Beta(ABC):
    """
    Abstract base class for beta schedules in probability paths.
    
    Mathematical Definition:
        β: [0,1] → [0,1], where β_0 = 1 and β_1 = 0
    
    The beta schedule controls the weight of the source (noise) component:
        X_t = α_t·z + β_t·X_0
    
    For Gaussian paths, β_t determines the standard deviation of the conditional:
        p_t(x|z) = N(x; α_t·z, β_t²·I)
    
    Properties:
        - Monotonically decreasing from 1 to 0
        - β_0 = 1: Start with full source distribution
        - β_1 = 0: End with delta at target (no variance)
    
    Subclasses must implement:
        - __call__: Compute β_t
        - dt (optional): Compute dβ_t/dt (has default autodiff implementation)
    
    Example:
        >>> beta = SquareRootBeta()
        >>> t = torch.tensor([[0.5]])
        >>> beta(t)  # Returns sqrt(0.5) ≈ 0.707
        >>> beta.dt(t)  # Returns derivative
    """
    
    def __init__(self):
        """
        Initialize and verify boundary conditions.
        
        Raises:
            AssertionError: If β_0 ≠ 1 or β_1 ≠ 0
        """
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1)), \
            "Beta schedule must satisfy β_0 = 1"
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1)), \
            "Beta schedule must satisfy β_1 = 0"
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute β_t at given time points.
        
        Args:
            t (torch.Tensor): Time values in [0, 1].
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Beta values at each time.
                Shape: (batch_size, 1)
        """
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dβ_t/dt using autodiff.
        
        Args:
            t (torch.Tensor): Time values in [0, 1].
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Derivative values at each time.
                Shape: (batch_size, 1)
        
        Note:
            Uses vmap and jacrev for vectorized autodiff.
            Subclasses can override with analytical derivatives for efficiency.
        """
        t = rearrange(t, 'b d -> b 1 d')
        dt = vmap(jacrev(self))(t)
        return rearrange(dt, 'b ... d -> b d')


class LinearAlpha(Alpha):
    """
    Linear alpha schedule: α_t = t
    
    Mathematical Definition:
        α_t = t
        dα_t/dt = 1
    
    Properties:
        - Simplest schedule with constant derivative
        - Linear interpolation toward target
        - Commonly used with SquareRootBeta for variance-preserving paths
    
    Example:
        >>> alpha = LinearAlpha()
        >>> t = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
        >>> alpha(t)
        tensor([[0.00], [0.25], [0.50], [0.75], [1.00]])
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute α_t = t.
        
        Args:
            t (torch.Tensor): Time values.
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Same as input (identity function).
                Shape: (batch_size, 1)
        """
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute dα_t/dt = 1.
        
        Args:
            t (torch.Tensor): Time values (unused, derivative is constant).
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Ones tensor.
                Shape: (batch_size, 1)
        """
        return torch.ones_like(t)


class SquareRootBeta(Beta):
    """
    Square root beta schedule: β_t = √(1-t)
    
    Mathematical Definition:
        β_t = √(1-t)
        dβ_t/dt = -1/(2√(1-t))
    
    Properties:
        - Variance decays linearly: β_t² = 1-t
        - Combined with LinearAlpha, gives variance-preserving interpolation
        - Derivative diverges as t → 1 (handled numerically)
    
    Note:
        The conditional variance is β_t² = 1-t, which decreases linearly.
        This is the standard choice for diffusion models.
    
    Example:
        >>> beta = SquareRootBeta()
        >>> t = torch.tensor([[0.0], [0.5], [0.99]])
        >>> beta(t)
        tensor([[1.0000], [0.7071], [0.1000]])
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute β_t = √(1-t).
        
        Args:
            t (torch.Tensor): Time values in [0, 1).
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Square root of (1-t).
                Shape: (batch_size, 1)
        
        Warning:
            Returns 0 at t=1, but derivative is undefined there.
        """
        return torch.sqrt(1 - t)
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute dβ_t/dt = -1/(2√(1-t)).
        
        Args:
            t (torch.Tensor): Time values in [0, 1).
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Derivative values (negative).
                Shape: (batch_size, 1)
        
        Warning:
            Diverges to -∞ as t → 1. Avoid using at t=1.
        """
        return -0.5 / torch.sqrt(1 - t)
