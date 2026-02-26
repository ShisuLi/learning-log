"""
Conditional Probability Paths

This module defines conditional probability paths p_t(x|z) used in conditional flow matching.
A conditional probability path specifies how samples evolve from a simple distribution
p_simple to a target distribution p_data, conditioned on a variable z.

For lab03 (image generation), tensors use the 4D convention (bs, c, h, w) so that
broadcast operations work correctly across spatial dimensions.

Key interfaces:
- ConditionalProbabilityPath: Abstract base class
- GaussianConditionalProbabilityPath: Gaussian bridge with analytical score and vector field
- LinearConditionalProbabilityPath: Deterministic linear interpolation path

Notation used in docstrings:
    bs   – batch size
    c    – number of image channels  (e.g. 1 for grayscale MNIST)
    h, w – spatial height / width    (e.g. 32 × 32 for MNIST after resize)
    dim  – flattened data dimension   (used for non-image data)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from .densities import Gaussian
from .densities import IsotropicGaussian


class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract base class for conditional probability paths.
    
    A conditional probability path defines a family of distributions p_t(x|z)
    parameterised by time t ∈ [0, 1] such that:
        - p_0(x|z) = p_simple(x)   (start: pure noise)
        - p_1(x|z) = δ_z(x)        (end: data sample z)
    
    For image data the spatial tensor convention (bs, c, h, w) is used
    throughout so that time t of shape (bs, 1, 1, 1) broadcasts correctly.
    
    Attributes:
        p_simple (Sampleable): Source distribution p_0(x), e.g. IsotropicGaussian
        p_data   (Sampleable): Target data distribution p(z), e.g. MNISTSampler
    
    Subclasses must implement:
        - sample_conditioning_variable : sample (z, y) from p_data
        - sample_conditional_path      : sample x ~ p_t(x|z)
        - conditional_vector_field     : evaluate u_t(x|z)
        - conditional_score            : evaluate ∇_x log p_t(x|z)
    """
    
    def __init__(self, p_simple, p_data):
        """
        Initialize conditional probability path.
        
        Args:
            p_simple (Sampleable): Source distribution p_0(x)
            p_data   (Sampleable): Target distribution p(z)
        """
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
    
    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the marginal distribution p_t(x) = ∫ p_t(x|z) p(z) dz.
        
        The marginal is obtained by ancestral sampling:
            1. z, _ = sample_conditioning_variable(bs)   →  z ~ p(z)
            2. x    = sample_conditional_path(z, t)      →  x ~ p_t(x|z)
        
        Args:
            t (torch.Tensor): Time values.
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Samples from the marginal p_t(x).
                Shape: (bs, c, h, w)
        """
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples)  # z ~ p(z)
        x = self.sample_conditional_path(z, t)                 # x ~ p_t(x|z)
        return x
    
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample the conditioning variable z and optional class label y from p_data.
        
        For MNIST this draws a random image z together with its digit label y.
        For unlabelled distributions (e.g. Gaussian), y is returned as None.
        
        Args:
            num_samples (int): Number of (z, y) pairs to sample.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - z: conditioning variable, shape (bs, c, h, w) or (bs, dim)
                - y: integer class labels,  shape (bs, label_dim),  or None
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x ~ p_t(x|z) from the conditional path at time t.
        
        Args:
            z (torch.Tensor): Conditioning variable (data sample).
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)  ← 4D so it broadcasts over spatial dims
        
        Returns:
            torch.Tensor: Sample from the conditional distribution p_t(x|z).
                Shape: (bs, c, h, w)
        """
        pass
    
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the conditional vector field u_t(x|z).
        
        The vector field is the drift that transforms p_simple into p_data
        when followed for t ∈ [0, 1].  It is used as the regression target
        in the Conditional Flow Matching (CFM) objective.
        
        Args:
            x (torch.Tensor): Position variable (noisy sample along the path).
                Shape: (bs, c, h, w)
            z (torch.Tensor): Conditioning variable (clean data sample).
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Conditional vector field u_t(x|z).
                Shape: (bs, c, h, w)  — same spatial layout as x
        """
        pass
    
    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the conditional score ∇_x log p_t(x|z).
        
        The score points in the direction of increasing log-probability
        of the conditional p_t(x|z) and is used in SDE samplers.
        
        Args:
            x (torch.Tensor): Position variable.
                Shape: (bs, c, h, w)
            z (torch.Tensor): Conditioning variable.
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Conditional score ∇_x log p_t(x|z).
                Shape: (bs, c, h, w)
        """
        pass


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Gaussian conditional probability path (Gaussian bridge).
    
    Defines the family of Gaussian conditionals:
        p_t(x|z) = N(x ; α_t·z,  β_t²·I)
    
    where the schedules satisfy the boundary conditions
        α_0 = 0,  α_1 = 1   (mean goes from 0 to z)
        β_0 = 1,  β_1 = 0   (std goes from 1 to 0)
    
    This path admits closed-form expressions for both the conditional
    vector field and the conditional score (see methods below).
    
    Closed-form conditional vector field (CFM regression target):
        u_t(x|z) = α̇_t·z + (β̇_t / β_t)·(x − α_t·z)
    
    Closed-form conditional score:
        ∇_x log p_t(x|z) = (α_t·z − x) / β_t²
    
    Args:
        p_data         (Sampleable): Target distribution p(z) (e.g. MNISTSampler).
        alpha          (Alpha):      Schedule α_t.
        beta           (Beta):       Schedule β_t.
        p_simple_shape (List[int]):  Shape of one sample (e.g. [1, 32, 32]) used to
                                     construct an IsotropicGaussian noise source.
                                     Mutually exclusive with p_simple.
        p_simple       (Sampleable): Explicit noise source. Overrides p_simple_shape.
    """
    
    def __init__(self, p_data, alpha, beta, p_simple_shape=None, p_simple=None):
        """
        Initialize Gaussian conditional probability path.
        
        Args:
            p_data         (Sampleable):      Target distribution p(z).
            alpha          (Alpha):           Schedule α_t satisfying α_0=0, α_1=1.
            beta           (Beta):            Schedule β_t satisfying β_0=1, β_1=0.
            p_simple_shape (List[int] | None): Shape for IsotropicGaussian source,
                                               e.g. [1, 32, 32] for MNIST.
            p_simple       (Sampleable | None): Pre-built noise source; takes priority
                                                over p_simple_shape.
        """
        if p_simple is not None:
            base = p_simple
        elif p_simple_shape is not None:
            base = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        else:
            base = Gaussian.isotropic(p_data.dim, std=1.0)
        super().__init__(base, p_data)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample (z, y) ~ p_data.
        
        Args:
            num_samples (int): Number of samples.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - z: data samples, shape (bs, c, h, w) or (bs, dim)
                - y: class labels,  shape (bs,),  or None
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x ~ p_t(x|z) = N(α_t·z, β_t²·I).
        
        Reparameterisation:
            x = α_t · z + β_t · ε,   ε ~ N(0, I)
        
        Args:
            z (torch.Tensor): Conditioning variable (clean data sample).
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)  ← broadcasts over spatial dims
        
        Returns:
            torch.Tensor: Noisy sample along the path.
                Shape: (bs, c, h, w)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the closed-form conditional vector field u_t(x|z).
        
        Formula:
            u_t(x|z) = α̇_t · z + (β̇_t / β_t) · (x − α_t · z)
        
        This is the optimal (variance-reducing) regression target for the
        CFM loss  E[‖u_θ(x,t) − u_t(x|z)‖²].
        
        Args:
            x (torch.Tensor): Noisy sample on the path.
                Shape: (bs, c, h, w)
            z (torch.Tensor): Clean conditioning variable.
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Conditional vector field u_t(x|z).
                Shape: (bs, c, h, w)
        """
        alpha_t    = self.alpha(t)      # (bs, 1, 1, 1)
        beta_t     = self.beta(t)       # (bs, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t)   # (bs, 1, 1, 1)
        dt_beta_t  = self.beta.dt(t)    # (bs, 1, 1, 1)
        return dt_alpha_t * z + dt_beta_t / beta_t * (x - alpha_t * z)
    
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the closed-form conditional score ∇_x log p_t(x|z).
        
        For a Gaussian p_t(x|z) = N(α_t·z, β_t²·I) the score is:
            ∇_x log p_t(x|z) = (α_t·z − x) / β_t²
        
        Args:
            x (torch.Tensor): Noisy sample on the path.
                Shape: (bs, c, h, w)
            z (torch.Tensor): Clean conditioning variable.
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Conditional score.
                Shape: (bs, c, h, w)
        """
        alpha_t = self.alpha(t)   # (bs, 1, 1, 1)
        beta_t  = self.beta(t)    # (bs, 1, 1, 1)
        return (z * alpha_t - x) / (beta_t ** 2)


class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Deterministic linear interpolation path (flow matching straight path).
    
    Defines the conditional:
        X_t = (1 − t)·X_0 + t·z
    
    where X_0 ~ p_simple is independent noise.  The path is deterministic
    given (X_0, z), so it does NOT have a tractable closed-form conditional
    score (see conditional_score below).
    
    Closed-form conditional vector field:
        u_t(x|z) = (z − x) / (1 − t)
    
    Note:
        This diverges as t → 1 and should not be evaluated exactly at t = 1.
    """
    
    def __init__(self, p_data, p_simple=None, p_simple_shape=None):
        """
        Initialize linear conditional path.
        
        Args:
            p_simple (Sampleable): Source (noise) distribution p_0(x).
            p_data   (Sampleable): Target data distribution p(z).
        """
        if p_simple is not None:
            base = p_simple
        elif p_simple_shape is not None:
            base = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        else:
            base = Gaussian.isotropic(p_data.dim, std=1.0)
        super().__init__(base, p_data)
    
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample (z, y) ~ p_data.
        
        Args:
            num_samples (int): Number of samples.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (z, y) where y may be None.
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x ~ p_t(x|z) via linear interpolation.
        
        Reparameterisation:
            x0, _ = p_simple.sample(bs)
            x_t   = (1 − t)·x0 + t·z
        
        Args:
            z (torch.Tensor): Conditioning variable.
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Linearly interpolated sample.
                Shape: (bs, c, h, w)
        """
        x0, _ = self.p_simple.sample(z.shape[0])
        xt = (1 - t) * x0 + t * z
        return xt
    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the conditional vector field u_t(x|z) = (z − x) / (1 − t).
        
        Args:
            x (torch.Tensor): Current position on the path. Shape: (bs, c, h, w)
            z (torch.Tensor): Conditioning variable.       Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1).              Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Velocity pointing from x toward z.
                Shape: (bs, c, h, w)
        
        Warning:
            Undefined at t = 1; avoid evaluating exactly there.
        """
        return (z - x) / (1 - t)
    
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Conditional score — not available in closed form for linear paths.
        
        For linear paths p_t(x|z) is the pushforward of p_simple under a
        deterministic map, making the score generally intractable.
        
        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "LinearConditionalProbabilityPath does not provide a closed-form conditional score."
        )
