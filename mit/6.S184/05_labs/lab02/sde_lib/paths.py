"""
Conditional Probability Paths

This module defines conditional probability paths p_t(x|z) used in conditional flow matching.
A conditional probability path specifies how samples evolve from a simple distribution
p_simple to a target distribution p_data, conditioned on a variable z.

Key interfaces:
- ConditionalProbabilityPath: Abstract base class
- GaussianConditionalProbabilityPath: Gaussian bridge with analytical score
- LinearConditionalProbabilityPath: Deterministic linear interpolation path

"""

from abc import ABC, abstractmethod
import torch
from .densities import Gaussian


class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract base class for conditional probability paths.
    
    A conditional probability path defines a family of distributions p_t(x|z)
    such that:
        - p_0(x|z) = p_simple(x)
        - p_1(x|z) = δ_z(x)
    
    The path is parameterized by time t ∈ [0, 1].
    
    Attributes:
        p_simple (Sampleable): Source distribution p_0(x)
        p_data (Sampleable): Target data distribution p(z)
    
    Subclasses must implement:
        - sample_conditioning_variable
        - sample_conditional_path
        - conditional_vector_field
        - conditional_score
    """
    
    def __init__(self, p_simple, p_data):
        """
        Initialize conditional probability path.
        
        Args:
            p_simple (Sampleable): Source distribution p_0(x)
            p_data (Sampleable): Target distribution p(z)
        """
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
    
    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the marginal distribution p_t(x).
        
        The marginal is obtained by sampling z ~ p(z), then x ~ p_t(x|z).
        
        Args:
            t (torch.Tensor): Time values.
                Shape: (batch_size, 1)
        
        Returns:
            torch.Tensor: Samples from p_t(x).
                Shape: (batch_size, dim)
        """
        num_samples = t.shape[0]
        z = self.sample_conditioning_variable(num_samples)  # z ~ p(z)
        x = self.sample_conditional_path(z, t)  # x ~ p_t(x|z)
        return x
    
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Sample conditioning variables z ~ p(z).
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x ~ p_t(x|z).
        """
        pass
    
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional vector field u_t(x|z).
        """
        pass
    
    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional score ∇_x log p_t(x|z).
        """
        pass


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Gaussian conditional probability path.
    
    Defines a Gaussian bridge:
        p_t(x|z) = N(x; α_t z, β_t^2 I)
    
    where α_t and β_t satisfy:
        α_0 = 0, α_1 = 1
        β_0 = 1, β_1 = 0
    
    This path admits a closed-form conditional score.
    """
    
    def __init__(self, p_data, alpha, beta):
        """
        Initialize Gaussian conditional path.
        
        Args:
            p_data (Sampleable): Target distribution p(z)
            alpha (Alpha): Schedule α_t
            beta (Beta): Schedule β_t
        """
        p_simple = Gaussian.isotropic(p_data.dim, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)
        return dt_alpha_t * z + dt_beta_t / beta_t * (x - alpha_t * z)
    
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / (beta_t ** 2)


class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Linear conditional probability path.
    
    Defines deterministic linear interpolation:
        X_t = (1 - t) X_0 + t z
    
    This path is deterministic given X_0 and z, and does not have
    a closed-form conditional score in general.
    """
    
    def __init__(self, p_simple, p_data):
        """
        Initialize linear conditional path.
        
        Args:
            p_simple (Sampleable): Source distribution p_0(x)
            p_data (Sampleable): Target distribution p(z)
        """
        super().__init__(p_simple, p_data)
    
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = self.p_simple.sample(z.shape[0])
        xt = (1 - t) * x0 + t * z
        return xt
    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (z - x) / (1 - t)
    
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Conditional score is not available in closed form.
        
        For linear paths, p_t(x|z) is the pushforward of p_simple under a
        deterministic map, making the score generally intractable.
        """
        raise NotImplementedError("LinearConditionalProbabilityPath does not provide a closed-form conditional score.")
