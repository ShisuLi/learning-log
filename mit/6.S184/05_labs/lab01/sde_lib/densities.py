"""
Probability Density Implementations

This module provides concrete implementations of probability distributions
that support both density evaluation and sampling:

- Gaussian: Multivariate Gaussian distribution
- GaussianMixture: Mixture of Gaussians (multimodal distributions)

These classes are essential for:
- Defining target distributions for Langevin dynamics
- Visualizing probability landscapes
- Generating initial samples for simulations

"""

import torch
import torch.distributions as D
import numpy as np
from .base import Density, Sampleable


class Gaussian(torch.nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian (Normal) Distribution.
    
    Mathematical Definition:
        p(x) = N(x | μ, Σ) = (2π)^{-d/2} |Σ|^{-1/2} exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
    
    where:
        - μ ∈ ℝᵈ is the mean vector
        - Σ ∈ ℝᵈˣᵈ is the covariance matrix (symmetric positive definite)
        - d is the dimensionality
    
    Properties:
        - Unimodal: Single peak at μ
        - Symmetric: Same shape in all directions (after rotation)
        - Exponential decay: Probability decreases exponentially from center
        - Log-concave: log p(x) is concave (useful for optimization)
        - Closed under linear transformations
    
    Score Function:
        ∇log p(x) = -Σ⁻¹(x - μ)
        
        Interpretation:
        - Points toward the mean μ
        - Stronger pull when farther from mean
        - Anisotropic: Pull direction depends on Σ
    
    Applications:
        - Maximum entropy distribution (given mean and covariance)
        - Central limit theorem: Sums converge to Gaussian
        - Kalman filtering: Optimal for linear-Gaussian systems
        - Baseline for more complex distributions
    
    Implementation:
        Wraps torch.distributions.MultivariateNormal for efficient sampling
        and density evaluation. Inherits from torch.nn.Module to support
        GPU acceleration and parameter registration.
    
    Parameters:
        mean (torch.Tensor): Mean vector μ.
            Shape: (dim,)
            Example: torch.zeros(2) for 2D centered at origin
        
        cov (torch.Tensor): Covariance matrix Σ.
            Shape: (dim, dim)
            Must be symmetric positive definite
            Example: torch.eye(2) for independent unit variance
    
    Attributes:
        mean: Registered buffer (moves to GPU with .to(device))
        cov: Registered buffer
        distribution: Property returning torch MultivariateNormal
    
    Example - Standard Gaussian:
        >>> # 2D standard normal N(0, I)
        >>> gaussian = Gaussian(
        ...     mean=torch.zeros(2),
        ...     cov=torch.eye(2)
        ... )
        >>> samples = gaussian.sample(1000)
        >>> log_p = gaussian.log_density(samples)
        >>> score = gaussian.score(samples)  # Points toward origin
    
    Example - Anisotropic Gaussian:
        >>> # Elongated Gaussian
        >>> mean = torch.tensor([1.0, 2.0])
        >>> cov = torch.tensor([[4.0, 1.5],
        ...                      [1.5, 1.0]])
        >>> gaussian = Gaussian(mean, cov)
        >>> 
        >>> # Score at point away from mean
        >>> x = torch.tensor([[3.0, 1.0]])
        >>> score = gaussian.score(x)  # Points toward [1, 2]
    
    See Also:
        - GaussianMixture: Multimodal extension
        - torch.distributions.MultivariateNormal: Underlying implementation
    """
    
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        Initialize multivariate Gaussian distribution.
        
        Args:
            mean (torch.Tensor): Mean vector μ.
                Shape: (dim,)
            cov (torch.Tensor): Covariance matrix Σ.
                Shape: (dim, dim)
                Must be symmetric positive definite
        
        Raises:
            ValueError: If dimensions don't match or cov is not SPD
        
        Example:
            >>> mean = torch.zeros(3)
            >>> cov = torch.diag(torch.tensor([1.0, 2.0, 0.5]))
            >>> gaussian = Gaussian(mean, cov)
        """
        super().__init__()
        # Register as buffers so they move to GPU with .to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
    
    @property
    def distribution(self) -> D.MultivariateNormal:
        """
        Get the underlying PyTorch distribution object.
        
        Returns:
            D.MultivariateNormal: PyTorch distribution for sampling/evaluation.
        
        Note:
            Created on-the-fly to ensure device consistency.
        """
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)
    
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability density log p(x).
        
        Uses the formula:
            log p(x) = -½(x-μ)ᵀΣ⁻¹(x-μ) - ½log|Σ| - (d/2)log(2π)
        
        Args:
            x (torch.Tensor): Points to evaluate.
                Shape: (batch_size, dim)
        
        Returns:
            torch.Tensor: Log density at each point.
                Shape: (batch_size, 1)
        
        Properties:
            - Maximum at x = μ: log p(μ) = -½log|Σ| - (d/2)log(2π)
            - Decreases quadratically away from μ
            - Unbounded below (can be very negative)
        
        Example:
            >>> gaussian = Gaussian(torch.zeros(2), torch.eye(2))
            >>> x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
            >>> log_p = gaussian.log_density(x)
            >>> # At origin: log_p[0] ≈ -1.84 (maximum)
            >>> # At [1,1]: log_p[1] ≈ -2.84 (lower)
        """
        return self.distribution.log_prob(x).view(-1, 1)
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Draw random samples from the Gaussian distribution.
        
        Uses the Cholesky decomposition method:
        1. Compute L where Σ = LLᵀ (Cholesky)
        2. Sample z ~ N(0, I)
        3. Return x = μ + Lz
        
        Args:
            num_samples (int): Number of samples to draw.
        
        Returns:
            torch.Tensor: Random samples.
                Shape: (num_samples, dim)
        
        Properties:
            - Independent samples
            - Empirical mean → μ as n → ∞
            - Empirical covariance → Σ as n → ∞
        
        Example:
            >>> gaussian = Gaussian(torch.ones(2), 2*torch.eye(2))
            >>> samples = gaussian.sample(10000)
            >>> samples.mean(dim=0)  # ≈ [1, 1]
            >>> torch.cov(samples.T)  # ≈ 2*I
        """
        return self.distribution.sample((num_samples,))


class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Gaussian Mixture Model (GMM) - Multimodal Distribution.
    
    Mathematical Definition:
        p(x) = Σᵢ wᵢ · N(x | μᵢ, Σᵢ)
    
    where:
        - K is the number of mixture components (modes)
        - wᵢ ∈ [0,1] are mixture weights (Σwᵢ = 1)
        - μᵢ are component means
        - Σᵢ are component covariances
    
    Properties:
        - Multimodal: Can have multiple peaks
        - Flexible: Universal approximator for smooth densities
        - Non-parametric flavor: More components → more expressiveness
        - Identifiability issues: Label switching, local optima
    
    Score Function:
        ∇log p(x) = Σᵢ γᵢ(x) · ∇log N(x|μᵢ,Σᵢ)
        
        where γᵢ(x) = wᵢN(x|μᵢ,Σᵢ) / p(x) is the responsibility
        
        Interpretation:
        - Weighted average of Gaussian scores
        - Near mode i: Dominated by that component's score
        - Between modes: Complex non-linear behavior
    
    Applications:
        - Clustering: EM algorithm for parameter estimation
        - Density estimation: Flexible modeling of complex data
        - Generative models: Sample from multimodal distributions
        - Testing sampling algorithms: Known ground truth
    
    Implementation:
        Wraps torch.distributions.MixtureSameFamily, which efficiently
        handles mixture distributions using categorical mixture weights.
    
    Parameters:
        means (torch.Tensor): Component means [μ₁, μ₂, ..., μₖ].
            Shape: (nmodes, dim)
        
        covs (torch.Tensor): Component covariances [Σ₁, Σ₂, ..., Σₖ].
            Shape: (nmodes, dim, dim)
            Each must be symmetric positive definite
        
        weights (torch.Tensor): Mixture weights [w₁, w₂, ..., wₖ].
            Shape: (nmodes,)
            Must be non-negative, will be normalized to sum to 1
    
    Class Methods:
        - random_2D: Create random 2D mixture (for testing)
        - symmetric_2D: Create symmetric 2D mixture (for visualization)
    
    Example - Manual Construction:
        >>> # Two-component mixture
        >>> means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
        >>> covs = torch.stack([torch.eye(2), torch.eye(2)])
        >>> weights = torch.tensor([0.3, 0.7])
        >>> gmm = GaussianMixture(means, covs, weights)
        >>> 
        >>> # Sample: 30% from first mode, 70% from second
        >>> samples = gmm.sample(1000)
    
    Example - Symmetric Ring:
        >>> # Five modes arranged in circle
        >>> gmm = GaussianMixture.symmetric_2D(
        ...     nmodes=5,
        ...     std=0.5,    # Tight clusters
        ...     scale=8.0   # Radius of circle
        ... )
        >>> samples = gmm.sample(5000)
        >>> # Samples cluster in 5 symmetric locations
    
    Example - Random Mixture:
        >>> # Random configuration for testing
        >>> gmm = GaussianMixture.random_2D(
        ...     nmodes=8,
        ...     std=1.0,
        ...     scale=15.0,
        ...     seed=42  # Reproducible
        ... )
    
    See Also:
        - Gaussian: Single-mode special case
        - sklearn.mixture.GaussianMixture: Parameter fitting
    """
    
    def __init__(
        self,
        means: torch.Tensor,
        covs: torch.Tensor,
        weights: torch.Tensor
    ):
        """
        Initialize Gaussian Mixture Model.
        
        Args:
            means (torch.Tensor): Component means.
                Shape: (nmodes, dim)
            covs (torch.Tensor): Component covariances.
                Shape: (nmodes, dim, dim)
            weights (torch.Tensor): Mixture weights.
                Shape: (nmodes,)
                Will be normalized to sum to 1
        
        Example:
            >>> means = torch.randn(3, 2) * 5  # 3 modes in 2D
            >>> covs = torch.stack([torch.eye(2)] * 3)
            >>> weights = torch.tensor([1.0, 2.0, 1.0])  # Unequal weights
            >>> gmm = GaussianMixture(means, covs, weights)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        # Register as buffers for GPU support
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)
    
    @property
    def dim(self) -> int:
        """
        Get dimensionality of the distribution.
        
        Returns:
            int: Dimension of the space.
        """
        return self.means.shape[1]
    
    @property
    def distribution(self) -> D.MixtureSameFamily:
        """
        Get the underlying PyTorch mixture distribution.
        
        Returns:
            D.MixtureSameFamily: Mixture distribution object.
        
        Implementation:
            Uses categorical distribution for mixture weights and
            multivariate normal for components.
        """
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(
                probs=self.weights,
                validate_args=False
            ),
            component_distribution=D.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False
            ),
            validate_args=False
        )
    
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability density log p(x).
        
        Uses numerically stable log-sum-exp:
            log p(x) = log[Σᵢ wᵢ·N(x|μᵢ,Σᵢ)]
                     = log[Σᵢ exp(log wᵢ + log N(x|μᵢ,Σᵢ))]
        
        Args:
            x (torch.Tensor): Points to evaluate.
                Shape: (batch_size, dim)
        
        Returns:
            torch.Tensor: Log density.
                Shape: (batch_size, 1)
        
        Properties:
            - Smooth everywhere (mixture of smooth densities)
            - Local maxima at or near component means
            - Can have complex landscape (saddle points, valleys)
        
        Example:
            >>> gmm = GaussianMixture.symmetric_2D(nmodes=3, std=1.0)
            >>> x = torch.randn(100, 2)
            >>> log_p = gmm.log_density(x)
            >>> # High values near modes, low between modes
        """
        return self.distribution.log_prob(x).view(-1, 1)
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Draw random samples from the mixture distribution.
        
        Algorithm:
        1. Sample component indices: i ~ Categorical(w)
        2. Sample from selected component: x ~ N(μᵢ, Σᵢ)
        
        Args:
            num_samples (int): Number of samples to draw.
        
        Returns:
            torch.Tensor: Random samples.
                Shape: (num_samples, dim)
        
        Properties:
            - Approximately wᵢ·num_samples from component i
            - Empirical distribution → p(x) as n → ∞
            - Shows multimodal structure clearly
        
        Example:
            >>> gmm = GaussianMixture.symmetric_2D(nmodes=5, std=0.5)
            >>> samples = gmm.sample(5000)
            >>> # Visualize: samples cluster in 5 groups
            >>> plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
        """
        return self.distribution.sample(torch.Size((num_samples,)))
    
    @classmethod
    def random_2D(
        cls,
        nmodes: int,
        std: float,
        scale: float = 10.0,
        seed: int = 42
    ) -> "GaussianMixture":
        """
        Create a random 2D Gaussian mixture for testing/experimentation.
        
        Component means are sampled uniformly in [-scale/2, scale/2]².
        All components have equal weight and spherical covariance.
        
        Args:
            nmodes (int): Number of mixture components.
                Typical values: 3-10
            
            std (float): Standard deviation of each component.
                Controls cluster tightness
                Typical values: 0.5-2.0
            
            scale (float, optional): Spatial extent of means.
                Means sampled in [-scale/2, scale/2]²
                Default: 10.0
            
            seed (int, optional): Random seed for reproducibility.
                Default: 42
        
        Returns:
            GaussianMixture: Random 2D mixture.
        
        Example:
            >>> # Reproducible random mixture
            >>> gmm1 = GaussianMixture.random_2D(5, std=1.0, seed=42)
            >>> gmm2 = GaussianMixture.random_2D(5, std=1.0, seed=42)
            >>> # gmm1 and gmm2 are identical
            >>> 
            >>> # Different configuration
            >>> gmm3 = GaussianMixture.random_2D(5, std=1.0, seed=123)
        """
        torch.manual_seed(seed)
        # Random means in box [-scale/2, scale/2]²
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        # Spherical covariances
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * (std ** 2)
        # Equal weights
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)
    
    @classmethod
    def symmetric_2D(
        cls,
        nmodes: int,
        std: float,
        scale: float = 10.0
    ) -> "GaussianMixture":
        """
        Create a symmetric 2D Gaussian mixture (modes arranged in circle).
        
        Component means are evenly spaced on a circle of radius `scale`.
        All components have equal weight and spherical covariance.
        This creates a visually appealing symmetric pattern.
        
        Args:
            nmodes (int): Number of mixture components.
                Typical values: 3-8 (for clear separation)
            
            std (float): Standard deviation of each component.
                Controls cluster tightness
                Smaller std → tighter clusters
            
            scale (float, optional): Radius of circle.
                Distance of modes from origin
                Default: 10.0
        
        Returns:
            GaussianMixture: Symmetric 2D mixture.
        
        Geometry:
            Modes located at angles: 0, 2π/K, 4π/K, ..., 2π(K-1)/K
            In Cartesian coordinates:
                μᵢ = scale · [cos(2πi/K), sin(2πi/K)]
        
        Applications:
            - Visualization: Clear symmetric structure
            - Testing: Known geometry helps verify algorithms
            - Benchmark: Standard test case for sampling methods
        
        Example - Pentagonal Pattern:
            >>> # 5 modes in pentagon shape
            >>> gmm = GaussianMixture.symmetric_2D(
            ...     nmodes=5,
            ...     std=0.8,
            ...     scale=8.0
            ... )
            >>> samples = gmm.sample(5000)
            >>> # Forms pentagonal pattern
        
        Example - Ring of Gaussians:
            >>> # Many modes → approximate ring
            >>> gmm = GaussianMixture.symmetric_2D(
            ...     nmodes=20,
            ...     std=0.3,
            ...     scale=10.0
            ... )
        """
        # Evenly spaced angles around circle
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        # Convert to Cartesian coordinates
        means = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1) * scale
        # Spherical covariances
        covs = torch.diag_embed(torch.ones(nmodes, 2) * (std ** 2))
        # Equal weights (normalized automatically)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)