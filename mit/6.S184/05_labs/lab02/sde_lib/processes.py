"""
Concrete Implementations of Stochastic Processes

This module provides implementations of fundamental stochastic processes:
- BrownianMotion: Pure diffusion process
- OUProcess: Mean-reverting Ornstein-Uhlenbeck process
- LangevinSDE: Score-based dynamics for sampling from arbitrary distributions

These processes are building blocks for understanding and implementing
more complex generative models and sampling algorithms.

"""

import torch
from einops import repeat
from .base import ODE, SDE, Density


class BrownianMotion(SDE):
    """
    Scaled Brownian Motion process.
    
    Mathematical Definition:
        dX_t = σ dW_t,  X_0 = x_0
    
    where:
        - σ > 0 is the diffusion coefficient (scale parameter)
        - W_t is standard Brownian motion
        - No drift term (pure random walk)
    
    Properties:
        - Markov property: Future depends only on present, not past
        - Gaussian increments: X_t - X_s ~ N(0, σ²(t-s)I)
        - Continuous paths but nowhere differentiable
        - Self-similar: X_t has same distribution as √t · X_1
        - Variance grows linearly: Var(X_t) = σ²t (NOT stationary)
    
    Statistical Properties:
        - Mean: E[X_t | X_0 = x_0] = x_0
        - Variance: Var(X_t | X_0 = 0) = σ²t · I
        - Distribution: X_t | X_0=0 ~ N(0, σ²t · I)
    
    Physical Interpretation:
        Models pure diffusion without any restoring force or drift:
        - Particles in a fluid (Einstein's Brownian motion)
        - Stock prices (random walk hypothesis)
        - Heat diffusion in homogeneous media
    
    Simulation Notes:
        - Does NOT converge to stationary distribution (variance → ∞)
        - Used as noise source for other SDEs
        - Foundation for Wiener process theory
    
    Parameters:
        sigma (float): Diffusion coefficient, controls spread rate.
            - Larger σ → faster spreading
            - σ = 0 → deterministic (no motion)
            - Typical values: 0.1 - 10.0
    
    Example:
        >>> # Simulate 2D Brownian motion
        >>> bm = BrownianMotion(sigma=1.0)
        >>> x0 = torch.zeros(1000, 2)  # Start at origin
        >>> ts = torch.linspace(0, 5, 500)
        >>> sim = EulerMaruyamaSimulator(bm)
        >>> xT = sim.simulate(x0, ts)
        >>> # At t=5, std should be ≈ σ√5 ≈ 2.24
        >>> xT.std(dim=0)  # Each dimension: ≈ 2.24
    
    See Also:
        - OUProcess: Brownian motion with mean reversion
        - GeometricBrownianMotion: Multiplicative version (finance)
    """
    
    def __init__(self, sigma: float):
        """
        Initialize Brownian motion with given diffusion coefficient.
        
        Args:
            sigma (float): Diffusion coefficient σ > 0.
                Controls the rate of spreading.
        
        Raises:
            ValueError: If sigma <= 0
        
        Example:
            >>> bm_slow = BrownianMotion(sigma=0.5)
            >>> bm_fast = BrownianMotion(sigma=2.0)
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift coefficient: u_t(x) = 0.
        
        Brownian motion has zero drift (no deterministic component).
        All motion comes from the random diffusion term.
        
        Args:
            xt (torch.Tensor): Current state.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time (unused).
                Shape: ()
        
        Returns:
            torch.Tensor: Zero drift.
                Shape: (batch_size, dim)
        
        Implementation:
            Returns zeros with same shape and device as xt.
        """
        return torch.zeros_like(xt)
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient: σ_t(x) = σ.
        
        The diffusion is constant (additive noise):
        - Independent of state x
        - Independent of time t
        - Same in all dimensions (isotropic)
        
        Args:
            xt (torch.Tensor): Current state (shape used for output).
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time (unused).
                Shape: ()
        
        Returns:
            torch.Tensor: Constant diffusion σ.
                Shape: (batch_size, dim)
                All entries equal to self.sigma
        
        Implementation:
            Uses torch.full_like for efficient constant tensor creation.
        """
        return torch.full_like(xt, self.sigma)


class OUProcess(SDE):
    """
    Ornstein-Uhlenbeck Process (mean-reverting diffusion).
    
    Mathematical Definition:
        dX_t = -θ X_t dt + σ dW_t,  X_0 = x_0
    
    where:
        - θ > 0 is the mean reversion rate (restoring force strength)
        - σ > 0 is the diffusion coefficient (noise strength)
        - Drift -θX_t pulls toward origin with force proportional to distance
    
    Properties:
        - Mean reversion: Trajectories pulled back toward zero
        - Stationary distribution: N(0, σ²/(2θ) · I)
        - Exponential decay: E[X_t | X_0=x_0] = x_0 · exp(-θt)
        - Relaxation time: τ = 1/θ (time to reach equilibrium)
        - Gaussian process: All finite-dimensional distributions are Gaussian
    
    Physical Interpretation:
        Models systems with restoring force:
        - Particle in harmonic potential with friction
        - Interest rates (Vasicek model)
        - Velocity of Brownian particle (Langevin equation)
        - Neuron membrane potential
    
    Equilibrium Analysis:
        At equilibrium (t → ∞), variance satisfies:
            d Var/dt = σ² - 2θ Var = 0
            ⟹  Var_∞ = σ²/(2θ)
        
        Balance interpretation:
            - Diffusion injects variance at rate σ²
            - Drift removes variance at rate 2θ·Var
            - Equilibrium when injection = removal
    
    Parameter Relationships:
        Define D ≡ σ²/(2θ) (diffusivity):
        - Small θ: Weak restoring force → large variance
        - Large θ: Strong restoring force → small variance
        - Large σ: Strong noise → large variance
        - OU processes with same D have same stationary distribution
    
    Connection to Langevin Dynamics:
        When target distribution is p(x) = N(0, σ²/(2θ)), the OU process
        is equivalent to Langevin dynamics for that target:
            ∇log p(x) = -2θx/σ²
            ⟹  (σ²/2)∇log p(x) = -θx  (matches OU drift!)
    
    Parameters:
        theta (float): Mean reversion rate θ > 0.
            - Larger θ → stronger pull to zero
            - θ → ∞: Overdamped limit
            - θ → 0: Approaches Brownian motion
            Typical values: 0.1 - 2.0
        
        sigma (float): Diffusion coefficient σ > 0.
            - Controls equilibrium variance
            - Balances with θ to determine spread
            Typical values: 0.5 - 5.0
    
    Example:
        >>> # Create OU process with σ²/(2θ) = 1.0
        >>> ou = OUProcess(theta=0.5, sigma=1.0)
        >>> x0 = torch.linspace(-10, 10, 100).view(-1, 1)
        >>> ts = torch.linspace(0, 20, 1000)
        >>> sim = EulerMaruyamaSimulator(ou)
        >>> xT = sim.simulate(x0, ts)
        >>> # Should converge to N(0, 1)
        >>> xT.mean()  # ≈ 0
        >>> xT.std()   # ≈ 1
    
    See Also:
        - BrownianMotion: Special case with θ=0
        - LangevinSDE: Generalization to arbitrary target distributions
    """
    
    def __init__(self, theta: float, sigma: float):
        """
        Initialize Ornstein-Uhlenbeck process.
        
        Args:
            theta (float): Mean reversion rate θ > 0.
                Controls strength of restoring force.
            sigma (float): Diffusion coefficient σ >= 0.
                Controls noise intensity.
        
        Raises:
            ValueError: If theta <= 0 or sigma < 0
        
        Notes:
            The ratio D = σ²/(2θ) determines the stationary variance.
            Two OU processes with same D converge to same distribution.
        
        Example:
            >>> # Weak reversion, moderate noise
            >>> ou_weak = OUProcess(theta=0.25, sigma=1.0)  # D = 2.0
            >>> 
            >>> # Strong reversion, strong noise (same D)
            >>> ou_strong = OUProcess(theta=1.0, sigma=2.0)  # D = 2.0
        """
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if sigma < 0:
            raise ValueError(f"sigma must be positive or 0, got {sigma}")
        
        self.theta = theta
        self.sigma = sigma
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift coefficient: u_t(x) = -θx.
        
        The drift is a linear restoring force pointing toward the origin:
        - Proportional to negative position: -θx
        - Stronger pull when farther from origin
        - Independent across dimensions
        
        This creates mean reversion: trajectories are pulled back toward zero.
        
        Args:
            xt (torch.Tensor): Current state.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time (unused for OU).
                Shape: ()
        
        Returns:
            torch.Tensor: Linear drift -θx.
                Shape: (batch_size, dim)
                Points toward origin from any x
        
        Properties:
            - If x > 0: drift < 0 (pulls left)
            - If x < 0: drift > 0 (pulls right)
            - If x = 0: drift = 0 (equilibrium, but unstable)
        
        Example:
            >>> ou = OUProcess(theta=0.5, sigma=1.0)
            >>> xt = torch.tensor([[2.0, -3.0]])
            >>> drift = ou.drift_coefficient(xt, t=None)
            >>> drift  # = -0.5 * [[2.0, -3.0]] = [[-1.0, 1.5]]
        """
        return -self.theta * xt
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient: σ_t(x) = σ.
        
        Like Brownian motion, OU has constant (additive) noise:
        - Independent of state
        - Independent of time
        - Isotropic (same in all directions)
        
        Args:
            xt (torch.Tensor): Current state (shape used for output).
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time (unused).
                Shape: ()
        
        Returns:
            torch.Tensor: Constant diffusion σ.
                Shape: (batch_size, dim)
        """
        return torch.full_like(xt, self.sigma)


class LangevinSDE(SDE):
    """
    (Overdamped) Langevin Dynamics for sampling from arbitrary distributions.
    
    Mathematical Definition:
        dX_t = (σ²/2) ∇log p(X_t) dt + σ dW_t
    
    where:
        - p(x) is the target probability density
        - ∇log p(x) is the score function (gradient of log density)
        - σ > 0 is the diffusion coefficient
        - The drift term (σ²/2)∇log p drives toward high-probability regions
    
    Theoretical Foundation:
        Langevin dynamics is the continuous-time limit of Langevin MCMC:
        1. The process has p(x) as its stationary distribution
        2. Starting from any initial distribution, X_t → p as t → ∞
        3. This is guaranteed by the Fokker-Planck equation:
            ∂ρ/∂t = -∇·[(σ²/2)∇log p · ρ] + (σ²/2)∇²ρ
           At equilibrium (∂ρ/∂t = 0), ρ = p is the unique solution
    
    Intuition - Drift Term:
        The score ∇log p(x) = ∇p(x)/p(x) points in direction of increasing
        probability. The drift (σ²/2)∇log p thus:
        - Pushes particles uphill on the probability landscape
        - Stronger push in regions with steep gradients
        - Zero push at local maxima (modes)
        - Balances with diffusion to maintain p(x)
    
    Intuition - Balance:
        Think of p(x) as a "height map":
        - Diffusion: Particles randomly spread out (entropy increase)
        - Drift: Score-based force concentrates particles in high-p regions
        - At equilibrium: These forces balance → particles distributed as p(x)
    
    Connection to Physics:
        This is the overdamped Langevin equation from statistical mechanics:
        - Models particle in potential V(x) with friction
        - Potential related to density: V(x) = -log p(x)
        - At equilibrium: Boltzmann distribution p(x) ∝ exp(-V(x))
    
    Applications:
        1. Sampling: Generate samples from p(x) without MCMC
        2. Score-based generative models: Learn ∇log p from data
        3. Diffusion models: Reverse-time Langevin for generation
        4. Bayesian inference: Sample from posterior p(x|data)
        5. Optimization: Find modes of p(x) (annealed Langevin)
    
    Convergence Properties:
        - Mixing time: O(exp(dim)) in worst case (curse of dimensionality)
        - For log-concave p: Fast mixing, polynomial time
        - For multimodal p: May get stuck in local modes
        - Proper σ choice is crucial: Too large → slow convergence
                                       Too small → long mixing time
    
    Parameters:
        sigma (float): Diffusion coefficient σ > 0.
            - Controls balance between drift and diffusion
            - Smaller σ: More deterministic, slower mixing
            - Larger σ: More stochastic, faster exploration but noisier
            - Optimal σ depends on target p and dimension
            Typical values: 0.3 - 1.5
        
        density (Density): Target distribution p(x).
            Must implement:
                - log_density(x): Compute log p(x)
                - score(x): Compute ∇log p(x) (provided by Density base class)
    
    Example - Gaussian Target:
        >>> # Sample from N(0, Σ) using Langevin dynamics
        >>> target = Gaussian(mean=torch.zeros(2), cov=2*torch.eye(2))
        >>> langevin = LangevinSDE(sigma=0.8, density=target)
        >>> 
        >>> # Start from uniform circle
        >>> angles = torch.linspace(0, 2*torch.pi, 1000)
        >>> x0 = 5 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        >>> 
        >>> # Simulate
        >>> sim = EulerMaruyamaSimulator(langevin)
        >>> ts = torch.linspace(0, 10, 2000)
        >>> xT = sim.simulate(x0, ts)
        >>> 
        >>> # Should match target
        >>> xT.mean(dim=0)  # ≈ [0, 0]
        >>> xT.cov()        # ≈ 2·I
    
    Example - Gaussian Mixture:
        >>> # Sample from multimodal distribution
        >>> target = GaussianMixture.symmetric_2D(nmodes=5, std=0.5, scale=8)
        >>> langevin = LangevinSDE(sigma=0.6, density=target)
        >>> 
        >>> # Start from broad Gaussian
        >>> x0 = torch.randn(10000, 2) * 15
        >>> 
        >>> # Long simulation to explore all modes
        >>> ts = torch.linspace(0, 20, 5000)
        >>> xT = sim.simulate(x0, ts)
        >>> # Particles should cluster in 5 modes
    
    Practical Tips:
        1. Choose σ carefully: Too large can cause instability
        2. Use enough simulation time: t ≥ 5-10 for most problems
        3. Small step sizes h: Typically h = 0.001 - 0.01
        4. For multimodal p: Start from multiple initializations
        5. Check convergence: Compare empirical distribution to target
    
    See Also:
        - OUProcess: Special case when p is Gaussian
        - Score-based generative models (Song et al., 2021)
        - MCMC: Discrete-time analog (Metropolis-adjusted Langevin)
    """
    
    def __init__(self, sigma: float, density: Density):
        """
        Initialize Langevin dynamics for given target distribution.
        
        Args:
            sigma (float): Diffusion coefficient σ > 0.
                Controls exploration vs exploitation tradeoff.
            density (Density): Target distribution p(x).
                Must provide log_density and score methods.
        
        Raises:
            ValueError: If sigma <= 0
            TypeError: If density does not implement Density interface
        
        Example:
            >>> # Target: Mixture of Gaussians
            >>> target = GaussianMixture.random_2D(nmodes=3, std=0.8)
            >>> langevin = LangevinSDE(sigma=0.7, density=target)
            >>> 
            >>> # Can now use with any simulator
            >>> sim = EulerMaruyamaSimulator(langevin)
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not isinstance(density, Density):
            raise TypeError(f"density must implement Density interface")
        
        self.sigma = sigma
        self.density = density
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift coefficient: u_t(x) = (σ²/2) ∇log p(x).
        
        The drift is the score function scaled by σ²/2:
        - Score ∇log p(x) points toward high-probability regions
        - Scaling by σ²/2 ensures correct equilibrium
        - Non-linear: Depends on local geometry of p
        
        This is the key component that drives convergence to p(x).
        
        Args:
            xt (torch.Tensor): Current state.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time (unused for Langevin).
                Shape: ()
        
        Returns:
            torch.Tensor: Score-based drift (σ²/2)∇log p(x).
                Shape: (batch_size, dim)
                Points toward high-probability regions
        
        Implementation:
            1. Compute score: self.density.score(xt)
               (Uses automatic differentiation on log_density)
            2. Scale by σ²/2
        
        Computational Cost:
            - Requires gradient computation: O(dim)
            - Batched: Efficient parallelization
            - Can be expensive for complex densities
        
        Example:
            >>> # For Gaussian N(0, I), score(x) = -x
            >>> target = Gaussian(mean=torch.zeros(2), cov=torch.eye(2))
            >>> langevin = LangevinSDE(sigma=1.0, density=target)
            >>> xt = torch.tensor([[1.0, -2.0]])
            >>> drift = langevin.drift_coefficient(xt, t=None)
            >>> # drift = 0.5 * (-xt) = [[-0.5, 1.0]]
            >>> # Points toward origin (center of Gaussian)
        """
        score = self.density.score(xt)
        return 0.5 * (self.sigma ** 2) * score
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient: σ_t(x) = σ.
        
        Like Brownian motion and OU, Langevin has constant diffusion:
        - Independent of state and time
        - Provides exploration (prevents deterministic collapse)
        - Must be balanced with drift for correct equilibrium
        
        Args:
            xt (torch.Tensor): Current state (shape used for output).
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time (unused).
                Shape: ()
        
        Returns:
            torch.Tensor: Constant diffusion σ.
                Shape: (batch_size, dim)
        
        Note:
            The constant diffusion is crucial: It ensures that the
            Fokker-Planck equation has p(x) as stationary solution.
            State-dependent diffusion would require different drift.
        """
        return torch.full_like(xt, self.sigma)


class ConditionalVectorFieldODE(ODE):
    """
    ODE defined by a conditional probability path.
    
    The drift is given by the conditional vector field u_t(x|z):
        dX_t = u_t(X_t | z) dt
    
    This is used to simulate trajectories conditioned on a fixed z.
    """
    
    def __init__(self, path, z: torch.Tensor):
        """
        Initialize conditional ODE.
        
        Args:
            path: ConditionalProbabilityPath instance
            z (torch.Tensor): Conditioning variable.
                Shape: (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift u_t(x|z) with z broadcast to batch size.
        """
        bs = xt.shape[0]
        z = repeat(self.z, '1 d -> b d', b=bs)
        return self.path.conditional_vector_field(xt, z, t)


class ConditionalVectorFieldSDE(SDE):
    """
    SDE defined by conditional vector field and conditional score.
    
    Dynamics:
        dX_t = [u_t(X_t|z) + (σ²/2) ∇_x log p_t(X_t|z)] dt + σ dW_t
    """
    
    def __init__(self, path, z: torch.Tensor, sigma: float):
        """
        Initialize conditional SDE.
        
        Args:
            path: ConditionalProbabilityPath instance
            z (torch.Tensor): Conditioning variable. Shape: (1, dim)
            sigma (float): Diffusion coefficient
        """
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bs = xt.shape[0]
        z = repeat(self.z, '1 d -> b d', b=bs)
        return self.path.conditional_vector_field(xt, z, t) + 0.5 * self.sigma ** 2 * self.path.conditional_score(xt, z, t)
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(xt, self.sigma)


class LearnedVectorFieldODE(ODE):
    """
    ODE driven by a learned vector field model.
    
    Dynamics:
        dX_t = u_t^θ(X_t) dt
    """
    
    def __init__(self, net: torch.nn.Module):
        self.net = net
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(xt, t)


class LangevinFlowSDE(SDE):
    """
    Learned Langevin SDE using both flow and score models.
    
    Dynamics:
        dX_t = [u_t^θ(x) + (σ²/2) s_t^θ(x)] dt + σ dW_t
    """
    
    def __init__(self, flow_model: torch.nn.Module, score_model: torch.nn.Module, sigma: float):
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.flow_model(xt, t) + 0.5 * self.sigma ** 2 * self.score_model(xt, t)
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.randn_like(xt)