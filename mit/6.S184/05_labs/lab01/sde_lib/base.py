"""
Base Classes for ODEs, SDEs, and Simulators

This module provides abstract base classes that define the interface for:
- Ordinary Differential Equations (ODEs)
- Stochastic Differential Equations (SDEs)
- Numerical simulators for both ODEs and SDEs
- Probability densities and sampleable distributions

The modular design allows for easy extension and composition of different
dynamical systems and simulation schemes.

"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from tqdm import tqdm


class ODE(ABC):
    """
    Abstract base class for Ordinary Differential Equations.
    
    An ODE describes deterministic evolution of a state through time according to:
        dX_t = u_t(X_t) dt,  X_0 = x_0
    
    where u_t is the drift coefficient (velocity field).
    
    Subclasses must implement:
        - drift_coefficient: Computes the velocity field u_t(x)
    """
    
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the drift coefficient (velocity field) u_t(x).
        
        The drift coefficient determines the deterministic direction and magnitude
        of movement at each point in space and time.
        
        Args:
            xt (torch.Tensor): Current state at time t.
                Shape: (batch_size, dim)
                - batch_size: Number of trajectories to simulate simultaneously
                - dim: Dimensionality of the state space
            t (torch.Tensor): Current time.
                Shape: () or (1,)
                Scalar value representing the current time
        
        Returns:
            torch.Tensor: Drift coefficient at (xt, t).
                Shape: (batch_size, dim)
                Same shape as xt, representing the velocity at each point
        
        Example:
            >>> ode = MyODE()
            >>> xt = torch.randn(100, 2)  # 100 particles in 2D
            >>> t = torch.tensor(0.5)
            >>> drift = ode.drift_coefficient(xt, t)  # shape: (100, 2)
        """
        pass


class SDE(ABC):
    """
    Abstract base class for Stochastic Differential Equations.
    
    An SDE describes stochastic evolution of a state through time according to:
        dX_t = u_t(X_t) dt + σ_t(X_t) dW_t,  X_0 = x_0
    
    where:
        - u_t is the drift coefficient (deterministic component)
        - σ_t is the diffusion coefficient (stochastic component)
        - W_t is a Brownian motion process
    
    The SDE combines deterministic flow with random perturbations, allowing
    for modeling of uncertain or noisy dynamical systems.
    
    Subclasses must implement:
        - drift_coefficient: Computes u_t(x)
        - diffusion_coefficient: Computes σ_t(x)
    """
    
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the drift coefficient u_t(x).
        
        The drift term represents the deterministic component of the dynamics,
        similar to the velocity field in an ODE.
        
        Args:
            xt (torch.Tensor): Current state at time t.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time.
                Shape: () or (1,)
        
        Returns:
            torch.Tensor: Drift coefficient.
                Shape: (batch_size, dim)
        """
        pass
    
    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion coefficient σ_t(x).
        
        The diffusion coefficient scales the random noise (Brownian motion).
        It can be:
        - Constant: σ_t(x) = σ (additive noise)
        - State-dependent: σ_t(x) = f(x) (multiplicative noise)
        - Time-dependent: σ_t(x) = σ(t)
        
        Larger diffusion leads to more stochastic behavior.
        
        Args:
            xt (torch.Tensor): Current state at time t.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time.
                Shape: () or (1,)
        
        Returns:
            torch.Tensor: Diffusion coefficient.
                Shape: (batch_size, dim)
                Can be scalar (same noise in all dimensions) or vector
        
        Note:
            In the Euler-Maruyama scheme, this is multiplied by
            sqrt(h) * N(0, I) where h is the time step.
        """
        pass


class Simulator(ABC):
    """
    Abstract base class for numerical simulation schemes.
    
    A Simulator discretizes continuous-time dynamics (ODE or SDE) into
    discrete time steps, allowing numerical integration from initial
    conditions to final states.
    
    The simulator follows the strategy pattern, allowing different
    numerical methods (Euler, Runge-Kutta, etc.) to be swapped
    without changing the interface.
    
    Subclasses must implement:
        - step: Perform one discrete time step
    
    Provided methods:
        - simulate: Integrate to final time (returns endpoint)
        - simulate_with_trajectory: Integrate and record full trajectory
    """
    
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Perform one discrete simulation step.
        
        This implements the numerical discretization scheme (e.g., Euler).
        Given the current state xt at time t, compute the next state at time t+h.
        
        Args:
            xt (torch.Tensor): Current state at time t.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time.
                Shape: () or (1,)
            h (torch.Tensor): Time step size (dt).
                Shape: () or (1,)
                Can be adaptive (varying h) or fixed
        
        Returns:
            torch.Tensor: Next state at time t + h.
                Shape: (batch_size, dim)
        
        Example:
            >>> sim = EulerSimulator(my_ode)
            >>> xt = torch.randn(100, 2)
            >>> t = torch.tensor(0.0)
            >>> h = torch.tensor(0.01)
            >>> xt_next = sim.step(xt, t, h)
        """
        pass
    
    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Simulate from initial state to final time.
        
        Integrates the dynamics over the provided time discretization,
        returning only the final state. This is memory-efficient when
        intermediate states are not needed.
        
        Args:
            x (torch.Tensor): Initial state at time ts[0].
                Shape: (batch_size, dim)
            ts (torch.Tensor): Time discretization.
                Shape: (num_timesteps,)
                Must be sorted: ts[0] < ts[1] < ... < ts[-1]
        
        Returns:
            torch.Tensor: Final state at time ts[-1].
                Shape: (batch_size, dim)
        
        Note:
            Uses @torch.no_grad() for memory efficiency.
            No gradients are computed during simulation.
        
        Example:
            >>> x0 = torch.randn(1000, 2)
            >>> ts = torch.linspace(0, 1, 100)
            >>> xT = simulator.simulate(x0, ts)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Simulate and record the full trajectory.
        
        Integrates the dynamics over the provided time discretization,
        storing the state at each time point. Useful for visualization
        and analysis of the evolution.
        
        Args:
            x (torch.Tensor): Initial state at time ts[0].
                Shape: (batch_size, dim)
            ts (torch.Tensor): Time discretization.
                Shape: (num_timesteps,)
        
        Returns:
            torch.Tensor: Full trajectory over time.
                Shape: (batch_size, num_timesteps, dim)
                - xs[:, 0, :] is the initial state
                - xs[:, i, :] is the state at time ts[i]
                - xs[:, -1, :] is the final state
        
        Warning:
            Memory usage scales with num_timesteps. For very long
            simulations, consider using simulate() with periodic checkpoints.
        
        Example:
            >>> x0 = torch.randn(100, 2)
            >>> ts = torch.linspace(0, 5, 500)
            >>> traj = simulator.simulate_with_trajectory(x0, ts)
            >>> # Plot trajectories
            >>> plt.plot(ts, traj[:, :, 0].T)
        """
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1), desc="Simulating"):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class Density(ABC):
    """
    Abstract base class for probability distributions with tractable density.
    
    A Density represents a probability distribution p(x) where we can
    evaluate the log probability density log p(x) at any point x.
    
    This is crucial for:
    - Score-based models: compute ∇log p(x)
    - Langevin dynamics: use score to drive sampling
    - Evaluation: measure likelihood of samples
    
    Subclasses must implement:
        - log_density: Compute log p(x)
    
    Provided methods:
        - score: Automatically compute ∇log p(x) using autograd
    """
    
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability density log p(x).
        
        Log density is preferred over raw density for numerical stability
        (avoids underflow/overflow) and computational convenience (sums
        instead of products).
        
        Args:
            x (torch.Tensor): Points at which to evaluate density.
                Shape: (batch_size, dim)
        
        Returns:
            torch.Tensor: Log density at each point.
                Shape: (batch_size, 1)
                Returns a column vector for consistency
        
        Properties:
            - log p(x) ∈ (-∞, 0] for normalized densities
            - Higher values indicate higher probability
            - Negative infinity indicates zero probability
        
        Example:
            >>> density = MyDensity()
            >>> x = torch.randn(100, 2)
            >>> log_p = density.log_density(x)  # shape: (100, 1)
        """
        pass
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the score function ∇_x log p(x).
        
        The score function is the gradient of the log density with respect
        to the input. It points in the direction of increasing probability
        density and is fundamental to:
        - Langevin dynamics
        - Score-based generative models
        - Diffusion models
        
        This implementation uses automatic differentiation (autograd) to
        compute the gradient, so subclasses only need to implement log_density.
        
        Args:
            x (torch.Tensor): Points at which to compute score.
                Shape: (batch_size, dim)
        
        Returns:
            torch.Tensor: Score (gradient) at each point.
                Shape: (batch_size, dim)
                Same shape as input
        
        Mathematical Definition:
            score(x) = ∇_x log p(x) = (∂/∂x_1 log p, ..., ∂/∂x_d log p)
        
        Properties:
            - Points toward regions of higher probability
            - Used in Langevin dynamics: dX_t = (σ²/2) score(X_t) dt + σ dW_t
        
        Implementation Details:
            Uses torch.func.jacrev (Jacobian reverse-mode) with vmap for
            efficient batched computation of gradients.
        
        Example:
            >>> density = Gaussian(mean=torch.zeros(2), cov=torch.eye(2))
            >>> x = torch.tensor([[1.0, 0.0]])
            >>> s = density.score(x)  # Points toward mean: [-1.0, 0.0]
        """
        from torch.func import jacrev, vmap
        
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        # Compute gradient for each sample in batch
        score = vmap(jacrev(self.log_density))(x)  # (batch_size, 1, 1, 1, dim)
        return score.squeeze((1, 2, 3))  # (batch_size, dim)


class Sampleable(ABC):
    """
    Abstract base class for distributions that can be sampled from.
    
    A Sampleable distribution allows drawing random samples x ~ p(x).
    This is essential for:
    - Generating training data
    - Initializing SDE simulations
    - Monte Carlo estimation
    
    Not all distributions are Sampleable:
    - Gaussian: Sampleable (Box-Muller, Cholesky)
    - Mixture of Gaussians: Sampleable
    - Unnormalized energy-based models: NOT directly sampleable
    
    Subclasses must implement:
        - sample: Draw random samples from p(x)
    """
    
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Draw random samples from the distribution.
        
        Generates independent and identically distributed (i.i.d.) samples
        from the probability distribution p(x).
        
        Args:
            num_samples (int): Number of samples to generate.
                Must be positive integer
        
        Returns:
            torch.Tensor: Random samples from p(x).
                Shape: (num_samples, dim)
                Each row is an independent sample
        
        Properties:
            - Samples should be independent
            - Distribution should match p(x) as num_samples → ∞
            - Randomness controlled by torch.manual_seed()
        
        Example:
            >>> dist = Gaussian(mean=torch.zeros(2), cov=torch.eye(2))
            >>> samples = dist.sample(1000)  # shape: (1000, 2)
            >>> # Empirical mean should be close to [0, 0]
            >>> samples.mean(dim=0)  # ≈ [0, 0]
        
        Note:
            For reproducibility, set random seed before sampling:
            >>> torch.manual_seed(42)
            >>> samples = dist.sample(100)
        """
        pass