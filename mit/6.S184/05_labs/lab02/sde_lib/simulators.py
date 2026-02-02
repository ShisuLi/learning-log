"""
Numerical Simulation Schemes for ODEs and SDEs

This module implements concrete numerical methods for integrating
ordinary and stochastic differential equations:

- EulerSimulator: First-order method for ODEs
- EulerMaruyamaSimulator: First-order method for SDEs

These methods discretize continuous-time dynamics into computable
steps, trading accuracy for computational feasibility.

"""

import torch
from .base import Simulator, ODE, SDE


class EulerSimulator(Simulator):
    """
    Euler method for simulating Ordinary Differential Equations.
    
    The Euler method is the simplest numerical scheme for ODEs, based on
    the first-order Taylor expansion:
    
        dX_t = u_t(X_t) dt
        →  X_{t+h} ≈ X_t + h · u_t(X_t)
    
    Properties:
        - Order of convergence: O(h)
        - Local error: O(h²)
        - Global error: O(h)
        - Stability: Conditionally stable (requires small h)
    
    Advantages:
        - Simple to implement
        - Low computational cost per step
        - Works for any ODE
    
    Disadvantages:
        - Low accuracy (requires many small steps)
        - Can be unstable for stiff equations
        - Better methods exist (RK4, adaptive schemes)
    
    When to use:
        - Educational purposes
        - Smooth, non-stiff problems
        - When simplicity is more important than accuracy
    
    Example:
        >>> # Simulate dx/dt = -x (exponential decay)
        >>> class ExponentialDecay(ODE):
        ...     def drift_coefficient(self, xt, t):
        ...         return -xt
        >>> ode = ExponentialDecay()
        >>> sim = EulerSimulator(ode)
        >>> x0 = torch.tensor([[1.0]])
        >>> ts = torch.linspace(0, 5, 100)
        >>> xT = sim.simulate(x0, ts)  # ≈ exp(-5) ≈ 0.007
    """
    
    def __init__(self, ode: ODE):
        """
        Initialize Euler simulator for a given ODE.
        
        Args:
            ode (ODE): The ODE system to simulate.
                Must implement drift_coefficient(xt, t)
        """
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Perform one Euler step: X_{t+h} = X_t + h · u_t(X_t).
        
        This is the forward Euler method, which approximates the integral
        of the drift coefficient over the interval [t, t+h] using the
        left endpoint (X_t).
        
        Args:
            xt (torch.Tensor): Current state at time t.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time.
                Shape: ()
            h (torch.Tensor): Step size.
                Shape: ()
                Smaller h → more accurate but more expensive
        
        Returns:
            torch.Tensor: Next state at time t+h.
                Shape: (batch_size, dim)
        
        Numerical Analysis:
            Local truncation error: O(h²)
            - True solution: X(t+h) = X(t) + h·u(X(t)) + O(h²)
            - Euler approximation: X_{t+h} = X_t + h·u(X_t)
            - Error per step: O(h²)
            - Error over [0,T]: O(h) (T/h steps, each with O(h²) error)
        
        Example:
            >>> xt = torch.tensor([[1.0, 2.0]])
            >>> t = torch.tensor(0.0)
            >>> h = torch.tensor(0.01)
            >>> # If drift is -X, then:
            >>> xt_next = xt + h * (-xt)  # ≈ [0.99, 1.98]
        """
        drift = self.ode.drift_coefficient(xt, t)
        return xt + h * drift


class EulerMaruyamaSimulator(Simulator):
    """
    Euler-Maruyama method for simulating Stochastic Differential Equations.
    
    The Euler-Maruyama method extends the Euler method to SDEs by approximating
    both the drift and diffusion terms:
    
        dX_t = u_t(X_t) dt + σ_t(X_t) dW_t
        →  X_{t+h} ≈ X_t + h · u_t(X_t) + √h · σ_t(X_t) · Z_t
    
    where Z_t ~ N(0, I) is standard Gaussian noise.
    
    Key Properties:
        - Strong convergence order: O(h^{1/2})
        - Weak convergence order: O(h)
        - The √h scaling is essential (see Theory section below)
    
    Theory - Why √h ?:
        Brownian motion has independent increments:
            dW_t ~ N(0, dt)  ⟹  √(dt) · N(0,1)
        
        Variance scales linearly with time:
            Var(W_t - W_s) = |t - s|
        
        Therefore, discretization requires:
            dW_t ≈ √h · Z_t,  where Z_t ~ N(0,1)
        
        Using just h (without √h) would give Var = O(h²), which is wrong!
    
    Stability:
        - More stable than deterministic Euler for some problems
        - Noise can help escape local minima
        - Still requires reasonably small h
    
    When to use:
        - Standard choice for SDE simulation
        - Balance between simplicity and accuracy
        - Foundation for understanding advanced methods
    
    Example:
        >>> # Brownian motion: dX = σ dW
        >>> class BrownianMotion(SDE):
        ...     def __init__(self, sigma):
        ...         self.sigma = sigma
        ...     def drift_coefficient(self, xt, t):
        ...         return torch.zeros_like(xt)
        ...     def diffusion_coefficient(self, xt, t):
        ...         return torch.full_like(xt, self.sigma)
        >>> sde = BrownianMotion(sigma=1.0)
        >>> sim = EulerMaruyamaSimulator(sde)
        >>> x0 = torch.zeros(1000, 1)
        >>> ts = torch.linspace(0, 1, 100)
        >>> xT = sim.simulate(x0, ts)
        >>> xT.std()  # ≈ 1.0 (since Var(W_1) = 1)
    """
    
    def __init__(self, sde: SDE):
        """
        Initialize Euler-Maruyama simulator for a given SDE.
        
        Args:
            sde (SDE): The SDE system to simulate.
                Must implement:
                    - drift_coefficient(xt, t)
                    - diffusion_coefficient(xt, t)
        """
        self.sde = sde
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Perform one Euler-Maruyama step.
        
        Implements the discretization:
            X_{t+h} = X_t + h · u_t(X_t) + √h · σ_t(X_t) · Z_t
        
        where:
            - u_t(X_t) is the drift coefficient
            - σ_t(X_t) is the diffusion coefficient
            - Z_t ~ N(0, I) is fresh random noise
        
        Args:
            xt (torch.Tensor): Current state at time t.
                Shape: (batch_size, dim)
            t (torch.Tensor): Current time.
                Shape: ()
            h (torch.Tensor): Step size.
                Shape: ()
        
        Returns:
            torch.Tensor: Next state at time t+h.
                Shape: (batch_size, dim)
        
        Implementation Details:
            1. Compute drift: u_t(X_t)
            2. Compute diffusion: σ_t(X_t)
            3. Sample noise: Z_t ~ N(0, I)
            4. Update: X_t + h·u_t + √h·σ_t·Z_t
        
        Numerical Properties:
            - Each step introduces O(√h) random error
            - Over n = T/h steps, accumulated error is O(√(n·h)) = O(√T)
            - Independent of h for fixed T (remarkable!)
        
        Random Number Generation:
            Uses torch.randn_like(xt) which:
            - Samples from standard normal N(0,1)
            - Same shape as xt
            - Independent across batch and dimensions
            - Can be seeded via torch.manual_seed()
        
        Example:
            >>> # Ornstein-Uhlenbeck: dX = -θX dt + σ dW
            >>> xt = torch.tensor([[1.0]])
            >>> drift = -0.5 * xt  # θ = 0.5
            >>> diffusion = torch.tensor([[1.0]])  # σ = 1.0
            >>> h = torch.tensor(0.01)
            >>> noise = torch.randn_like(xt)
            >>> xt_next = xt + h*drift + torch.sqrt(h)*diffusion*noise
        """
        # Deterministic component
        drift = self.sde.drift_coefficient(xt, t)
        
        # Stochastic component
        diffusion = self.sde.diffusion_coefficient(xt, t)
        noise = torch.randn_like(xt)  # Z_t ~ N(0, I)
        
        # Euler-Maruyama update
        return xt + h * drift + torch.sqrt(h) * diffusion * noise