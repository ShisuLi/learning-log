"""
Numerical simulators for ODEs and SDEs - Production Version

Optimized implementations of Euler and Euler-Maruyama methods.
"""

import torch
from .base import Simulator, ODE, SDE


class EulerSimulator(Simulator):
    """
    Euler method for ODEs: X_{t+h} = X_t + h · u_t(X_t)
    
    Args:
        ode: ODE system to simulate
    """
    
    def __init__(self, ode: ODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform one Euler step."""
        drift = self.ode.drift_coefficient(xt, t, **kwargs)
        return xt + h * drift


class EulerMaruyamaSimulator(Simulator):
    """
    Euler-Maruyama method for SDEs:
    X_{t+h} = X_t + h · u_t(X_t) + √h · σ_t(X_t) · Z_t
    
    Args:
        sde: SDE system to simulate
    """
    
    def __init__(self, sde: SDE):
        self.sde = sde
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform one Euler-Maruyama step."""
        drift = self.sde.drift_coefficient(xt, t, **kwargs)
        diffusion = self.sde.diffusion_coefficient(xt, t, **kwargs)
        
        # Generate Brownian increment: √h · Z_t where Z_t ~ N(0, I)
        noise = torch.randn_like(xt)
        sqrt_h = torch.sqrt(h)
        
        return xt + h * drift + sqrt_h * diffusion * noise
