"""
SDE Library - Tools for Simulating and Analyzing Stochastic Differential Equations

This package provides a complete framework for working with ODEs and SDEs:

Core Components:
    - Base classes: Abstract interfaces for extensibility
    - Simulators: Numerical integration schemes (Euler, Euler-Maruyama)
    - Processes: Concrete SDE implementations (Brownian, OU, Langevin)
    - Densities: Probability distributions (Gaussian, Mixture)
    - Visualization: Plotting and animation tools

Quick Start:
    >>> from sde_lib import BrownianMotion, EulerMaruyamaSimulator
    >>> 
    >>> # Create Brownian motion
    >>> bm = BrownianMotion(sigma=1.0)
    >>> sim = EulerMaruyamaSimulator(bm)
    >>> 
    >>> # Simulate
    >>> import torch
    >>> x0 = torch.zeros(100, 2)
    >>> ts = torch.linspace(0, 5, 500)
    >>> xT = sim.simulate(x0, ts)

For Langevin Dynamics:
    >>> from sde_lib import LangevinSDE, GaussianMixture
    >>> 
    >>> # Define target distribution
    >>> target = GaussianMixture.symmetric_2D(nmodes=5, std=0.8, scale=8)
    >>> 
    >>> # Create Langevin SDE
    >>> langevin = LangevinSDE(sigma=0.7, density=target)
    >>> sim = EulerMaruyamaSimulator(langevin)
    >>> 
    >>> # Visualize evolution
    >>> from sde_lib import plot_distribution_evolution, Gaussian
    >>> source = Gaussian(torch.zeros(2), 15*torch.eye(2))
    >>> plot_distribution_evolution(
    ...     num_samples=5000,
    ...     source_distribution=source,
    ...     simulator=sim,
    ...     target_density=target,
    ...     timesteps=torch.linspace(0, 10, 2000),
    ...     plot_interval=500,
    ...     bins=150,
    ...     scale=12
    ... )

"""

# === Base Classes ===
from .base import (
    ODE,
    SDE,
    Simulator,
    Density,
    Sampleable
)

# === Numerical Simulators ===
from .simulators import (
    EulerSimulator,
    EulerMaruyamaSimulator
)

# === Stochastic Processes ===
from .processes import (
    BrownianMotion,
    OUProcess,
    LangevinSDE,
    ConditionalVectorFieldODE,
    ConditionalVectorFieldSDE,
    LearnedVectorFieldODE,
    LangevinFlowSDE
)

# === Probability Densities ===
from .densities import (
    Gaussian,
    GaussianMixture,
    CirclesSampleable,
    MoonsSampleable,
    CheckerboardSampleable
)

# === Schedules ===
from .schedules import (
    Alpha,
    Beta,
    LinearAlpha,
    SquareRootBeta
)

# === Conditional Probability Paths ===
from .paths import (
    ConditionalProbabilityPath,
    GaussianConditionalProbabilityPath,
    LinearConditionalProbabilityPath
)

# === Models ===
from .models import (
    build_mlp,
    MLPVectorField,
    MLPScore,
    ScoreFromVectorField
)

# === Trainers ===
from .trainers import (
    Trainer,
    ConditionalFlowMatchingTrainer,
    ConditionalScoreMatchingTrainer
)

# === Visualization Tools ===
from .visualization import (
    plot_trajectories_1d,
    hist2d_samples,
    hist2d_sampleable,
    scatter_sampleable,
    kdeplot_sampleable,
    imshow_density,
    contour_density,
    get_trajectory_snapshot_indices,
    plot_distribution_evolution,
    animate_distribution_evolution
)


# === Public API ===
__all__ = [
    # Base classes
    'ODE',
    'SDE',
    'Simulator',
    'Density',
    'Sampleable',
    
    # Simulators
    'EulerSimulator',
    'EulerMaruyamaSimulator',
    
    # Processes
    'BrownianMotion',
    'OUProcess',
    'LangevinSDE',
    'ConditionalVectorFieldODE',
    'ConditionalVectorFieldSDE',
    'LearnedVectorFieldODE',
    'LangevinFlowSDE',
    
    # Densities
    'Gaussian',
    'GaussianMixture',
    'CirclesSampleable',
    'MoonsSampleable',
    'CheckerboardSampleable',

    # Schedules
    'Alpha',
    'Beta',
    'LinearAlpha',
    'SquareRootBeta',

    # Paths
    'ConditionalProbabilityPath',
    'GaussianConditionalProbabilityPath',
    'LinearConditionalProbabilityPath',

    # Models
    'build_mlp',
    'MLPVectorField',
    'MLPScore',
    'ScoreFromVectorField',

    # Trainers
    'Trainer',
    'ConditionalFlowMatchingTrainer',
    'ConditionalScoreMatchingTrainer',
    
    # Visualization
    'plot_trajectories_1d',
    'hist2d_samples',
    'hist2d_sampleable',
    'scatter_sampleable',
    'kdeplot_sampleable',
    'imshow_density',
    'contour_density',
    'get_trajectory_snapshot_indices',
    'plot_distribution_evolution',
    'animate_distribution_evolution',
]

__version__ = '1.1.0'
__author__ = 'Shisu Li'
__email__ = 'liss22@mails.tsinghua.edu.cn'


# === Version Information ===
def get_version_info():
    """
    Get detailed version and dependency information.
    
    Returns:
        dict: Version information including:
            - version: Package version
            - torch_version: PyTorch version
            - device: Available compute devices
    """
    import torch
    
    info = {
        'sde_lib_version': __version__,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }
    
    if torch.cuda.is_available():
        info['cuda_device'] = torch.cuda.get_device_name(0)
    
    return info


# === Convenience Functions ===
def quick_langevin_demo(
    nmodes: int = 5,
    num_samples: int = 5000,
    sigma: float = 0.7,
    simulation_time: float = 10.0,
    num_timesteps: int = 2000,
    device: str = 'cpu'
):
    """
    Run a quick Langevin dynamics demonstration.
    
    Creates a symmetric Gaussian mixture target, runs Langevin dynamics
    from a broad Gaussian source, and visualizes the evolution.
    
    Args:
        nmodes (int): Number of modes in target distribution.
        num_samples (int): Number of particles to simulate.
        sigma (float): Diffusion coefficient for Langevin.
        simulation_time (float): Total simulation time.
        num_timesteps (int): Number of simulation steps.
        device (str): Computation device ('cpu' or 'cuda').
    
    Example:
        >>> import sde_lib
        >>> sde_lib.quick_langevin_demo(nmodes=8, num_samples=10000)
    """
    import torch
    
    print("=" * 60)
    print("SDE Library - Quick Langevin Dynamics Demo")
    print("=" * 60)
    
    # Create target distribution
    print(f"\n1. Creating target: {nmodes}-mode Gaussian mixture")
    target = GaussianMixture.symmetric_2D(
        nmodes=nmodes,
        std=0.8,
        scale=8.0
    ).to(device)
    
    # Create Langevin SDE
    print(f"2. Setting up Langevin dynamics (σ={sigma})")
    langevin = LangevinSDE(sigma=sigma, density=target)
    simulator = EulerMaruyamaSimulator(langevin)
    
    # Source distribution
    print("3. Initializing from broad Gaussian")
    source = Gaussian(
        mean=torch.zeros(2),
        cov=15 * torch.eye(2)
    ).to(device)
    
    # Simulate and visualize
    print(f"4. Simulating {num_samples} particles for t∈[0, {simulation_time}]")
    print(f"   Using {num_timesteps} timesteps (dt={simulation_time/num_timesteps:.4f})")
    
    plot_distribution_evolution(
        num_samples=num_samples,
        source_distribution=source,
        simulator=simulator,
        target_density=target,
        timesteps=torch.linspace(0, simulation_time, num_timesteps).to(device),
        plot_interval=num_timesteps // 3,  # 3 snapshots
        bins=150,
        scale=12,
        device=device
    )
    
    print("\n✓ Demo complete!")
    print("=" * 60)


# === Module Initialization ===
def _check_dependencies():
    """
    Check if all required dependencies are available.
    
    Raises:
        ImportError: If critical dependencies are missing.
    """
    import importlib
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm'
    }
    
    optional = {
        'celluloid': 'Celluloid (for animations)',
    }
    
    missing_required = []
    missing_optional = []
    
    for module, name in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_required.append(name)
    
    for module, name in optional.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_optional.append(name)
    
    if missing_required:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_required)}\n"
            f"Install with: pip install torch numpy matplotlib seaborn tqdm"
        )
    
    if missing_optional:
        import warnings
        warnings.warn(
            f"Optional dependencies not found: {', '.join(missing_optional)}\n"
            f"Some features (animations) may not work.\n"
            f"Install with: pip install celluloid",
            UserWarning
        )


# Run dependency check on import
try:
    _check_dependencies()
except ImportError as e:
    import sys
    print(f"ERROR: {e}", file=sys.stderr)
    raise


# === Welcome Message ===
def _print_welcome():
    """Print welcome message on first import."""
    import sys
    
    # Only print in interactive sessions
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        print(f"SDE Library v{__version__} loaded successfully!")
        print(f"Try: sde_lib.quick_langevin_demo()")


# Uncomment to enable welcome message
# _print_welcome()