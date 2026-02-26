"""
SDELib - Production-Ready Flow Matching Library

A minimal, efficient library for training and sampling from flow-based generative models.
Extracted from MIT 6.S184 course materials, optimized for production use.

Quick Start:
    >>> from sdelib import (
    ...     GaussianConditionalProbabilityPath,
    ...     UNet,
    ...     CFGTrainer,
    ...     CFGVectorFieldODE,
    ...     EulerSimulator,
    ...     LinearAlpha,
    ...     LinearBeta
    ... )

Core Components:
    - Base classes: ODE, SDE, Simulator, ConditionalProbabilityPath
    - Simulators: EulerSimulator, EulerMaruyamaSimulator
    - Schedules: LinearAlpha, LinearBeta, SquareRootBeta
    - Paths: GaussianConditionalProbabilityPath
    - Models: UNet with classifier-free guidance
    - Processes: CFGVectorFieldODE for inference
    - Trainers: CFGTrainer for efficient training
"""

# Base classes
from .base import (
    ODE,
    SDE,
    Simulator,
    Sampleable
)

# Simulators
from .simulators import (
    EulerSimulator,
    EulerMaruyamaSimulator
)

# Schedules
from .schedules import (
    Alpha,
    Beta,
    LinearAlpha,
    LinearBeta,
    SquareRootBeta
)

# Probability paths
from .paths import (
    ConditionalProbabilityPath,
    GaussianConditionalProbabilityPath,
    LinearConditionalProbabilityPath,
    IsotropicGaussian
)

# Models
from .models import (
    ConditionalVectorField,
    UNet,
    FourierEncoder
)

# Processes
from .processes import (
    CFGVectorFieldODE
)

# Trainers
from .trainers import (
    Trainer,
    CFGTrainer
)

__version__ = "1.0.0"

__all__ = [
    # Base
    "ODE",
    "SDE",
    "Simulator",
    "Sampleable",
    # Simulators
    "EulerSimulator",
    "EulerMaruyamaSimulator",
    # Schedules
    "Alpha",
    "Beta",
    "LinearAlpha",
    "LinearBeta",
    "SquareRootBeta",
    # Paths
    "ConditionalProbabilityPath",
    "GaussianConditionalProbabilityPath",
    "IsotropicGaussian",
    # Models
    "ConditionalVectorField",
    "UNet",
    "FourierEncoder",
    # Processes
    "CFGVectorFieldODE",
    # Trainers
    "Trainer",
    "CFGTrainer",
]
