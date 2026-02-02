"""
Training Utilities for Flow Matching and Score Matching

This module provides reusable trainer classes for conditional flow matching
and conditional score matching, built on top of a generic Trainer base class.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from tqdm import tqdm
from .paths import ConditionalProbabilityPath
from .models import MLPVectorField, MLPScore


def _resolve_device(model: torch.nn.Module, device: Optional[str]) -> torch.device:
    """
    Resolve device for training.
    
    Args:
        model (torch.nn.Module): Model to train.
        device (Optional[str]): User-specified device.
    
    Returns:
        torch.device: Target device.
    """
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


class Trainer(ABC):
    """
    Base trainer class.
    
    Subclasses must implement get_train_loss.
    """
    
    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = _resolve_device(model, device)
    
    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        """
        Compute training loss for a batch.
        """
        pass
    
    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """
        Build optimizer for training.
        """
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train(self, num_epochs: int, lr: float = 1e-3, **kwargs):
        """
        Train the model.
        
        Args:
            num_epochs (int): Number of training iterations
            lr (float): Learning rate
            **kwargs: Additional args passed to get_train_loss
        
        Returns:
            list: Recorded losses (optional extension)
        """
        self.model.to(self.device)
        opt = self.get_optimizer(lr)
        self.model.train()
        pbar = tqdm(enumerate(range(num_epochs)))
        losses = []
        for idx, _ in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')
        self.model.eval()
        return losses


class ConditionalFlowMatchingTrainer(Trainer):
    """
    Trainer for Conditional Flow Matching (CFM).
    
    Objective:
        L_CFM = E[ || u_t^θ(x) - u_t^{ref}(x|z) ||^2 ]
    """
    
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, device: Optional[str] = None):
        super().__init__(model, device=device)
        self.path = path
    
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute CFM loss.
        
        Args:
            batch_size (int): Number of samples in batch
        
        Returns:
            torch.Tensor: Scalar loss
        """
        z = self.path.sample_conditioning_variable(batch_size)
        t = torch.rand(batch_size, 1).to(z)
        x = self.path.sample_conditional_path(z, t)
        ut_theta = self.model(x, t)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        loss = torch.sum(torch.square(ut_theta - ut_ref), dim=-1)
        return torch.mean(loss)


class ConditionalScoreMatchingTrainer(Trainer):
    """
    Trainer for Conditional Score Matching (CSM).
    
    Objective:
        L_CSM = E[ || s_t^θ(x) - ∇_x log p_t(x|z) ||^2 ]
    """
    
    def __init__(self, path: ConditionalProbabilityPath, model: MLPScore, device: Optional[str] = None):
        super().__init__(model, device=device)
        self.path = path
    
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute CSM loss.
        
        Args:
            batch_size (int): Number of samples in batch
        
        Returns:
            torch.Tensor: Scalar loss
        """
        z = self.path.sample_conditioning_variable(batch_size)
        t = torch.rand(batch_size, 1).to(z)
        x = self.path.sample_conditional_path(z, t)
        s_theta = self.model(x, t)
        s_ref = self.path.conditional_score(x, z, t)
        loss = torch.sum(torch.square(s_theta - s_ref), dim=-1)
        return torch.mean(loss)
