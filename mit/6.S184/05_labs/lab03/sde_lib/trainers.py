"""
Training Utilities for Flow Matching and Score Matching

This module provides reusable trainer classes for conditional flow matching
and conditional score matching, built on top of a generic Trainer base class.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from tqdm import tqdm
from .paths import ConditionalProbabilityPath, GaussianConditionalProbabilityPath
from .models import MLPVectorField, MLPScore, ConditionalVectorField


MiB = 1024 ** 2


def model_size_b(model: torch.nn.Module) -> int:
    """
    Calculate model size in bytes.
    
    Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    
    Args:
        model (torch.nn.Module): The model to measure.
    
    Returns:
        int: Total size in bytes (parameters + buffers).
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


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
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
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


class CFGTrainer(Trainer):
    """
    Classifier-free guidance trainer for conditional flow matching.

    Implements lab03 objective: optionally drop labels with probability eta
    and regress model vector field against reference conditional vector field.
    """

    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, device: Optional[str] = None):
        assert 0 < eta < 1, "eta must be in (0,1)"
        super().__init__(model, device=device)
        self.eta = eta
        self.path = path.to(self.device)

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute CFG training loss for conditional flow matching.
        
        Training objective:
            𝐏_CFM(θ) = 𝔼[ ‖ u_t^θ(x|y) - u_t^{ref}(x|z) ‖² ]
        
        where:
            - (z, y) ~ p_data(z, y)  : sample image z and label y from MNIST
            - With probability η, replace y → ∅ = 10  (null label for unconditional)
            - t ~ U[0, 1]              : uniform time
            - x ~ p_t(x|z)             : noisy sample from conditional path
            - u_t^θ(x|y)              : learned vector field (model prediction)
            - u_t^{ref}(x|z)           : reference vector field (regression target)
        
        Args:
            batch_size (int): Number of training samples per batch.
        
        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        # Step 1: Sample (z, y) from p_data (MNIST images + labels)
        z, y = self.path.p_data.sample(batch_size)  # z: (bs, 1, 32, 32), y: (bs,)
        z = z.to(self.device)
        y = y.to(self.device)

        # Step 2: Randomly drop labels with probability η (set to null label ∅ = 10)
        #         This enables classifier-free guidance by training both
        #         conditional u_t(x|y) and unconditional u_t(x|∅) modes
        mask = torch.rand(batch_size, device=self.device) < self.eta
        y = y.clone()
        y[mask] = 10  # ∅ (null label)

        # Step 3: Sample t ~ U[0,1] and x ~ p_t(x|z)
        t = torch.rand(batch_size, 1, 1, 1, device=self.device)  # (bs, 1, 1, 1) for broadcasting
        x = self.path.sample_conditional_path(z, t)              # (bs, 1, 32, 32)

        # Step 4: Regress model vector field u_t^θ(x|y) to reference u_t^{ref}(x|z)
        ut_theta = self.model(x, t, y)                           # Model prediction
        ut_ref = self.path.conditional_vector_field(x, z, t)     # Ground truth target
        
        # Compute MSE loss: E[‖u_θ - u_ref‖²]
        loss = torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref)).mean()
        
        return loss
