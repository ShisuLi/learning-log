"""
Training utilities for flow matching - Production Version

Provides efficient trainer for classifier-free guidance.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from tqdm import tqdm


def model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in MB."""
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / (1024 ** 2)


class Trainer(ABC):
    """
    Base trainer class.
    
    Args:
        model: Neural network to train
        device: Target device (inferred from model if None)
    """
    
    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = self._resolve_device(model, device)
    
    @staticmethod
    def _resolve_device(model: torch.nn.Module, device: Optional[str]) -> torch.device:
        """Resolve device for training."""
        if device is not None:
            return torch.device(device)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        """Compute training loss for a batch."""
        pass
    
    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Build optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train(self, num_epochs: int, lr: float = 1e-3, **kwargs):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training iterations
            lr: Learning rate
            **kwargs: Additional args passed to get_train_loss
        
        Returns:
            List of losses per epoch
        """
        print(f'Training model with size: {model_size_mb(self.model):.3f} MiB')
        
        self.model.to(self.device)
        opt = self.get_optimizer(lr)
        self.model.train()
        
        losses = []
        pbar = tqdm(range(num_epochs))
        for idx in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.4f}')
        
        self.model.eval()
        return losses


class CFGTrainer(Trainer):
    """
    Classifier-free guidance trainer for conditional flow matching.
    
    Implements CFG training objective with label dropping:
        𝐏_CFM(θ) = 𝔼[ ‖ u_t^θ(x|y) - u_t^{ref}(x|z) ‖² ]
    
    where labels are randomly replaced with null label ∅ with probability η.
    
    Args:
        path: Gaussian conditional probability path
        model: Conditional vector field network
        eta: Label dropout probability (0 < η < 1)
        null_label: Null label value (default: 10)
        device: Target device
    """
    
    def __init__(self, path, model, eta: float, null_label: int = 10, device: Optional[str] = None):
        assert 0 < eta < 1, "eta must be in (0, 1)"
        super().__init__(model, device=device)
        self.eta = eta
        self.null_label = null_label
        self.path = path.to(self.device)
    
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute CFG training loss.
        
        Procedure:
            1. Sample (z, y) ~ p_data
            2. Randomly drop labels: y → ∅ with probability η
            3. Sample t ~ U[0,1] and x ~ p_t(x|z)
            4. Compute MSE: ‖u_t^θ(x|y) - u_t^{ref}(x|z)‖²
        
        Args:
            batch_size: Number of samples per batch
        
        Returns:
            Scalar loss
        """
        # Sample (z, y) from data
        z, y = self.path.p_data.sample(batch_size)
        z = z.to(self.device)
        y = y.to(self.device)
        
        # Label dropout: y → ∅ with probability η
        mask = torch.rand(batch_size, device=self.device) < self.eta
        y = y.clone()
        y[mask] = self.null_label
        
        # Sample t and x ~ p_t(x|z)
        t = torch.rand(batch_size, 1, 1, 1, device=self.device)
        x = self.path.sample_conditional_path(z, t)
        
        # Compute loss: ‖u_θ - u_ref‖²
        ut_theta = self.model(x, t, y)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        
        loss = torch.sum(torch.square(ut_theta - ut_ref), dim=[1, 2, 3]).mean()
        return loss
