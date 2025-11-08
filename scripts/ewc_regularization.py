#!/usr/bin/env python3
"""
Elastic Weight Consolidation (EWC) for Continual Learning

Implements EWC regularization to prevent catastrophic forgetting in PEFT models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.

    Based on: https://arxiv.org/abs/1612.00796
    """

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 0.1  # Regularization strength

    def compute_fisher_information(self, dataloader, num_samples: int = 1000):
        """
        Compute Fisher Information Matrix for current model parameters.

        Args:
            dataloader: DataLoader with current task data
            num_samples: Number of samples to use for estimation
        """
        logger.info("Computing Fisher Information Matrix...")

        # Store original model parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora' in name:  # Only for LoRA parameters
                original_params[name] = param.clone().detach()

        # Initialize Fisher information
        self.fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora' in name:
                self.fisher_information[name] = torch.zeros_like(param)

        # Set model to evaluation mode
        self.model.eval()

        # Sample from dataloader
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break

            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(**inputs)
            loss = nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

            # Compute gradients
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients (Fisher information approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'lora' in name and param.grad is not None:
                    self.fisher_information[name] += param.grad.pow(2)

            sample_count += len(labels)

        # Normalize by number of samples
        for name in self.fisher_information:
            self.fisher_information[name] /= sample_count

        # Store current optimal parameters
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora' in name:
                self.optimal_params[name] = original_params[name].clone()

        logger.info(f"Computed Fisher information for {len(self.fisher_information)} parameters")

    def ewc_loss(self, current_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Args:
            current_params: Current model parameters

        Returns:
            EWC regularization loss
        """
        if not self.fisher_information or not self.optimal_params:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for name, param in current_params.items():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]

                # EWC loss: λ * Σ(F_i * (θ_i - θ*_i)²)
                param_diff = param - optimal
                loss += torch.sum(fisher * param_diff.pow(2))

        return self.ewc_lambda * loss

    def save_ewc_state(self, path: str):
        """Save EWC state to disk"""
        state = {
            'fisher_information': {k: v.cpu() for k, v in self.fisher_information.items()},
            'optimal_params': {k: v.cpu() for k, v in self.optimal_params.items()},
            'ewc_lambda': self.ewc_lambda
        }

        torch.save(state, path)
        logger.info(f"Saved EWC state to {path}")

    def load_ewc_state(self, path: str):
        """Load EWC state from disk"""
        if not Path(path).exists():
            logger.warning(f"EWC state file not found: {path}")
            return

        state = torch.load(path, map_location='cpu')

        self.fisher_information = {k: v.to(self.device) for k, v in state['fisher_information'].items()}
        self.optimal_params = {k: v.to(self.device) for k, v in state['optimal_params'].items()}
        self.ewc_lambda = state.get('ewc_lambda', 0.1)

        logger.info(f"Loaded EWC state from {path}")

    def update_lambda(self, new_lambda: float):
        """Update EWC regularization strength"""
        self.ewc_lambda = new_lambda
        logger.info(f"Updated EWC lambda to {new_lambda}")

    def get_ewc_stats(self) -> Dict[str, Any]:
        """Get statistics about EWC state"""
        if not self.fisher_information:
            return {'initialized': False}

        fisher_norms = [v.norm().item() for v in self.fisher_information.values()]
        param_norms = [v.norm().item() for v in self.optimal_params.values()]

        return {
            'initialized': True,
            'num_parameters': len(self.fisher_information),
            'ewc_lambda': self.ewc_lambda,
            'avg_fisher_norm': np.mean(fisher_norms),
            'avg_param_norm': np.mean(param_norms),
            'max_fisher_norm': max(fisher_norms),
            'max_param_norm': max(param_norms)
        }

class EWCOptimizer:
    """Wrapper for optimizer with EWC regularization"""

    def __init__(self, optimizer, ewc: ElasticWeightConsolidation):
        self.optimizer = optimizer
        self.ewc = ewc

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        # Get current parameters
        current_params = {}
        for name, param in self.ewc.model.named_parameters():
            if param.requires_grad and 'lora' in name:
                current_params[name] = param

        # Compute EWC loss
        ewc_loss = self.ewc.ewc_loss(current_params)

        # Add EWC loss to model
        if ewc_loss > 0:
            # We need to backpropagate the EWC loss
            ewc_loss.backward()

        # Take optimization step
        loss = self.optimizer.step(closure)

        return loss

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

def create_ewc_dataloader(dataset, batch_size: int = 4, num_samples: int = 500):
    """
    Create a dataloader for Fisher information computation.

    Uses a subset of data for efficient computation.
    """
    from torch.utils.data import DataLoader, Subset

    # Sample subset for Fisher computation
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )

    return dataloader

def main():
    """Example usage of EWC"""
    from transformers import AutoModelForCausalLM

    # Load a small model for demonstration
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16
    )

    # Initialize EWC
    ewc = ElasticWeightConsolidation(model)

    print("EWC initialized for model with", sum(p.numel() for p in model.parameters()), "parameters")

    # In a real scenario, you would:
    # 1. Load or train initial model
    # 2. Compute Fisher information on current task data
    # 3. Use EWC during training on new tasks

    stats = ewc.get_ewc_stats()
    print("EWC stats:", stats)

if __name__ == "__main__":
    main()