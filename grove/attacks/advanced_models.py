"""
Model pruning implementation for advanced model stealing attacks.

"""

import torch
import torch.nn as nn


class ModelPruner:
    """
    Class for pruning models as part of advanced attacks.
    """
    
    def __init__(self, pruning_ratio: float = 0.1):
        """
        Initialize pruner.
        
        Args:
            pruning_ratio: Fraction of parameters to prune
        """
        self.pruning_ratio = pruning_ratio
    
    def random_prune(self, model: nn.Module) -> nn.Module:
        """
        Apply random pruning.
        This should cause significant degradation at high ratios.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model (copy)
        """
        import copy
        
        pruned_model = copy.deepcopy(model)
        
        # Apply pruning to parameters 
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:  # Only prune trainable parameters
                bitmask = torch.rand_like(param) > self.pruning_ratio
                with torch.no_grad():
                    param.copy_(torch.mul(param, bitmask.float()))
        
        return pruned_model
    
    def verify_pruning_ratio(self, original_model: nn.Module, pruned_model: nn.Module) -> float:
        """
        Verify what percentage of weights were actually set to zero.
        
        Args:
            original_model: Original model before pruning
            pruned_model: Model after pruning
            
        Returns:
            Actual pruning ratio (fraction of weights set to zero)
        """
        total_params = 0
        zero_params = 0
        
        for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), 
            pruned_model.named_parameters()
        ):
            if param1.requires_grad:
                total_params += param2.numel()
                zero_params += (param2 == 0).sum().item()
        
        actual_ratio = zero_params / total_params if total_params > 0 else 0
        return actual_ratio 