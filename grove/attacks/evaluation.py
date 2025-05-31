"""
Comprehensive evaluation module for model stealing attacks.
Provides detailed metrics including accuracy, fidelity, FPR, FNR with CSV output.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime


class ModelEvaluator:
    """
    Comprehensive evaluator for model stealing attacks with detailed metrics.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            device: Device to use for computations
        """
        self.device = device
        self.results_history = []
        
    def compute_classification_metrics(self, 
                                     y_true: torch.Tensor, 
                                     y_pred: torch.Tensor,
                                     y_pred_proba: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with classification metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if y_pred_proba is not None and isinstance(y_pred_proba, torch.Tensor):
            y_pred_proba = y_pred_proba.detach().cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Get per-class metrics first (without averaging)
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Compute weighted averages manually
        total_support = np.sum(support_per_class) if support_per_class is not None else len(y_true)
        
        if total_support > 0 and support_per_class is not None:
            precision = np.average(precision_per_class, weights=support_per_class)
            recall = np.average(recall_per_class, weights=support_per_class)
            f1 = np.average(f1_per_class, weights=support_per_class)
        else:
            # Fallback to simple averages
            precision = np.mean(precision_per_class) if len(precision_per_class) > 0 else 0.0
            recall = np.mean(recall_per_class) if len(recall_per_class) > 0 else 0.0
            f1 = np.mean(f1_per_class) if len(f1_per_class) > 0 else 0.0
        
        # Confusion matrix for FPR/FNR calculation
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate FPR and FNR per class, then average
        num_classes = cm.shape[0]
        fpr_per_class = []
        fnr_per_class = []
        
        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            # FPR = FP / (FP + TN)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            # FNR = FN / (FN + TP)  
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            fpr_per_class.append(fpr)
            fnr_per_class.append(fnr)
        
        # Average FPR and FNR (weighted by support if available)
        if support_per_class is not None and total_support > 0:
            weighted_fpr = np.average(fpr_per_class, weights=support_per_class)
            weighted_fnr = np.average(fnr_per_class, weights=support_per_class)
        else:
            # Simple average as fallback
            weighted_fpr = np.mean(fpr_per_class) if len(fpr_per_class) > 0 else 0.0
            weighted_fnr = np.mean(fnr_per_class) if len(fnr_per_class) > 0 else 0.0
        
        # AUC if probabilities available
        auc_score = 0.0
        if y_pred_proba is not None and num_classes >= 2:
            try:
                if num_classes == 2:
                    auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc_score = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'fpr': float(weighted_fpr),
            'fnr': float(weighted_fnr),
            'auc': float(auc_score)
        }
        
        return metrics
    
    def compute_fidelity(self, 
                        target_preds: torch.Tensor, 
                        surrogate_preds: torch.Tensor,
                        return_detailed: bool = False) -> Dict[str, float]:
        """
        Compute fidelity between target and surrogate model predictions.
        
        Args:
            target_preds: Target model predictions (logits or probabilities)
            surrogate_preds: Surrogate model predictions (logits or probabilities)
            return_detailed: Whether to return detailed fidelity metrics
            
        Returns:
            Dictionary with fidelity metrics
        """
        # Convert to probabilities if needed
        if target_preds.requires_grad:
            target_preds = target_preds.detach()
        if surrogate_preds.requires_grad:
            surrogate_preds = surrogate_preds.detach()
        
        # Get predicted classes
        target_classes = target_preds.argmax(dim=1)
        surrogate_classes = surrogate_preds.argmax(dim=1)
        
        # Basic fidelity (agreement rate)
        fidelity = (target_classes == surrogate_classes).float().mean().item()
        
        result = {'fidelity': fidelity}
        
        if return_detailed:
            # Convert to probabilities for detailed analysis
            target_probs = F.softmax(target_preds, dim=1)
            surrogate_probs = F.softmax(surrogate_preds, dim=1)
            
            # KL divergence
            kl_div = F.kl_div(
                F.log_softmax(surrogate_preds, dim=1),
                target_probs,
                reduction='batchmean'
            ).item()
            
            # L2 distance between probability distributions
            l2_dist = torch.norm(target_probs - surrogate_probs, p=2, dim=1).mean().item()
            
            # Cosine similarity between probability distributions
            cos_sim = F.cosine_similarity(target_probs, surrogate_probs, dim=1).mean().item()
            
            result.update({
                'kl_divergence': kl_div,
                'l2_distance': l2_dist,
                'cosine_similarity': cos_sim
            })
        
        return result
    
    def compute_embedding_similarity(self, 
                                   target_embs: torch.Tensor,
                                   surrogate_embs: torch.Tensor) -> Dict[str, float]:
        """
        Compute similarity metrics between target and surrogate embeddings.
        
        Args:
            target_embs: Target model embeddings
            surrogate_embs: Surrogate model embeddings
            
        Returns:
            Dictionary with embedding similarity metrics
        """
        if target_embs.requires_grad:
            target_embs = target_embs.detach()
        if surrogate_embs.requires_grad:
            surrogate_embs = surrogate_embs.detach()
        
        # Normalize embeddings
        target_norm = F.normalize(target_embs, p=2, dim=1)
        surrogate_norm = F.normalize(surrogate_embs, p=2, dim=1)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(target_norm, surrogate_norm, dim=1).mean().item()
        
        # L2 distance
        l2_dist = torch.norm(target_embs - surrogate_embs, p=2, dim=1).mean().item()
        
        # L1 distance  
        l1_dist = torch.norm(target_embs - surrogate_embs, p=1, dim=1).mean().item()
        
        # Pearson correlation coefficient (per dimension, then average)
        correlations = []
        for dim in range(target_embs.shape[1]):
            target_dim = target_embs[:, dim]
            surrogate_dim = surrogate_embs[:, dim]
            
            # Compute correlation
            target_centered = target_dim - target_dim.mean()
            surrogate_centered = surrogate_dim - surrogate_dim.mean()
            
            numerator = (target_centered * surrogate_centered).sum()
            denominator = torch.sqrt((target_centered ** 2).sum() * (surrogate_centered ** 2).sum())
            
            if denominator > 1e-8:
                corr = (numerator / denominator).item()
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'embedding_cosine_similarity': cos_sim,
            'embedding_l2_distance': l2_dist,
            'embedding_l1_distance': l1_dist,
            'embedding_correlation': avg_correlation
        }
    def evaluate_surrogate_model(self,
                               target_model: torch.nn.Module,
                               surrogate_trainer,
                               test_data,
                               dataset_name: str,
                               attack_config: Dict) -> Dict[str, float]:
        """
        Comprehensive evaluation of surrogate model performance.
        
        Args:
            target_model: Target model
            surrogate_trainer: Trained surrogate model trainer
            test_data: Test data
            dataset_name: Name of dataset
            attack_config: Attack configuration
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        target_model.eval()
        surrogate_trainer.surrogate_model.eval()
        surrogate_trainer.classifier.eval()
        
        with torch.no_grad():
            # Get target model outputs
            target_embs, target_preds = target_model(test_data)
            target_classes = target_preds.argmax(dim=1)
            target_probs = F.softmax(target_preds, dim=1)
            
            # Get surrogate model outputs
            surrogate_embs = surrogate_trainer.surrogate_model(test_data)
            surrogate_logits = surrogate_trainer.classifier(surrogate_embs)
            surrogate_classes = surrogate_logits.argmax(dim=1)
            surrogate_probs = F.softmax(surrogate_logits, dim=1)
        
        # Ground truth labels
        true_labels = test_data.y
        
        # 1. Target model performance
        target_metrics = self.compute_classification_metrics(
            true_labels, target_classes, target_probs
        )
        target_metrics = {f'target_{k}': v for k, v in target_metrics.items()}
        
        # 2. Surrogate model performance  
        surrogate_metrics = self.compute_classification_metrics(
            true_labels, surrogate_classes, surrogate_probs
        )
        surrogate_metrics = {f'surrogate_{k}': v for k, v in surrogate_metrics.items()}
        
        # 3. Fidelity metrics
        fidelity_metrics = self.compute_fidelity(target_preds, surrogate_logits, return_detailed=True)
        
        # 4. Embedding similarity metrics
        embedding_metrics = self.compute_embedding_similarity(target_embs, surrogate_embs)
        
        # 5. Additional attack-specific metrics
        accuracy_gap = abs(target_metrics['target_accuracy'] - surrogate_metrics['surrogate_accuracy'])
        fpr_gap = abs(target_metrics['target_fpr'] - surrogate_metrics['surrogate_fpr'])
        fnr_gap = abs(target_metrics['target_fnr'] - surrogate_metrics['surrogate_fnr'])
        
        # Combine all metrics
        all_metrics = {
            **target_metrics,
            **surrogate_metrics,
            **fidelity_metrics,
            **embedding_metrics,
            'accuracy_gap': accuracy_gap,
            'fpr_gap': fpr_gap,
            'fnr_gap': fnr_gap,
            'dataset': dataset_name,
            'attack_type': attack_config.get('structure', 'original'),
            'surrogate_architecture': attack_config.get('surrogate_architecture', 'unknown'),
            'recovery_from': attack_config.get('recovery_from', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        return all_metrics

    def save_results_to_csv(self,
                           results_df: pd.DataFrame,
                           output_dir: str,
                           filename_prefix: str = "model_stealing_evaluation") -> str:
        """
        Save evaluation results to CSV file.
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Output directory
            filename_prefix: Prefix for filename
            
        Returns:
            Path to saved CSV file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        csv_filename = f"Model_Stealing_Evaluation_{filename_prefix}.csv"
        csv_path = output_path / csv_filename
        
        # Save to CSV
        results_df.to_csv(csv_path, index=False)
        
        print(f"💾 Evaluation results saved to: {csv_path}")
        
        return str(csv_path)
    