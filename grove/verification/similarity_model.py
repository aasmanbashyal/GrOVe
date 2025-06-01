"""
Similarity Model (Csim) for GNN Model Verification.
Implements the Csim classifier to detect whether a suspect model is a surrogate or independent
using pre-saved embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Any, Optional, Union
import joblib
import os
from pathlib import Path


class SimilarityModel:
    """
    Csim - Similarity model for verifying if a suspect model is a surrogate or independent
    using pre-saved embeddings.
    """
    
    def __init__(self, 
                 target_model_name: str,
                 device: str = 'cuda',
                 random_state: int = 42):
        """
        Initialize Csim similarity model.
        
        Args:
            target_model_name: Name/ID of the target model this Csim is for
            device: Device to use for computations
            random_state: Random state for reproducibility
        """
        self.target_model_name = target_model_name
        self.device = device
        self.random_state = random_state
        self.classifier = None
        self.is_trained = False
        
    def _compute_distance_vector(self, 
                                embedding1: torch.Tensor, 
                                embedding2: torch.Tensor) -> np.ndarray:
        """
        Compute element-wise squared distance vector between two embeddings.
        
        Args:
            embedding1: First embedding [num_nodes, embedding_dim]
            embedding2: Second embedding [num_nodes, embedding_dim]
            
        Returns:
            Distance vectors [num_nodes, embedding_dim]
        """
        # Ensure embeddings are on CPU and converted to numpy
        emb1 = embedding1.detach().cpu().numpy() if torch.is_tensor(embedding1) else embedding1
        emb2 = embedding2.detach().cpu().numpy() if torch.is_tensor(embedding2) else embedding2
        
        # Compute element-wise squared distance
        distance_vector = (emb1 - emb2) ** 2
        
        return distance_vector
    
    def _load_embedding_from_file(self, embedding_path: str) -> torch.Tensor:
        """
        Load embeddings from a saved file.
        
        Args:
            embedding_path: Path to the embedding file
            
        Returns:
            Embeddings tensor
        """
        print(f"Loading embeddings from: {embedding_path}")
        
        # Load the embedding data
        embedding_data = torch.load(embedding_path, weights_only=False)
        
        # Extract embeddings from the saved format
        if isinstance(embedding_data, dict) and 'best_embeddings' in embedding_data:
            best_embeddings = embedding_data['best_embeddings']
            if 'verification' in best_embeddings:
                if isinstance(best_embeddings['verification'], Data):
                    embeddings = best_embeddings['verification'].x
                else:
                    embeddings = best_embeddings['verification']
            else:
                raise ValueError(f"No verification embeddings found in {embedding_path}")
        elif isinstance(embedding_data, Data):
            embeddings = embedding_data.x
        elif isinstance(embedding_data, torch.Tensor):
            embeddings = embedding_data
        else:
            raise ValueError(f"Unsupported embedding format in {embedding_path}: {type(embedding_data)}")
        
        # Ensure tensor format
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        print(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings
    
    def prepare_training_data_from_embeddings(self,
                                            target_embedding_path: str,
                                            surrogate_embedding_paths: List[str],
                                            independent_embedding_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for Csim using pre-saved embeddings.
        Following the paper specification:
        - For each Ft, we have multiple Fs and Fi using different architectures
        - Each node in Dv has three corresponding embeddings: ht (target), hs (surrogate), hi (independent)
        - Generate distance vector between ht and hs (positive/similar)
        - Generate distance vector between ht and hi (negative/not similar)
        - Distance vector is element-wise squared distance between embeddings
        
        Args:
            target_embedding_path: Path to target model embeddings (Ft)
            surrogate_embedding_paths: Paths to surrogate model embeddings (Fs)
            independent_embedding_paths: Paths to independent model embeddings (Fi)
            
        Returns:
            Tuple of (X_train, y_train) where X_train is distance vectors and y_train is labels
        """
        print("Preparing training data for Csim from saved embeddings...")
        print(f"Target: {target_embedding_path}")
        print(f"Surrogates: {len(surrogate_embedding_paths)} models")
        print(f"Independent: {len(independent_embedding_paths)} models")
        
        # Load target embeddings (ht)
        target_embeddings = self._load_embedding_from_file(target_embedding_path)
        target_emb_np = target_embeddings.detach().cpu().numpy()
        
        distance_vectors = []
        labels = []
        
        # Positive samples: (ht, hs) pairs from target-surrogate
        print(f"\nProcessing surrogate models for positive samples...")
        for i, surrogate_path in enumerate(surrogate_embedding_paths):
            surrogate_embeddings = self._load_embedding_from_file(surrogate_path)
            surrogate_emb_np = surrogate_embeddings.detach().cpu().numpy()
            
            # Ensure same shape
            if target_emb_np.shape != surrogate_emb_np.shape:
                print(f"Warning: Shape mismatch - Target: {target_emb_np.shape}, Surrogate: {surrogate_emb_np.shape}")
                min_nodes = min(target_emb_np.shape[0], surrogate_emb_np.shape[0])
                target_emb_np_crop = target_emb_np[:min_nodes]
                surrogate_emb_np = surrogate_emb_np[:min_nodes]
            else:
                target_emb_np_crop = target_emb_np
            
            # Compute distance vectors for each node (element-wise squared distance)
            dist_vectors = self._compute_distance_vector(target_emb_np_crop, surrogate_emb_np)
            
            distance_vectors.extend(dist_vectors)
            labels.extend([1] * len(dist_vectors))  # 1 = similar (positive)
            
            print(f"  Surrogate {i+1}: {len(dist_vectors)} positive samples")
        
        # Negative samples: (ht, hi) pairs from target-independent
        print(f"\nProcessing independent models for negative samples...")
        for i, independent_path in enumerate(independent_embedding_paths):
            independent_embeddings = self._load_embedding_from_file(independent_path)
            independent_emb_np = independent_embeddings.detach().cpu().numpy()
            
            # Ensure same shape
            if target_emb_np.shape != independent_emb_np.shape:
                print(f"Warning: Shape mismatch - Target: {target_emb_np.shape}, Independent: {independent_emb_np.shape}")
                min_nodes = min(target_emb_np.shape[0], independent_emb_np.shape[0])
                target_emb_np_crop = target_emb_np[:min_nodes]
                independent_emb_np = independent_emb_np[:min_nodes]
            else:
                target_emb_np_crop = target_emb_np
            
            # Compute distance vectors for each node (element-wise squared distance)
            dist_vectors = self._compute_distance_vector(target_emb_np_crop, independent_emb_np)
            
            distance_vectors.extend(dist_vectors)
            labels.extend([0] * len(dist_vectors))  # 0 = not similar (negative)
            
            print(f"  Independent {i+1}: {len(dist_vectors)} negative samples")
        
        X_train = np.array(distance_vectors)
        y_train = np.array(labels)
        
        print(f"\nTraining data prepared:")
        print(f"  Total samples: {len(X_train)}")
        print(f"  Positive samples (surrogate): {np.sum(y_train)}")
        print(f"  Negative samples (independent): {len(y_train) - np.sum(y_train)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        return X_train, y_train

    def train_from_embeddings(self,
                            target_embedding_path: str,
                            surrogate_embedding_paths: List[str],
                            independent_embedding_paths: List[str],
                            use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train the Csim similarity classifier using pre-saved embeddings.
        
        Args:
            target_embedding_path: Path to target model embeddings
            surrogate_embedding_paths: Paths to surrogate model embeddings
            independent_embedding_paths: Paths to independent model embeddings
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        print(f"Training Csim for target model: {self.target_model_name}")
        
        # Prepare training data from saved embeddings
        X_train, y_train = self.prepare_training_data_from_embeddings(
            target_embedding_path, surrogate_embedding_paths, independent_embedding_paths
        )
        
        if use_grid_search:
            print("Performing grid search for hyperparameter tuning...")
            
            # Grid search parameters
            param_grid = {
                'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 128)],
                'activation': ['tanh', 'relu'],
                'random_state': [self.random_state],
                'max_iter': [1000]
            }
            
            # Create base classifier
            base_classifier = MLPClassifier()
            
            # Grid search with 10-fold cross validation
            grid_search = GridSearchCV(
                base_classifier, 
                param_grid, 
                cv=10, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.classifier = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"Best parameters: {best_params}")
            print(f"Best cross-validation score: {best_score:.4f}")
            
        else:
            # Use default parameters
            print("Training with default parameters...")
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(128,),
                activation='relu',
                random_state=self.random_state,
                max_iter=1000
            )
            self.classifier.fit(X_train, y_train)
            best_params = "default"
            best_score = 0.0
        
        # Evaluate on training data
        train_predictions = self.classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_train, train_predictions, 
                                  target_names=['Independent', 'Surrogate']))
        
        self.is_trained = True
        
        results = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'train_accuracy': train_accuracy,
            'num_samples': len(X_train),
            'num_positive': np.sum(y_train),
            'num_negative': len(y_train) - np.sum(y_train)
        }
        
        return results
    
    def verify_from_embeddings(self,
                              target_embedding_path: str,
                              suspect_embedding_path: str,
                              threshold: float = 0.5) -> Dict[str, Any]:
        """
        Verify if a suspect model is a surrogate or independent using pre-saved embeddings.
        
        Args:
            target_embedding_path: Path to target model embeddings
            suspect_embedding_path: Path to suspect model embeddings
            threshold: Similarity threshold 
            
        Returns:
            Dictionary with verification results
        """
        if not self.is_trained:
            raise ValueError("Csim must be trained before verification")
        
        print(f"Verifying suspect model against target: {self.target_model_name}")
        print(f"Target embeddings: {target_embedding_path}")
        print(f"Suspect embeddings: {suspect_embedding_path}")
        
        # Load embeddings
        target_embeddings = self._load_embedding_from_file(target_embedding_path)
        suspect_embeddings = self._load_embedding_from_file(suspect_embedding_path)
        
        # Ensure same shape
        target_emb_np = target_embeddings.detach().cpu().numpy()
        suspect_emb_np = suspect_embeddings.detach().cpu().numpy()
        
        if target_emb_np.shape != suspect_emb_np.shape:
            print(f"Warning: Shape mismatch - Target: {target_emb_np.shape}, Suspect: {suspect_emb_np.shape}")
            min_nodes = min(target_emb_np.shape[0], suspect_emb_np.shape[0])
            target_emb_np = target_emb_np[:min_nodes]
            suspect_emb_np = suspect_emb_np[:min_nodes]
        
        # Compute distance vectors
        distance_vectors = self._compute_distance_vector(target_emb_np, suspect_emb_np)
        
        # Classify each pair
        similarities = self.classifier.predict(distance_vectors)
        similarity_probs = self.classifier.predict_proba(distance_vectors)[:, 1]  # Probability of being similar
        
        # Calculate statistics
        num_similar = np.sum(similarities)
        total_pairs = len(similarities)
        similarity_percentage = num_similar / total_pairs
        mean_similarity_prob = np.mean(similarity_probs)
        
        # Make decision
        is_surrogate = similarity_percentage > threshold
        
        print(f"Verification Results:")
        print(f"  Total embedding pairs: {total_pairs}")
        print(f"  Similar pairs: {num_similar}")
        print(f"  Similarity percentage: {similarity_percentage:.1%}")
        print(f"  Mean similarity probability: {mean_similarity_prob:.4f}")
        print(f"  Decision: {'SURROGATE' if is_surrogate else 'INDEPENDENT'}")
        
        results = {
            'is_surrogate': is_surrogate,
            'similarity_percentage': similarity_percentage,
            'mean_similarity_prob': mean_similarity_prob,
            'num_similar': num_similar,
            'total_pairs': total_pairs,
            'threshold': threshold,
            'decision': 'surrogate' if is_surrogate else 'independent'
        }
        
        return results
    
    def save(self, save_path: str):
        """
        Save the trained Csim model.
        
        Args:
            save_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'target_model_name': self.target_model_name,
            'device': self.device,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, save_path)
        print(f"Csim model saved to: {save_path}")
    
    def load(self, load_path: str):
        """
        Load a trained Csim model.
        
        Args:
            load_path: Path to load the model from
        """
        model_data = joblib.load(load_path)
        
        self.classifier = model_data['classifier']
        self.target_model_name = model_data['target_model_name']
        self.device = model_data['device']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"Csim model loaded from: {load_path}")


class CsimManager:
    """
    Manager for handling multiple Csim models for different target models using saved embeddings.
    """
    
    def __init__(self, base_save_dir: str = "models/csim"):
        """
        Initialize Csim manager.
        
        Args:
            base_save_dir: Base directory to save Csim models
        """
        self.base_save_dir = base_save_dir
        self.csim_models = {}
    
    def train_csim_from_embeddings(self,
                                  target_model_name: str,
                                  target_embedding_path: str,
                                  surrogate_embedding_paths: List[str],
                                  independent_embedding_paths: List[str],
                                  device: str = 'cuda',
                                  use_grid_search: bool = True) -> SimilarityModel:
        """
        Args:
            target_model_name: Name/ID of target model
            target_embedding_path: Path to target model embeddings
            surrogate_embedding_paths: Paths to surrogate model embeddings
            independent_embedding_paths: Paths to independent model embeddings 
            device: Device to use
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Trained Csim model
        """
        print(f"Training Csim from embeddings for target model: {target_model_name}")
        print(f"Target embedding: {target_embedding_path}")
        print(f"Surrogate embeddings: {len(surrogate_embedding_paths)} files")
        print(f"Independent embeddings: {len(independent_embedding_paths)} files")
        
        csim = SimilarityModel(target_model_name, device=device)
        results = csim.train_from_embeddings(
            target_embedding_path, 
            surrogate_embedding_paths, 
            independent_embedding_paths,
            use_grid_search=use_grid_search
        )
        
        # Save the trained model
        save_path = os.path.join(self.base_save_dir, f"csim_{target_model_name}.pkl")
        csim.save(save_path)
        
        # Store in manager
        self.csim_models[target_model_name] = csim
        
        print(f"✅ Csim training completed for {target_model_name}")
        print(f"Training results: {results}")
        
        return csim
    
    def verify_from_embeddings(self,
                              target_model_name: str,
                              target_embedding_path: str,
                              suspect_embedding_path: str,
                              threshold: float = 0.5) -> Dict[str, Any]:
        """
        Verify a suspect model against a target using pre-saved embeddings.
        
        Args:
            target_model_name: Name of target model
            target_embedding_path: Path to target model embeddings
            suspect_embedding_path: Path to suspect model embeddings
            threshold: Similarity threshold
            
        Returns:
            Verification results
        """
        # Load Csim if not already loaded
        if target_model_name not in self.csim_models:
            csim_path = os.path.join(self.base_save_dir, f"csim_{target_model_name}.pkl")
            if os.path.exists(csim_path):
                csim = SimilarityModel(target_model_name)
                csim.load(csim_path)
                self.csim_models[target_model_name] = csim
            else:
                raise ValueError(f"No trained Csim found for target model: {target_model_name}")
        
        csim = self.csim_models[target_model_name]
        return csim.verify_from_embeddings(target_embedding_path, suspect_embedding_path, threshold) 