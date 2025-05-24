import os
import logging
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from collections import Counter

# Configure logger (minimal)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def _normalize_features_if_needed(features_list: list, dataset_name: str):
    """Helper to normalize feature lengths if they are inconsistent."""
    feature_lengths = [len(feat_vec) for feat_vec in features_list]
    if len(set(feature_lengths)) == 1:
        return torch.FloatTensor(np.array(features_list, dtype=np.float32))

    logger.warning(f"Dataset '{dataset_name}': Node features have varying dimensions. Normalizing...")
    most_common_len = Counter(feature_lengths).most_common(1)[0][0]
    logger.info(f"Normalizing all features to length {most_common_len}.")

    normalized_features_array = np.zeros((len(features_list), most_common_len), dtype=np.float32)
    for i, feat_vec in enumerate(features_list):
        current_len = len(feat_vec)
        if current_len == most_common_len:
            normalized_features_array[i, :] = feat_vec
        elif current_len < most_common_len:
            normalized_features_array[i, :current_len] = feat_vec
        else: # current_len > most_common_len
            normalized_features_array[i, :] = feat_vec[:most_common_len]
    return torch.FloatTensor(normalized_features_array)

def _parse_npz_to_pyg_data(npz_file_path: str, dataset_name: str):
    """
    Parser for NPZ contents to PyTorch Geometric Data object.
    Expects keys: 'node_attr', 'node_label', 'adj_matrix'.
    """
    if not os.path.exists(npz_file_path):
        logger.error(f"NPZ file for '{dataset_name}' not found: {npz_file_path}")
        raise FileNotFoundError(f"NPZ file for '{dataset_name}' not found: {npz_file_path}")

    logger.info(f"Loading '{dataset_name}' from NPZ: {npz_file_path}")
    try:
        npz_data = np.load(npz_file_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load NPZ file '{npz_file_path}': {e}")
        raise

    # 1. Node Features ('node_attr')
    if 'node_attr' not in npz_data:
        raise ValueError(f"'node_attr' key missing in {npz_file_path}")
    node_attr_raw = npz_data['node_attr']
    x = None

    if sp.issparse(node_attr_raw):
        logger.info(f"Node attributes are sparse. Converting to dense array.")
        x = torch.FloatTensor(node_attr_raw.toarray())
    elif isinstance(node_attr_raw, np.ndarray):
        if node_attr_raw.dtype != np.dtype('O'): # Standard numeric array
            x = torch.FloatTensor(node_attr_raw.astype(np.float32))
        else: # Object array - handle common cases
            if node_attr_raw.size == 0:
                raise ValueError(f"'node_attr' object array in '{npz_file_path}' is empty.")
            
            # Case: array of sparse items (features for individual nodes)
            if all(m is None or sp.issparse(m) for m in node_attr_raw.flat):
                logger.debug(f"'{dataset_name}': 'node_attr' is array of sparse items.")
                processed_features = []
                for item in node_attr_raw.flat:
                    if item is None: continue
                    dense_item = item.toarray()
                    processed_features.append(dense_item.flatten())
                if not processed_features:
                     raise ValueError(f"No valid features from 'node_attr' (array of sparse items) in '{npz_file_path}'.")
                x = _normalize_features_if_needed(processed_features, dataset_name)
            # Case: array of ndarrays (e.g. list of feature vectors)
            elif all(m is None or isinstance(m, np.ndarray) for m in node_attr_raw.flat):
                logger.debug(f"'{dataset_name}': 'node_attr' is array of ndarrays.")
                processed_features = [item for item in node_attr_raw.flat if item is not None]
                if not processed_features:
                     raise ValueError(f"No valid features from 'node_attr' (array of ndarrays) in '{npz_file_path}'.")
                x = _normalize_features_if_needed(processed_features, dataset_name)
            else:
                raise ValueError(f"Unhandled object array structure for 'node_attr' in {npz_file_path}. First element type: {type(node_attr_raw.flat[0]) if node_attr_raw.size > 0 else 'N/A'}")
    else:
        raise ValueError(f"Unexpected type for 'node_attr': {type(node_attr_raw)} in {npz_file_path}")
    
    if x is None:
        raise ValueError(f"Failed to process 'node_attr' in {npz_file_path}")

    # 2. Node Labels ('node_label')
    if 'node_label' not in npz_data:
        raise ValueError(f"'node_label' key missing in {npz_file_path}")
    node_label_raw = npz_data['node_label']
    
    if isinstance(node_label_raw, np.ndarray):
        logger.info(f"Node labels are numpy array. Converting to long tensor.")
        if node_label_raw.dtype == np.dtype('O'):
            logger.info(f"Node labels are object array. Converting to long tensor.")
            y = torch.LongTensor([int(lbl) for lbl in node_label_raw.flat])
        else:
            y = torch.LongTensor(node_label_raw.astype(np.int64))
    else:
        raise ValueError(f"Unexpected type for 'node_label': {type(node_label_raw)} in {npz_file_path}")

    # 3. Adjacency Matrix ('adj_matrix') to Edge Index
    if 'adj_matrix' not in npz_data:
        raise ValueError(f"'adj_matrix' key missing in {npz_file_path}")
    adj_raw = npz_data['adj_matrix']
    logger.info(f"Adjacency matrix is {adj_raw}")
    adj_processed = adj_raw
    # Handle common case of sparse matrix wrapped in 0-d or 1-element object array
    if isinstance(adj_raw, np.ndarray) and adj_raw.dtype == np.dtype('O'):
        if adj_raw.ndim == 0 and sp.issparse(adj_raw.item()):
            adj_processed = adj_raw.item()
        elif adj_raw.size == 1 and sp.issparse(adj_raw.flat[0]):
            adj_processed = adj_raw.flat[0]

    if sp.issparse(adj_processed):
        adj_coo = adj_processed.tocoo()
        edge_index = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
    elif isinstance(adj_processed, np.ndarray) and adj_processed.ndim == 2: # Dense adj
        rows, cols = adj_processed.nonzero()
        edge_index = torch.LongTensor(np.vstack((rows, cols)))
    else:
        raise ValueError(f"Unexpected type/structure for 'adj_matrix': {type(adj_processed)} in {npz_file_path}")

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    if data.y is not None:
        data.num_classes = len(torch.unique(data.y))
    else:
        data.num_classes = None
        logger.warning(f"'{dataset_name}': Node labels missing; 'num_classes' is None.")
    
    logger.info(f"Loaded '{dataset_name}': {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} feats, {data.num_classes or 'N/A'} classes.")
    return data

def load_npz_graph_dataset(dataset_name: str, data_directory: str):
    """
    Loads a graph dataset from `dataset_name.npz` in `data_directory`
    """
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be a non-empty string.")
    if not isinstance(data_directory, str) or not os.path.isdir(data_directory):
        raise ValueError(f"data_directory '{data_directory}' must be a valid directory.")

    npz_filename = f"{dataset_name}.npz"
    npz_file_path = os.path.join(data_directory, npz_filename)

    return _parse_npz_to_pyg_data(npz_file_path, dataset_name)

if __name__ == "__main__":
    load_npz_graph_dataset("citeseer", "data/raw")