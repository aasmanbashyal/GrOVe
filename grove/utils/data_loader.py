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
        logger.info(f"NPZ file keys: {npz_data.files}")
    except Exception as e:
        logger.error(f"Failed to load NPZ file '{npz_file_path}': {e}")
        raise

    # 1. Node Features ('node_attr')
    if 'node_attr' not in npz_data:
        raise ValueError(f"'node_attr' key missing in {npz_file_path}")
    node_attr_raw = npz_data['node_attr']
    logger.info(f"Node attributes shape/type: {node_attr_raw.shape if hasattr(node_attr_raw, 'shape') else type(node_attr_raw)}")
    
    # Handle node attributes
    if isinstance(node_attr_raw, np.ndarray) and node_attr_raw.dtype == np.dtype('O'):
        if node_attr_raw.size == 1 and sp.issparse(node_attr_raw.flat[0]):
            x = torch.FloatTensor(node_attr_raw.flat[0].toarray())
        else:
            raise ValueError(f"Unexpected node_attr format in {npz_file_path}")
    elif sp.issparse(node_attr_raw):
        x = torch.FloatTensor(node_attr_raw.toarray())
    else:
        raise ValueError(f"Unexpected type for 'node_attr': {type(node_attr_raw)} in {npz_file_path}")

    # 2. Node Labels ('node_label')
    if 'node_label' not in npz_data:
        raise ValueError(f"'node_label' key missing in {npz_file_path}")
    node_label_raw = npz_data['node_label']
    logger.info(f"Node labels shape/type: {node_label_raw.shape if hasattr(node_label_raw, 'shape') else type(node_label_raw)}")
    
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
    logger.info(f"Adjacency matrix shape/type: {adj_raw.shape if hasattr(adj_raw, 'shape') else type(adj_raw)}")
    
    # Handle adjacency matrix
    if isinstance(adj_raw, np.ndarray) and adj_raw.dtype == np.dtype('O'):
        if adj_raw.size == 1 and sp.issparse(adj_raw.flat[0]):
            adj_processed = adj_raw.flat[0]
        else:
            raise ValueError(f"Unexpected adj_matrix format in {npz_file_path}")
    elif sp.issparse(adj_raw):
        adj_processed = adj_raw
    else:
        raise ValueError(f"Unexpected type for 'adj_matrix': {type(adj_raw)} in {npz_file_path}")

    # Convert to edge index
    adj_coo = adj_processed.tocoo()
    edge_index = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))

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