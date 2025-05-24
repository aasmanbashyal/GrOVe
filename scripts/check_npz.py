import numpy as np
import scipy.sparse as sp

def inspect_npz(file_path):
    print(f"\nInspecting NPZ file: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    
    print("\nAvailable keys:", data.files)
    
    for key in data.files:
        item = data[key]
        print(f"\n{key}:")
        print(f"Type: {type(item)}")
        if isinstance(item, np.ndarray):
            print(f"Shape: {item.shape}")
            print(f"dtype: {item.dtype}")
            if item.dtype == np.dtype('O'):
                print("First element type:", type(item.flat[0]))
                if sp.issparse(item.flat[0]):
                    print("First element shape:", item.flat[0].shape)
        elif sp.issparse(item):
            print(f"Shape: {item.shape}")
            print(f"Format: {item.format}")
            print(f"Number of non-zero elements: {item.nnz}")

if __name__ == "__main__":
    inspect_npz("data/raw/citeseer.npz") 