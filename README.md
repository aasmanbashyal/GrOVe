# GrOVe: Graph Neural Network Ownership Verification via Embeddings

This repository contains the code for reproducing experiments from the paper "GrOVe: Ownership Verification of Graph Neural Networks using Embeddings"

## Quick Start

### **Software Prerequisites**
- Docker and docker-compose Desktop 
- **GPU**: NVIDIA GPU with CUDA > 12.6+ 
- Git for repository cloning
- NVIDIA drivers (for GPU acceleration)

### Tested on Docker

### **Step 2: Clone & Setup**
```bash
# Clone repository
git clone https://github.com/aasmanbashyal/GrOVe.git
cd GrOVe

# Download datasets (place in data/raw/)
# From: https://github.com/xinleihe/GNNStealing/tree/master/code/datasets
mkdir -p data/raw
# Download: acm.npz, amazon.npz, citeseer.npz, coauthor.npz, dblp.npz, pubmed.npz

docker-compose up --build
```
**What happens**: Runs preprocessing, target and independent training, basic and
advanced attacks, CSim training/verification, comprehensive evaluation, and result
organization. The default is a full run across 6 datasets and 3 target models.
Missing raw datasets stop the run before training starts. Any failed command
stops subsequent stages and returns a nonzero exit status.

The runner resolves the repository root from its own location, so it also works
when invoked outside the repository directory. It runs Python modules from that
root, making the `grove` package available outside Docker.

### Training validation and checkpoints

Target, independent, and traditional surrogate training hold out 20% of their
selected training graph for early stopping. The split uses a separate fixed
validation seed, so target and independent runs sharing the same graph use the
same partition despite different initialization seeds. `test.pt` is evaluated
only after model selection; `validation.pt` or `verification.pt` remains the
ownership-verification graph used to generate embeddings.

Best-model weights are copied when validation improves and restored before
final evaluation and embedding generation. Saved weights, final metrics, and
saved verification embeddings therefore refer to the same selected checkpoint.
`final_test_acc` reports the held-out test result separately from validation.
Existing results should be regenerated to use this validation protocol.

### Regression tests without datasets or a GPU

```bash
uv venv --python 3.12
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-test.txt
uv run --no-project python -m pytest tests -q
```

Training tests use tiny synthetic graphs. Pipeline tests execute Bash with stub
Python commands to check phase order, missing inputs, and failure propagation;
they do not run the full experiment suite. On Windows they use Git Bash when
available. The ownership-verification CLI now returns exit code 0 for either
successful classification and nonzero for execution errors.


---

## Project Structure & File Contents

### **Core Implementation (`grove/`)**
- **`models/gnn.py`**: GNN architectures (GAT, GIN, GraphSAGE) with forward methods returning (embeddings, predictions)
- **`attacks/model_stealing.py`**: Core attack framework with SurrogateModelTrainer, DistributionShiftTrainer, Type I/II attacks
- **`attacks/evaluation.py`**: ModelEvaluator with fidelity, classification metrics, embedding similarity computation
- **`attacks/advanced_models.py`**: Sophisticated attacks (adversarial training, knowledge distillation, ensemble attacks)
- **`attacks/graph_reconstruction.py`**: Graph structure reconstruction (IDGL, KNN, random, adaptive) for Type II attacks
- **`verification/similarity_model.py`**: CSim model and OwnershipVerifier for ownership verification
- **`utils/data_loader.py`**: NPZ dataset loading, PyTorch Geometric conversion, train/val/test split creation
- **`utils/data_processing.py`**: Feature normalization, graph augmentation, data format conversion

### **Execution Scripts (`scripts/`)**
- **`run_all_experiments.sh`**: Main experiment runner (executed by docker-compose) with 5-phase pipeline
- **`all_model_stealing_evaluation.py`**: Comprehensive evaluation testing all 9 architecture combinations
- **`train_model.py`**: Target model training with configurable architectures and roles
- **`train_stealing_surrogate.py`**: Surrogate model training with advanced attack options
- **`preprocess_data.py`**: Data preprocessing pipeline with overlapped/non-overlapped splits
- **`visualize_embeddings.py`**: t-SNE/PCA visualization with multiple perplexity values
- **`organize_results.py`**: Results organization preserving existing visualizations

### **Data & Models**
- **`data/raw/`**: Original datasets (acm.npz, amazon.npz, citeseer.npz, coauthor.npz, dblp.npz, pubmed.npz)
- **`data/processed/`**: Preprocessed data splits (query_data, val_data, test_data, verification_data)
- **`saved_models/`**: Trained target models with consistent naming convention
- **`embeddings/`**: Model embeddings for target, independent, and surrogate models

### **Results & Outputs**
- **`results/basic_attacks/`**: Type I/II attack results with CSV metrics
- **`results/advanced_attacks/`**: Sophisticated attack results (fine-tuning, double extraction, pruning)
- **`results/comprehensive_evaluation/`**: Complete evaluation reports with analysis text files
- **`results/visualizations/`**: t-SNE plots with unique timestamped directories
- **`results/tables/`**: Summary CSV files aggregating all results

---

## Execution Flow (`docker-compose up`)

### **Phase 1: Basic Experimental Pipeline**
1. **Data Preprocessing**: Load NPZ files → Create non-overlapped splits → PyTorch Geometric format
2. **Target Training**: Train GAT/GIN/SAGE models using train split with seed 42
3. **Independent Training**: Train one independently initialized baseline per target architecture with seed 789
4. **Basic Attacks**: 
   - Type I (original structure): Query target → Train surrogate on original graph
   - Type II (IDGL structure): Query target → Reconstruct graph → Train surrogate
5. **Visualization**: Generate t-SNE plots comparing target/independent/surrogate embeddings

### **Phase 2: Advanced Attacks**
- **Fine-tuning attacks**: Improved surrogate training with pre-trained initialization
- **Double extraction**: Two-stage attack strategy for enhanced performance
- **Distribution shift**: Attacks with noise/transformations during training
- **Pruning attacks**: Test robustness against pruned targets (0.3, 0.5, 0.7 ratios)

### **Phase 3: CSim Training & Verification**
- **CSim model training**: Learn similarity metrics using grid search optimization
- **Ownership verification**: Test verification on surrogate models with threshold 0.5

### **Phase 4: Comprehensive Evaluation**
- **Cross-architecture testing**: All 9 combinations (GAT/GIN/SAGE target × GAT/GIN/SAGE surrogate)
- **Recovery methods**: Test both embedding-based and prediction-based attacks
- **Metric computation**: Fidelity, accuracy, embedding similarity, KL divergence
- **Report generation**: CSV files and comprehensive analysis text reports

### **Phase 5: Results Organization**
- **File organization**: Move results to structured directories preserving existing files
- **Summary generation**: Create aggregated CSV tables and overall summaries

---

## Results Structure & Contents

### **Key Result Files**
| File Type | Location | Contents |
|-----------|----------|----------|
| **Main Results** | `results/comprehensive_evaluation/{dataset}/` | Model_Stealing_Evaluation_comprehensive_evaluation_{dataset}.csv |
| **Analysis Reports** | `results/comprehensive_evaluation/{dataset}/` | comprehensive_analysis_{timestamp}.txt |
| **Basic Attacks** | `results/basic_attacks/` | Type I/II attack results with detailed metrics |
| **Advanced Attacks** | `results/advanced_attacks/` | Fine-tuning, double extraction, pruning results |
| **Visualizations** | `results/visualizations/` | t-SNE embedding plots with unique directories |
| **Summary Tables** | `results/tables/` | basic_attacks_summary.csv, advanced_attacks_summary.csv, overall_results_summary.csv |

### **CSV Columns Explained**
| Column | Description | Interpretation |
|--------|-------------|----------------|
| **`fidelity`** | Agreement between target/surrogate predictions | **0.95-1.0** = Near-perfect, **0.85-0.95** = Highly successful, **0.70-0.85** = Moderate, **<0.70** = Poor |
| **`target_accuracy`** | Target model test accuracy | Reference performance baseline |
| **`surrogate_accuracy`** | Surrogate model test accuracy | Attack capability indicator |
| **`accuracy_gap`** | \|target_accuracy - surrogate_accuracy\| | **<0.05** = Excellent, **0.05-0.10** = Good, **0.10-0.20** = Moderate, **>0.20** = Poor |
| **`embedding_cosine_similarity`** | Embedding similarity (when dimensions match) | **>0.8** = High similarity, **0.5-0.8** = Moderate, **<0.5** = Low |
| **`cross_architecture`** | Different target/surrogate architectures | Tests transferability across model types |
| **`kl_divergence`** | KL divergence between probability distributions | Lower values indicate better mimicking |
| **`embedding_l2_distance`** | L2 distance between embeddings | Lower values indicate closer representations |


## Installation & Configuration

### **Prerequisites**
- Docker Desktop (recommended) 
- NVIDIA GPU with CUDA 12.6+

### **Quick Docker Setup**
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop/
# Clone and run
git clone https://github.com/aasmanbashyal/GrOVe.git
cd GrOVe
docker-compose up --build
```

### **Custom Configuration**
```bash
# Native run using the installed Python environment
DATASETS="citeseer acm" MODELS="gat gin" EPOCHS=100 DEVICE=cuda \
    bash scripts/run_all_experiments.sh

# Reuse prepared splits and trained target checkpoints, skipping phases 1-3
DATASETS="citeseer" MODELS="gat" DEVICE=cpu \
    bash scripts/run_all_experiments.sh evaluate
```

Environment options are `DATASETS`, `MODELS`, `DEVICE`, `EPOCHS`, `SPLIT_TYPE`,
`VALIDATION_FRACTION`, `VALIDATION_SEED`, and `PYTHON_BIN`. Lists are
space-separated. Defaults are the six datasets above, `gat gin sage`, `cuda`,
200, `non-overlapped`, 0.2, 42, and `python3`, respectively. `MODELS` controls
the target/baseline models prepared by the full run. Comprehensive evaluation
discovers saved target checkpoints and tests all three surrogate architectures.
Use a clean model-output directory when limiting a new experiment's targets.

The `evaluate` mode checks the processed data and each requested target
checkpoint before running. It never silently skips a missing prerequisite.
CSim training uses only actual independent embeddings; target embeddings are
never substituted as independent negative examples.

### **Manual Evaluation**
```bash
python scripts/all_model_stealing_evaluation.py \
    --dataset citeseer \
    --models-dir saved_models/non-overlapped \
    --output-dir results/custom \
    --epochs 100 \
    --include-advanced \
    --device cuda
```

---

##  Embedding Distribution Plots

Below are t-SNE visualizations of embeddings for different models and datasets. The captions describe the model, dataset, and perplexity value used for each plot.

### Basic Model Comparisons - Non-overlapped Split

<table>
<tr>
<td align="center" width="33%">
<img src="visualizations/non-overlapped/gin_citeseer/citeseer_tsne_per_30.png" alt="GIN Citeseer" width="300"/>
<br><b>GIN - Citeseer Dataset</b>
</td>
<td align="center" width="33%">
<img src="visualizations/non-overlapped/gat_citeseer/citeseer_tsne_per_30.png" alt="GAT Citeseer" width="300"/>
<br><b>GAT - Citeseer Dataset</b>
</td>
<td align="center" width="33%">
<img src="visualizations/non-overlapped/sage_citeseer/citeseer_tsne_per_30.png" alt="GraphSAGE Citeseer" width="300"/>
<br><b>GraphSAGE - Citeseer Dataset</b>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="visualizations/non-overlapped/gat_acm/acm_tsne_per_30.png" alt="GAT ACM" width="300"/>
<br><b>GAT - ACM Dataset</b>
</td>
<td align="center" width="33%">
<img src="visualizations/non-overlapped/gin_acm/acm_tsne_per_30.png" alt="GIN ACM" width="300"/>
<br><b>GIN - ACM Dataset</b>
</td>
<td align="center" width="33%">
<img src="visualizations/non-overlapped/sage_acm/acm_tsne_per_30.png" alt="GraphSAGE ACM" width="300"/>
<br><b>GraphSAGE - ACM Dataset</b>
</td>
</tr>
</table>

### Advanced Attack Techniques - GAT on Citeseer

<table>
<tr>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_fine_tuning/citeseer_tsne_per_30.png" alt="Fine-tuned GAT" width="300"/>
<br><b>Fine-tuned GAT</b>
</td>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_double_extraction/citeseer_tsne_per_30.png" alt="Double Extraction Type 1" width="300"/>
<br><b>Double Extraction Type 1</b>
</td>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_double_extraction_type2/citeseer_tsne_per_30.png" alt="Double Extraction Type 2" width="300"/>
<br><b>Double Extraction Type 2</b>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_distribution_shift/citeseer_tsne_per_30.png" alt="Distribution Shift" width="300"/>
<br><b>Distribution Shift</b>
</td>
<td align="center" width="33%">
</td>
<td align="center" width="33%">
</td>
</tr>
</table>

### Pruning Analysis - GAT on Citeseer

<table>
<tr>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_pruning_01/citeseer_tsne_per_30.png" alt="Pruning 0.1" width="300"/>
<br><b>Pruning Ratio: 0.1</b>
</td>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_pruning_02/citeseer_tsne_per_30.png" alt="Pruning 0.2" width="300"/>
<br><b>Pruning Ratio: 0.2</b>
</td>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_pruning_03/citeseer_tsne_per_30.png" alt="Pruning 0.3" width="300"/>
<br><b>Pruning Ratio: 0.3</b>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_pruning_04/citeseer_tsne_per_30.png" alt="Pruning 0.4" width="300"/>
<br><b>Pruning Ratio: 0.4</b>
</td>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_pruning_05/citeseer_tsne_per_30.png" alt="Pruning 0.5" width="300"/>
<br><b>Pruning Ratio: 0.5</b>
</td>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_pruning_06/citeseer_tsne_per_30.png" alt="Pruning 0.6" width="300"/>
<br><b>Pruning Ratio: 0.6</b>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="test_advance/new_visualizations/non-overlapped/gat_citeseer_pruning_07/citeseer_tsne_per_30.png" alt="Pruning 0.7" width="300"/>
<br><b>Pruning Ratio: 0.7</b>
</td>
<td align="center" width="33%">
</td>
<td align="center" width="33%">
</td>
</tr>
</table>

### **Visualization Interpretation**
- **Close clusters**: Models learn similar representations
- **Overlapping points**: High functional similarity between target and surrogate
- **Distinct clusters**: Models learn different features (indicates attack failure)
- **Color coding**: Target (blue circles), Independent (green squares), Surrogate (red triangles)


## Citation

```bibtex
@article{grove2023,
  title={GrOVe: Ownership Verification of Graph Neural Networks using Embeddings},
  author={Asim Waheed, Vasisht Duddu, and N. Asokan},
  journal={arXiv preprint},
  year={2023}
}
```

## References

```
https://github.com/
ssg-research/GrOVe
```

```
https://github.com/
xinleihe/GNNStealing
```
