# GrOVe: Graph Neural Network Ownership Verification via Embeddings

This repository contains the code for reproducing experiments from the paper "GrOVe: Ownership Verification of Graph Neural Networks using Embeddings"

## Installation 

### Using Docker Compose

1. Install Docker Desktop for Windows from [Docker's official website](https://www.docker.com/products/docker-desktop/)

2. Clone this repository:
```bash
git clone https://github.com/aasmanbashyal/GrOVe.git
cd GrOVe
```

3. Build and run using Docker Compose:
```bash
docker-compose up --build
```

## Dataset

The datasets used in this project can be downloaded from GitHub repository:

1. Download the datasets:
```bash
https://github.com/xinleihe/GNNStealing/tree/master/code/datasets
```

2. Place the extracted datasets in the `data/raw/` directory.



## Embedding Distribution Plots

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