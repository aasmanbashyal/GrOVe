from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from scripts import train_model as training


def graph(node_count=8, labels=None, one_hot=False, marker=0):
    if labels is None:
        labels = torch.arange(node_count) % 2
    x = torch.stack((torch.full((node_count,), float(marker)), torch.arange(node_count).float()), dim=1)
    edges = torch.cartesian_prod(torch.arange(node_count), torch.arange(node_count)).t().contiguous()
    if one_hot:
        labels = torch.nn.functional.one_hot(labels, num_classes=3).float()
    return Data(x=x, edge_index=edges, y=labels)


def write_graphs(root, test_labels=None, one_hot=False, split_type='non-overlapped'):
    directory = root / 'data' / 'processed' / split_type / 'tiny'
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ('target_train', 'independent_train', 'query_train'):
        torch.save(graph(one_hot=one_hot), directory / f'{filename}.pt')
    verification_name = 'verification' if split_type == 'overlapped' else 'validation'
    torch.save(graph(marker=7, one_hot=one_hot), directory / f'{verification_name}.pt')
    torch.save(graph(labels=test_labels, marker=99, one_hot=one_hot), directory / 'test.pt')
    return directory


def test_training_validation_split_is_disjoint_deterministic_and_remaps_edges():
    source = graph()
    train, validation = training.split_training_graph(source, 0.25, seed=12)
    train_ids = set(train.x[:, 1].long().tolist())
    validation_ids = set(validation.x[:, 1].long().tolist())
    assert not train_ids & validation_ids
    assert train_ids | validation_ids == set(range(source.num_nodes))
    assert (train.num_nodes, validation.num_nodes) == (6, 2)
    for part in (train, validation):
        assert part.edge_index.shape == (2, part.num_nodes ** 2)
        assert part.edge_index.min() == 0
        assert part.edge_index.max() == part.num_nodes - 1
        assert torch.equal(part.y, source.y[part.x[:, 1].long()])
    torch.manual_seed(999)
    repeated = training.split_training_graph(source, 0.25, seed=12)
    assert torch.equal(train.x, repeated[0].x)
    assert torch.equal(validation.x, repeated[1].x)
    assert source.num_nodes == 8


def test_validation_split_preserves_edgeless_graph_shape():
    source = graph()
    source.edge_index = torch.empty((2, 0), dtype=torch.long)
    parts = training.split_training_graph(source)
    assert all(part.edge_index.shape == (2, 0) for part in parts)


@pytest.mark.parametrize('fraction', [0, 1, -0.1, float('nan')])
def test_invalid_validation_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match='validation_fraction'):
        training.split_training_graph(graph(), fraction)


def test_training_graph_must_have_nodes_for_both_partitions():
    with pytest.raises(ValueError, match='at least two nodes'):
        training.split_training_graph(graph(node_count=1))


@pytest.mark.parametrize('labels', [torch.tensor(0), torch.tensor([0.5] * 8),
                                   torch.zeros((8, 2)), torch.full((8, 2), float('nan'))])
def test_invalid_labels_fail_explicitly(labels):
    source = graph()
    source.y = labels
    with pytest.raises(ValueError, match='labels'):
        training.class_indices(source, 'training')


def test_best_state_is_restored_and_test_data_cannot_select_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = write_graphs(tmp_path, test_labels=torch.zeros(8, dtype=torch.long))
    # A controlled optimizer makes epoch 2 best, followed by an objectively worse
    # epoch. Real tensors/autograd exercise snapshot aliasing and serialization.
    for name in ('target_train', 'independent_train', 'query_train'):
        torch.save(graph(labels=torch.zeros(8, dtype=torch.long), one_hot=True), directory / f'{name}.pt')
    instances = []
    optimizers = []

    class ControlledModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            instances.append(self)

        def forward(self, batch):
            embeddings = self.weight.expand(batch.num_nodes, 1)
            return embeddings, torch.cat((embeddings, -embeddings, torch.zeros_like(embeddings)), dim=1)

    class ControlledOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)
            self.values = iter([1.0, 2.0, -1.0, -2.0])
            self.steps = 0
            optimizers.append(self)

        def zero_grad(self):
            for parameter in self.parameters:
                parameter.grad = None

        def step(self):
            self.steps += 1
            with torch.no_grad():
                self.parameters[0].fill_(next(self.values))

    monkeypatch.setattr(training, 'GATModel', ControlledModel)
    monkeypatch.setattr(training.torch.optim, 'Adam', ControlledOptimizer)
    original_load = torch.load
    test_reads = []

    def checked_load(path, *args, **kwargs):
        if Path(path).name == 'test.pt':
            test_reads.append(path)
            assert optimizers[-1].steps == 3
            assert instances[-1].weight.item() == 2.0
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(training.torch, 'load', checked_load)
    for name, labels in [('first', torch.zeros(8, dtype=torch.long)),
                         ('second', torch.ones(8, dtype=torch.long))]:
        torch.save(graph(labels=labels, marker=99), directory / 'test.pt')
        training.train_model('gat', 'tiny', tmp_path / name, device='cpu', epochs=4,
                             early_stopping=1, hidden_dim=4, validation_fraction=0.25)

    checkpoints = [original_load(tmp_path / name / 'gat_tiny_target_non-overlapped.pt', weights_only=False)
                   for name in ('first', 'second')]
    assert len(test_reads) == 2
    for checkpoint in checkpoints:
        assert checkpoint['model_state_dict']['weight'].item() == 2.0
        assert checkpoint['training_history']['best_epoch'] == 1
        assert len(checkpoint['training_history']['val_losses']) == 3
    assert checkpoints[0]['training_history']['val_losses'] == checkpoints[1]['training_history']['val_losses']
    assert checkpoints[0]['training_history']['final_test_acc'] == 1.0
    assert checkpoints[1]['training_history']['final_test_acc'] == 0.0
    embeddings = original_load(tmp_path / 'first' / 'gat_tiny_target.pt', weights_only=False)
    for key in ('best_embeddings', 'final_embeddings'):
        assert torch.equal(embeddings[key]['verification'].x, torch.full((8, 1), 2.0))
    metrics = pd.read_csv(tmp_path / 'first' / 'gat_tiny_target_non-overlapped_metrics.csv')
    assert metrics.iloc[0]['final_val_acc'] == 1.0
    assert metrics.iloc[0]['final_test_acc'] == 1.0


@pytest.mark.parametrize('model,role,one_hot,split_type', [
    ('gat', 'target', True, 'non-overlapped'),
    ('gin', 'independent', False, 'non-overlapped'),
    ('sage', 'surrogate', True, 'overlapped'),
])
def test_real_models_train_on_tiny_graphs_with_both_label_formats(tmp_path, monkeypatch, model, role, one_hot, split_type):
    monkeypatch.chdir(tmp_path)
    directory = write_graphs(tmp_path, one_hot=one_hot, split_type=split_type)
    if one_hot:
        # Preprocessing encodes each saved graph independently, so a split may
        # have fewer one-hot columns than the complete training class schema.
        test = graph(marker=99)
        test.y = torch.nn.functional.one_hot(test.y, num_classes=2).float()
        torch.save(test, directory / 'test.pt')
    output = tmp_path / 'new-output-directory'
    training.train_model(model, 'tiny', output, device='cpu', epochs=2, hidden_dim=4,
                         dropout=0, model_role=role, split_type=split_type)
    independent_suffix = '_gat' if role == 'independent' else ''
    checkpoint = torch.load(output / f'{model}_tiny_{role}{independent_suffix}_{split_type}.pt', weights_only=False)
    assert checkpoint['model_config']['output_dim'] == (3 if one_hot else 2)
    assert checkpoint['training_history']['train_nodes'] == 7
    assert checkpoint['training_history']['validation_nodes'] == 1
    assert np.isfinite(checkpoint['training_history']['val_losses']).all()
    assert 0 <= checkpoint['training_history']['final_test_acc'] <= 1
