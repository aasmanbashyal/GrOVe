"""Exercise the shell runner without downloading data or training models."""

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def find_bash():
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if git_bash.is_file():
            return str(git_bash)
    return shutil.which("bash")


@pytest.fixture
def runner(tmp_path):
    bash = find_bash()
    if bash is None:
        pytest.skip("Bash is required for pipeline integration tests")
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    runner_path = scripts / "run_all_experiments.sh"
    runner_path.write_text((ROOT / "scripts/run_all_experiments.sh").read_text(), newline="\n")
    stub = tmp_path / "stub.py"
    stub.write_text('''
import json, os, sys
from pathlib import Path

assert sys.argv[1] == "-m"
name = sys.argv[2].rsplit(".", 1)[-1]
args = sys.argv[3:]
with open(os.environ["TEST_COMMAND_LOG"], "a") as log:
    log.write(json.dumps({"name": name, "args": args, "cwd": os.getcwd()}) + "\\n")
if os.environ.get("TEST_FAIL_STAGE") == name:
    sys.exit(7)

def value(option):
    return args[args.index(option) + 1]

def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

if name == "preprocess_data":
    for split in ("target_train", "query_train", "validation", "test"):
        touch(Path(value("--output-dir")) / (split + ".pt"))
elif name == "train_model":
    model, dataset = value("--model"), value("--dataset")
    role, split = value("--model-role"), value("--split-type")
    suffix = "_" + value("--independent-model") if role == "independent" else ""
    touch(Path(value("--output-dir")) / f"{model}_{dataset}_{role}{suffix}_{split}.pt")
    touch(Path(value("--embeddings-dir")) / f"{model}_{dataset}_{role}{suffix}.pt")
elif name == "train_stealing_surrogate":
    model, dataset, structure = value("--model"), value("--dataset"), value("--structure")
    suffix = "_" + value("--advanced-attack") if "--advanced-attack" in args else ""
    touch(Path(value("--embeddings-dir")) / f"{model}_{dataset}_surrogate_{structure}{suffix}.pt")
''')
    wrapper = tmp_path / "python-stub"
    wrapper.write_text('#!/usr/bin/env bash\nexec "$TEST_PYTHON" "$TEST_STUB" "$@"\n', newline="\n")
    wrapper.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    env = dict(os.environ, DATASETS="toy", MODELS="gat", DEVICE="cpu", EPOCHS="2",
               SPLIT_TYPE="non-overlapped", PYTHON_BIN=wrapper.as_posix(),
               TEST_PYTHON=Path(sys.executable).as_posix(), TEST_STUB=stub.as_posix(),
               TEST_COMMAND_LOG=str(log))

    def run(mode=None, **overrides):
        result = subprocess.run([bash, runner_path.as_posix(), *([mode] if mode else [])],
                                cwd=tmp_path, env={**env, **overrides},
                                capture_output=True, text=True, timeout=45)
        commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, commands

    return project, run


def raw_dataset(project, dataset="toy"):
    raw = project / "data/raw" / f"{dataset}.npz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.touch()


def prepared_dataset(project, model=True):
    processed = project / "data/processed/non-overlapped/toy"
    processed.mkdir(parents=True)
    for name in ("target_train", "query_train", "validation", "test"):
        (processed / f"{name}.pt").touch()
    if model:
        checkpoint = project / "saved_models/non-overlapped/gat_toy/gat_toy_target_non-overlapped.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()


def test_default_full_run_builds_prerequisites_before_evaluation(runner, monkeypatch):
    project, run = runner
    raw_dataset(project)
    result, commands = run()
    assert result.returncode == 0, result.stderr + result.stdout
    names = [command["name"] for command in commands]
    assert names[:3] == ["preprocess_data", "train_model", "train_model"]
    assert names.index("train_csim_from_embeddings") < names.index("all_model_stealing_evaluation")
    assert names[-2:] == ["all_model_stealing_evaluation", "organize_results"]
    assert names.count("verify_ownership_from_embeddings") == 3
    assert names.count("train_stealing_surrogate") == 8
    assert all(Path(command["cwd"]) == project for command in commands)
    training = [command["args"] for command in commands if command["name"] == "train_model"]
    assert training[0][training[0].index("--model-role") + 1] == "target"
    assert training[1][training[1].index("--model-role") + 1] == "independent"
    assert all(args[args.index("--device") + 1] == "cpu" for args in training)
    assert all(args[args.index("--epochs") + 1] == "2" for args in training)
    assert all("--validation-fraction" in args and "--validation-seed" in args for args in training)

    # Validate the runner's arguments against the real CLIs, stopping immediately
    # after parsing so no dataset is read and no model is trained.
    class ArgumentsAccepted(BaseException):
        pass

    parse_args = argparse.ArgumentParser.parse_args

    def validate_and_stop(parser, *args, **kwargs):
        parse_args(parser, *args, **kwargs)
        raise ArgumentsAccepted

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", validate_and_stop)
    for command in commands:
        script = ROOT / "scripts" / (command["name"] + ".py")
        monkeypatch.setattr(sys, "argv", [str(script), *command["args"]])
        with pytest.raises(ArgumentsAccepted):
            runpy.run_path(str(script), run_name="__main__")


def test_preprocessing_runs_once_per_dataset_not_once_per_model(runner):
    project, run = runner
    raw_dataset(project)
    result, commands = run(MODELS="gat gin")
    assert result.returncode == 0, result.stderr + result.stdout
    assert sum(command["name"] == "preprocess_data" for command in commands) == 1
    assert sum(command["name"] == "train_model" for command in commands) == 4


@pytest.mark.parametrize("stage", ["preprocess_data", "train_model", "train_stealing_surrogate",
                                  "train_csim_from_embeddings", "verify_ownership_from_embeddings",
                                  "all_model_stealing_evaluation", "organize_results"])
def test_any_failed_stage_stops_the_pipeline(runner, stage):
    project, run = runner
    raw_dataset(project)
    result, commands = run(TEST_FAIL_STAGE=stage)
    assert result.returncode == 7
    assert commands[-1]["name"] == stage
    assert "Experiment pipeline completed" not in result.stdout


def test_missing_raw_input_fails_before_any_training(runner):
    _, run = runner
    result, commands = run()
    assert result.returncode != 0
    assert "data/raw/toy.npz" in result.stderr
    assert commands == []


def test_evaluate_mode_checks_all_inputs_before_running(runner):
    project, run = runner
    prepared_dataset(project, model=False)
    result, commands = run("evaluate")
    assert result.returncode != 0
    assert "gat_toy_target_non-overlapped.pt" in result.stderr
    assert commands == []


def test_evaluate_mode_reuses_prepared_data_and_models(runner):
    project, run = runner
    prepared_dataset(project)
    result, commands = run("evaluate")
    assert result.returncode == 0, result.stderr
    assert [command["name"] for command in commands] == ["all_model_stealing_evaluation", "organize_results"]


def test_csim_requires_real_independent_embeddings(tmp_path):
    from scripts.train_csim_from_embeddings import find_embedding_files

    folder = tmp_path / "non-overlapped/gat_toy"
    folder.mkdir(parents=True)
    target = folder / "gat_toy_target.pt"
    target.touch()
    (folder / "gat_toy_surrogate_original.pt").touch()
    with pytest.raises(ValueError, match="No independent embeddings"):
        find_embedding_files(tmp_path, "gat", "toy")
    independent = folder / "gat_toy_independent_gat.pt"
    independent.touch()
    files = find_embedding_files(tmp_path, "gat", "toy")
    assert files["independents"] == [str(independent)]
    assert str(target) not in files["independents"]


@pytest.mark.parametrize("classification, expected_exit", [(True, 0), (False, 0), (None, 1)])
def test_verification_exit_status_tracks_execution_not_classification(tmp_path, monkeypatch, classification, expected_exit):
    from grove.verification import similarity_model

    class FakeManager:
        def __init__(self, **kwargs):
            pass

        def verify_from_embeddings(self, **kwargs):
            if classification is None:
                raise RuntimeError("simulated verification failure")
            return dict(total_pairs=2, num_similar=1, similarity_percentage=0.5, threshold=0.5,
                        mean_similarity_prob=0.5, decision="surrogate" if classification else "independent",
                        is_surrogate=classification)

    target = tmp_path / "target.pt"
    suspect = tmp_path / "suspect.pt"
    target.touch()
    suspect.touch()
    monkeypatch.setattr(similarity_model, "CsimManager", FakeManager)
    monkeypatch.setattr(sys, "argv", ["verify", "--target-model-name", "target",
                        "--target-embedding", str(target), "--suspect-embedding", str(suspect),
                        "--csim-model-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / "scripts/verify_ownership_from_embeddings.py"), run_name="__main__")
    assert result.value.code == expected_exit
