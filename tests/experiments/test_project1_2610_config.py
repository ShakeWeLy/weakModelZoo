from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "segment" / "project1_2610"
sys.path.insert(0, str(PROJECT_DIR))

from trainer.config import apply_overrides, available_models, build_config, deep_merge, model_output_dir  # noqa: E402
from trainer.labels import LabelSpace  # noqa: E402
from trainer.recorder import unique_run_name  # noqa: E402


def test_deep_merge_overrides_nested_without_mutating_base():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    merged = deep_merge(base, {"a": {"y": 3}, "c": 4})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base["a"]["y"] == 2


def test_model_config_overrides_global():
    cfg = build_config("swin_unet")
    assert cfg["hyperparameters"]["optimizer"] == "sgd"
    assert cfg["hyperparameters"]["weight_decay"] == 0.0001
    assert cfg["model"]["params"]["embed_dim"] == 96
    assert "quick" not in cfg
    assert model_output_dir(cfg) == PROJECT_DIR / "Swin-UNet" / "runs"


def test_quick_and_cli_overrides_order():
    cfg = build_config("swin_unet", quick=True, epochs=7,
                       overrides=["model.params.embed_dim=24", "dataset.batch_size=3"])
    assert cfg["hyperparameters"]["num_epochs"] == 7
    assert cfg["train"]["limit_batches"] == 5
    assert cfg["model"]["params"]["depths"] == [1, 1, 1, 1]
    assert cfg["model"]["params"]["embed_dim"] == 24
    assert cfg["dataset"]["batch_size"] == 3


def test_apply_overrides_falls_back_to_string():
    cfg = apply_overrides({}, ["train.device=cuda:1", "train.ids=[1, 2]"])
    assert cfg["train"] == {"device": "cuda:1", "ids": [1, 2]}


@pytest.mark.parametrize("model", available_models())
def test_every_model_config_has_arch(model):
    cfg = build_config(model)
    assert cfg["model"]["arch"]
    assert cfg["model"]["display_name"]


def test_label_space():
    space = LabelSpace.from_config(build_config("unet"))
    assert space.num_classes == 9
    assert space.metric_class_ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert space.class_name(0) == "background"
    assert space.class_name(8) == "pancreas"
    assert space.class_name_cn(5) == "肝脏"

    bad = apply_overrides(build_config("unet"), ["train.metric_class_ids=[1, 9]"])
    with pytest.raises(ValueError):
        LabelSpace.from_config(bad)


def test_unique_run_name(tmp_path):
    assert unique_run_name(tmp_path, "exp_V1") == "exp_V1"
    (tmp_path / "exp_V1").mkdir()
    (tmp_path / "exp_V2").mkdir()
    assert unique_run_name(tmp_path, "exp_V1") == "exp_V3"
    (tmp_path / "plain").mkdir()
    assert unique_run_name(tmp_path, "plain") == "plain_2"
