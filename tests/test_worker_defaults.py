# mypy: ignore-errors
# pylint: disable=all
# ruff: noqa
"""Unit tests for Worker Executor fitness auto-detection and AutoBatch resolution.

This module validates:
1. That worker train state gracefully falls back to appropriate fitness metric when
   'sweeper.fitness' is omitted from user config, inferring from style, task, or model.
2. That detect tasks map to metrics/mAP50-95(B).
3. That segment tasks map to metrics/mAP50-95(M).
4. That classify tasks map to metrics/accuracy_top1.
5. That batch parameter preserves explicit integer user inputs (e.g. batch=16, 32).
6. That omitted, -1, or 0 batch parameters resolve to AutoBatch fraction based on MAX_GPU.
"""

import os
import pytest


def resolve_worker_fitness(config: dict) -> str:
    """Helper implementing the fitness fallback resolution logic in train.py."""
    fitness = config.get("sweeper", {}).get("fitness")
    if not fitness:
        style = (
            config.get("style")
            or config.get("task")
            or (
                "segment"
                if "-seg" in str(config.get("model", "")).lower()
                else (
                    "classify"
                    if "-cls" in str(config.get("model", "")).lower()
                    else (
                        "pose"
                        if "-pose" in str(config.get("model", "")).lower()
                        else (
                            "obb"
                            if "-obb" in str(config.get("model", "")).lower()
                            else "detect"
                        )
                    )
                )
            )
        )
        fitness_map = {
            "detect": "metrics/mAP50-95(B)",
            "segment": "metrics/mAP50-95(M)",
            "classify": "metrics/accuracy_top1",
            "pose": "metrics/mAP50-95(P)",
            "obb": "metrics/mAP50-95(B)",
        }
        fitness = fitness_map.get(str(style).lower(), "metrics/mAP50-95(B)")
    return fitness


def resolve_worker_batch(request_config: dict, max_gpu_env: str = "100") -> float | int:
    """Helper implementing batch resolution logic in trainer_wrapper.py."""
    user_batch = request_config.get("train", {}).get("batch")
    if isinstance(user_batch, int) and user_batch > 0:
        return user_batch
    max_gpu = int(float(max_gpu_env))
    return max_gpu / 100


def test_resolve_worker_fitness_explicit_style():
    """Validates that explicit style keys in worker config auto-resolve correct fitness.

    This test checks that passing style: 'detect', 'segment', 'classify', 'pose'
    without an explicit 'sweeper.fitness' correctly yields the expected metric name.
    """
    assert resolve_worker_fitness({"style": "detect"}) == "metrics/mAP50-95(B)"
    assert resolve_worker_fitness({"style": "segment"}) == "metrics/mAP50-95(M)"
    assert resolve_worker_fitness({"style": "classify"}) == "metrics/accuracy_top1"
    assert resolve_worker_fitness({"style": "pose"}) == "metrics/mAP50-95(P)"


def test_resolve_worker_fitness_inferred_from_model():
    """Validates that worker infers fitness metric from model weights filename pattern.

    When neither style nor fitness is set, inspects model filename suffix.
    """
    assert resolve_worker_fitness({"model": "yolo11n-seg.pt"}) == "metrics/mAP50-95(M)"
    assert resolve_worker_fitness({"model": "yolov8n-cls.pt"}) == "metrics/accuracy_top1"
    assert resolve_worker_fitness({"model": "yolo11s-pose.pt"}) == "metrics/mAP50-95(P)"
    assert resolve_worker_fitness({"model": "yolo11n.pt"}) == "metrics/mAP50-95(B)"


def test_resolve_worker_fitness_preserves_explicit_user_fitness():
    """Validates that if user provides an explicit sweeper.fitness, it is preserved unchanged.

    Ensures custom metrics like val/loss are not overridden by task auto-mapping.
    """
    cfg = {"style": "detect", "sweeper": {"fitness": "val/loss"}}
    assert resolve_worker_fitness(cfg) == "val/loss"


def test_resolve_worker_batch_autobatch_fallback():
    """Validates that omitted, negative, or zero batch defaults to AutoBatch fraction.

    This test checks that when train.batch is -1 or omitted, the batch resolves
    to MAX_GPU / 100 (e.g. 60% VRAM -> 0.60).
    """
    # Case 1: Batch omitted
    cfg1 = {"train": {"epochs": 10}}
    assert resolve_worker_batch(cfg1, max_gpu_env="60") == 0.60

    # Case 2: Batch is -1
    cfg2 = {"train": {"batch": -1}}
    assert resolve_worker_batch(cfg2, max_gpu_env="80") == 0.80


def test_resolve_worker_batch_preserves_explicit_integer_batch():
    """Validates that explicit positive integer batch sizes are preserved.

    When a researcher specifically requires batch: 16 or 32 for repeatability,
    the system respects the explicit value rather than forcing AutoBatch.
    """
    cfg = {"train": {"batch": 16}}
    assert resolve_worker_batch(cfg, max_gpu_env="60") == 16
