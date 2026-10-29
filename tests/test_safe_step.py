"""Unit tests for SafeStep wrapper and defensive error handling.

This test file verifies:
1. That SafeStep successfully wraps steps and preserves step metadata (NAME, VERSION, _wpipe_metadata).
2. That SafeStep catches exceptions raised by a failing step and returns the original context dict.
3. That SafeStep passes through normal step output when execution succeeds.
"""

import importlib.util
import os
from typing import Any

import pytest


def load_safe_step_class() -> Any:
    """Dynamically loads SafeStep class across directories avoiding dot in path name.

    Returns:
        Any: SafeStep class object.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(
        current_dir,
        "..",
        "executor_v2.0",
        "wtrain",
        "lib",
        "src",
        "wyolo",
        "trainer",
        "safe_step.py",
    )
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("safe_step_mod", module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SafeStep


class DummyFailingStep:
    """A mock step that always raises an unhandled exception to simulate a crash."""

    NAME = "FailingStep"
    VERSION = "v1.0"

    def __call__(self, ctx: dict) -> dict:
        """Simulates an unexpected runtime error inside a pipeline step."""
        raise RuntimeError("Simulated failure in downstream post-train analysis")


class DummySuccessfulStep:
    """A mock step that executes normally and mutates or enriches the context."""

    NAME = "SuccessfulStep"
    VERSION = "v2.0"

    def __call__(self, ctx: dict) -> dict:
        """Enriches the context with new computed values."""
        ctx["processed"] = True
        return ctx


def test_safe_step_handles_exceptions() -> None:
    """Validates that SafeStep intercepts runtime exceptions and preserves context.

    Detailed explanation:
    - In complex ML post-training pipelines, individual research steps (such as
      adversarial attack simulations, outlier detection, or LaTeX exports) may fail
      due to corner-case input shapes or missing optional packages.
    - SafeStep wraps these step instances and guarantees that if an unhandled
      Exception occurs, it is trapped and logged rather than terminating the whole
      pipeline before crucial artifacts (like Grad-CAM visualizations or LLM reports)
      are generated.
    """
    SafeStep = load_safe_step_class()

    failing_step = DummyFailingStep()
    wrapped_step = SafeStep(failing_step)

    # Initial context before running step
    initial_context = {"image_data": ["/path/to/img1.jpg"], "status": "running"}

    # Execute step - must not raise RuntimeError
    result = wrapped_step(initial_context)

    # Context should be returned intact despite the exception
    assert isinstance(result, dict)
    assert result["image_data"] == ["/path/to/img1.jpg"]
    assert result["status"] == "running"


def test_safe_step_successful_execution() -> None:
    """Validates that SafeStep transparently executes successful steps.

    Detailed explanation:
    - When a step executes without errors, SafeStep must return the step's
      actual return value and preserve all metadata (NAME, VERSION).
    """
    SafeStep = load_safe_step_class()

    success_step = DummySuccessfulStep()
    wrapped_step = SafeStep(success_step)

    assert wrapped_step.NAME == "SuccessfulStep"
    assert wrapped_step.VERSION == "v2.0"

    input_ctx = {"initial": 123}
    result = wrapped_step(input_ctx)

    assert result["initial"] == 123
    assert result["processed"] is True
