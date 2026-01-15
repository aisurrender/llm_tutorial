"""Common utilities shared across tutorial steps."""

import math
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "auto") -> str:
    """
    Get the best available device.

    Args:
        preferred: "auto", "cuda", "mps", or "cpu"

    Returns:
        Device string for torch
    """
    if preferred != "auto":
        if preferred == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return "cpu"
        if preferred == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available, falling back to CPU")
            return "cpu"
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Learning rate schedule: Warmup + Cosine Decay.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate

    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step >= total_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
