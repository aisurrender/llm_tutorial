"""Error handling and validation utilities for the tutorial."""

import sys
from pathlib import Path


class TutorialError(Exception):
    """Base exception for tutorial-related errors."""


class DataNotFoundError(TutorialError):
    """Raised when required data files are missing."""


class DeviceNotAvailableError(TutorialError):
    """Raised when requested device is not available."""


def check_file_exists(filepath: str | Path, create_sample_fn=None) -> Path:
    """
    Check if a file exists. Optionally create sample data if missing.

    Args:
        filepath: Path to check
        create_sample_fn: Optional function to create sample data

    Returns:
        Path object

    Raises:
        DataNotFoundError: If file doesn't exist and no create function provided
    """
    path = Path(filepath)
    if not path.exists():
        if create_sample_fn is not None:
            print(f"Data file not found: {filepath}")
            print("Creating sample data...")
            create_sample_fn(filepath)
            return path

        raise DataNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please ensure the file exists or provide sample data creation function."
        )
    return path


def check_checkpoint_exists(checkpoint_path: str | Path) -> Path:
    """
    Check if a checkpoint file exists.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Path object

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train a model first or provide a valid checkpoint path."
        )
    return path


def validate_device(device: str) -> str:
    """
    Validate and potentially fix device string.

    Args:
        device: Requested device ("cuda", "mps", "cpu", "auto")

    Returns:
        Valid device string

    Raises:
        DeviceNotAvailableError: If requested device is not available
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise DeviceNotAvailableError(
            "CUDA is not available on this machine.\n"
            "Options:\n"
            "  1. Use --device cpu (slower)\n"
            "  2. Use --device mps (Mac M1/M2/M3)\n"
            "  3. Install CUDA-compatible PyTorch"
        )

    if device == "mps" and not torch.backends.mps.is_available():
        raise DeviceNotAvailableError(
            "MPS is not available on this machine.\n"
            "MPS requires macOS 12.3+ with M1/M2/M3 chip.\n"
            "Use --device cpu instead."
        )

    return device


def setup_tutorial_path():
    """Add the tutorial root to Python path for cross-directory imports."""
    tutorial_root = Path(__file__).parent.parent
    if str(tutorial_root) not in sys.path:
        sys.path.insert(0, str(tutorial_root))
    return tutorial_root
