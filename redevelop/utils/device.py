# encoding: utf-8
"""
Device utilities for cross-platform support (CUDA, MPS, CPU)
"""

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Args:
        preferred: Device preference - "auto", "cuda", "mps", or "cpu"
                   "auto" will select the best available accelerator
    
    Returns:
        torch.device: The selected device
    
    Examples:
        >>> device = get_device()  # Auto-detect best device
        >>> device = get_device("mps")  # Force MPS on Apple Silicon
        >>> device = get_device("cpu")  # Force CPU
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        dict: Information about available devices
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "recommended": "cpu"
    }
    
    if info["cuda_available"]:
        info["recommended"] = "cuda"
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    elif info["mps_available"]:
        info["recommended"] = "mps"
    
    return info

