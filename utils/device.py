"""Device selection utilities."""
from __future__ import annotations

import torch


def get_device(preferred: str = "cuda") -> torch.device:
    """Return the best available device.

    Args:
        preferred: Preferred device string ('cuda' or 'cpu').

    Returns:
        torch.device instance.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device
