"""Custom collate function for multimodal batches with optional modalities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def multimodal_collate_fn(
    batch: List[Tuple[Dict[str, Any], int]]
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Collate variable-modality samples into a batch.

    Handles:
      - Tensor modalities: stacked along batch dimension.
      - Dict modalities (speech, text, facial): each key's tensors stacked.
      - None modalities: kept as None in the output batch.

    Args:
        batch: List of (inputs_dict, label) tuples from dataset __getitem__.

    Returns:
        (batched_inputs, labels_tensor)
        batched_inputs: Dict modality → batched tensor/dict or None.
        labels_tensor: (B,) long tensor of labels.
    """
    inputs_list: List[Dict[str, Any]] = [item[0] for item in batch]
    labels: List[int] = [item[1] for item in batch]

    modality_keys = list(inputs_list[0].keys())
    batched: Dict[str, Any] = {}

    for modality in modality_keys:
        samples = [inp[modality] for inp in inputs_list]

        # All None → modality absent from this batch
        if all(s is None for s in samples):
            batched[modality] = None
            continue

        # Replace None samples with zeros (missing within batch)
        if isinstance(samples[0], dict) or any(s is None for s in samples):
            samples = _fill_missing_with_zeros(samples)

        if isinstance(samples[0], dict):
            # Dict modality (speech/text/facial) → stack each key
            batched[modality] = _collate_dict_modality(samples)
        elif isinstance(samples[0], torch.Tensor):
            batched[modality] = torch.stack(samples, dim=0)
        else:
            batched[modality] = samples  # fallback

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return batched, labels_tensor


def _fill_missing_with_zeros(
    samples: List[Optional[Any]]
) -> List[Any]:
    """Replace None entries with zero tensors matching the shape of existing entries."""
    # Find a reference sample
    reference = next(s for s in samples if s is not None)

    filled = []
    for s in samples:
        if s is None:
            filled.append(_zeros_like(reference))
        else:
            filled.append(s)
    return filled


def _zeros_like(sample: Any) -> Any:
    """Recursively create zero tensors matching the structure of sample."""
    if isinstance(sample, torch.Tensor):
        return torch.zeros_like(sample)
    if isinstance(sample, dict):
        return {k: _zeros_like(v) for k, v in sample.items()}
    return sample


def _collate_dict_modality(
    samples: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Stack tensors for each key across samples in a dict-type modality."""
    keys = samples[0].keys()
    batched_dict: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [s[key] for s in samples]
        try:
            batched_dict[key] = torch.stack(tensors, dim=0)
        except Exception:
            batched_dict[key] = tensors  # fallback for variable-length
    return batched_dict
