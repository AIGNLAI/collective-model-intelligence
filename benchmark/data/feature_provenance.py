"""Validate the frozen CLIP feature identity used by report Section 6.3."""

from collections.abc import Mapping
from typing import Any

import numpy as np

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIP_FEATURE_PROTOCOL = "clip-b32-full-text-chunks-all-images-equal-concat-l2-v1"


def validate_multimodal_identity(identity: Any, *, frozen: bool = True) -> int:
    expected = {
        "model_id": CLIP_MODEL_ID,
        "revision": CLIP_MODEL_REVISION,
        "protocol": CLIP_FEATURE_PROTOCOL,
        "dimension": 1024,
        "component_dimension": 512,
        "text_pooling": "token_weighted_full_text_chunks_75",
        "image_pooling": "equal_mean_all_images",
        "fusion": "concatenate_normalized_modalities_div_sqrt2",
        "normalized": True,
        "modalities": ["text", "images"],
    }
    if not isinstance(identity, Mapping) or any(
        identity.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("The report requires complete, consistent CLIP encoder provenance")
    if frozen:
        checksum = identity.get("feature_archive_sha256", "")
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Multimodal feature archive must have a frozen SHA256 identity")
    return 1024


def validate_features(values: Any, rows: int, dimension: int = 1024) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (rows, dimension) or not np.isfinite(matrix).all():
        raise ValueError(f"Multimodal features must be finite with shape [{rows}, {dimension}]")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
        raise ValueError("Multimodal features must be L2-normalized and nonzero")
    return matrix
