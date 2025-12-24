"""zunigram_py - Python bindings for zunigram Poseidon2 PRF.

This module provides STARK-compatible LLM watermark detection using the
exact same Poseidon2 PRF as the zunigram STARK prover.
"""

from .zunigram_py import (
    prf,
    is_green,
    classify_token,
    classify_tokens,
    get_green_tokens,
    count_green,
    compute_z_score,
    compute_threshold,
    generate_secret_key,
    get_half_prime,
    get_m31_prime,
    get_secret_key_size,
)

__all__ = [
    "prf",
    "is_green",
    "classify_token",
    "classify_tokens",
    "get_green_tokens",
    "count_green",
    "compute_z_score",
    "compute_threshold",
    "generate_secret_key",
    "get_half_prime",
    "get_m31_prime",
    "get_secret_key_size",
]

__version__ = "0.1.0"

