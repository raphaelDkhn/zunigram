"""Zunigram LLM Watermarking Module.

This module provides STARK-compatible LLM watermarking using the Unigram approach
with Poseidon2 PRF for token classification.
"""

from .watermark import ZunigramWatermark, WatermarkConfig
from .detect import ZunigramDetector, DetectionResult
from .generate import ZunigramGenerator

__all__ = [
    "ZunigramWatermark",
    "WatermarkConfig",
    "ZunigramDetector",
    "DetectionResult",
    "ZunigramGenerator",
]

__version__ = "0.1.0"

