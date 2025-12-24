from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from .watermark import ZunigramWatermark, WatermarkConfig

try:
    import zunigram_py
except ImportError:
    raise ImportError(
        "zunigram_py not installed. Build with: cd crates/zunigram-py && maturin develop --release"
    )


@dataclass
class DetectionResult:
    """Result of watermark detection.
    
    Attributes:
        is_watermarked: Whether watermark was detected
        z_score: Computed z-score
        p_value: Statistical p-value for detection
        green_count: Number of green tokens found
        total_tokens: Total number of tokens analyzed
        threshold: Green count threshold used
        confidence: Confidence level (1 - p_value)
    """
    is_watermarked: bool
    z_score: float
    p_value: float
    green_count: int
    total_tokens: int
    threshold: int
    confidence: float
    
    @property
    def green_ratio(self) -> float:
        """Proportion of green tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.green_count / self.total_tokens
    
    def __str__(self) -> str:
        status = "WATERMARKED" if self.is_watermarked else "NOT WATERMARKED"
        result = (
            f"Detection Result: {status}\n"
            f"  Z-score: {self.z_score:.4f}\n"
            f"  P-value: {self.p_value:.2e}\n"
            f"  Confidence: {self.confidence:.4%}\n"
            f"  Green tokens: {self.green_count}/{self.total_tokens} "
            f"({self.green_ratio:.2%})\n"
            f"  Threshold: {self.threshold}"
        )
        
        # Warn when threshold equals total tokens (detection effectively impossible)
        if self.threshold >= self.total_tokens:
            result += f"\n  ⚠ Note: Threshold equals total tokens - detection unreliable with {self.total_tokens} tokens"
        
        return result
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_watermarked": self.is_watermarked,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "green_count": self.green_count,
            "total_tokens": self.total_tokens,
            "green_ratio": self.green_ratio,
            "threshold": self.threshold,
            "confidence": self.confidence,
        }
    
    def for_stark_proof(self) -> dict:
        """Get data needed for STARK proof generation.
        
        Returns:
            Dictionary with token_ids, green_count, threshold, and secret_key
            that can be passed to the zunigram prover.
        """
        return {
            "green_count": self.green_count,
            "total_tokens": self.total_tokens,
            "threshold": self.threshold,
        }


class ZunigramDetector:
    """Watermark detector using Poseidon2 PRF.
    
    This detector uses the same PRF as the zunigram STARK prover,
    enabling zero-knowledge verification of detection results.
    
    Example:
        >>> config = WatermarkConfig.with_seed(42)
        >>> detector = ZunigramDetector(config, vocab_size=32000)
        >>> 
        >>> result = detector.detect(token_ids)
        >>> print(result)
        Detection Result: WATERMARKED
          Z-score: 6.2345
          P-value: 2.31e-10
          ...
    """
    
    def __init__(
        self,
        watermark_config: WatermarkConfig,
        vocab_size: int,
    ):
        """Initialize detector.
        
        Args:
            watermark_config: Watermark configuration with secret key
            vocab_size: Vocabulary size of the tokenizer
        """
        self.config = watermark_config
        self.vocab_size = vocab_size
        self.watermark = ZunigramWatermark(watermark_config, vocab_size)
    
    def detect(
        self,
        token_ids: List[int],
        z_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Detect watermark in a token sequence.
        
        Args:
            token_ids: List of token IDs to analyze
            z_threshold: Override z-score threshold (uses config default if None)
        
        Returns:
            DetectionResult with detection statistics
        """
        if z_threshold is None:
            z_threshold = self.config.z_threshold
        
        # Count green tokens
        green_count = self.watermark.count_green(token_ids)
        total_tokens = len(token_ids)
        
        # Compute z-score
        z_score = zunigram_py.compute_z_score(
            green_count, total_tokens, self.config.gamma
        )
        
        # Compute threshold
        threshold = zunigram_py.compute_threshold(
            total_tokens, z_threshold, self.config.gamma
        )
        
        # Compute p-value using normal CDF approximation
        p_value = self._compute_p_value(z_score)
        
        # Determine if watermarked
        is_watermarked = z_score >= z_threshold
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            z_score=z_score,
            p_value=p_value,
            green_count=green_count,
            total_tokens=total_tokens,
            threshold=threshold,
            confidence=1.0 - p_value,
        )
    
    def detect_text(
        self,
        text: str,
        tokenizer,
        z_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Detect watermark in text using a tokenizer.
        
        Args:
            text: Text to analyze
            tokenizer: HuggingFace tokenizer or compatible
            z_threshold: Override z-score threshold
        
        Returns:
            DetectionResult with detection statistics
        """
        # Tokenize text
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return self.detect(token_ids, z_threshold)
    
    def analyze_tokens(
        self,
        token_ids: List[int],
    ) -> List[Tuple[int, bool, int]]:
        """Analyze individual token classifications.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            List of (token_id, is_green, prf_output) tuples
        """
        results = []
        for token_id in token_ids:
            prf_output = zunigram_py.prf(self.config.secret_key, token_id)
            is_green = zunigram_py.is_green(prf_output)
            results.append((token_id, is_green, prf_output))
        return results
    
    def get_detection_curve(
        self,
        max_tokens: int = 500,
        z_threshold: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """Get (num_tokens, threshold) pairs for detection curve.
        
        Useful for understanding detection sensitivity at different lengths.
        
        Args:
            max_tokens: Maximum number of tokens to compute
            z_threshold: Z-score threshold to use
        
        Returns:
            List of (num_tokens, green_threshold) tuples
        """
        if z_threshold is None:
            z_threshold = self.config.z_threshold
        
        curve = []
        for n in range(10, max_tokens + 1, 10):
            threshold = zunigram_py.compute_threshold(n, z_threshold, self.config.gamma)
            curve.append((n, threshold))
        return curve
    
    def _compute_p_value(self, z_score: float) -> float:
        """Compute one-tailed p-value from z-score.
        
        Uses the complementary error function for accuracy.
        """
        # For large positive z-scores, return very small p-value
        if z_score > 8:
            return 1e-15
        if z_score < -8:
            return 1.0
        
        # Standard normal CDF: P(Z > z) = 0.5 * erfc(z / sqrt(2))
        return 0.5 * math.erfc(z_score / math.sqrt(2))
    
    def estimate_false_positive_rate(
        self,
        z_threshold: Optional[float] = None,
    ) -> float:
        """Estimate false positive rate at the given threshold.
        
        Returns:
            Probability of detecting watermark in random text
        """
        if z_threshold is None:
            z_threshold = self.config.z_threshold
        
        return self._compute_p_value(z_threshold)


class BatchDetector:
    """Batch watermark detection for multiple texts.
    
    Optimized for processing many texts with the same configuration.
    """
    
    def __init__(
        self,
        watermark_config: WatermarkConfig,
        vocab_size: int,
    ):
        """Initialize batch detector.
        
        Args:
            watermark_config: Watermark configuration
            vocab_size: Vocabulary size
        """
        self.detector = ZunigramDetector(watermark_config, vocab_size)
    
    def detect_batch(
        self,
        token_sequences: List[List[int]],
        z_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Detect watermarks in multiple token sequences.
        
        Args:
            token_sequences: List of token ID lists
            z_threshold: Override z-score threshold
        
        Returns:
            List of DetectionResult objects
        """
        return [
            self.detector.detect(tokens, z_threshold)
            for tokens in token_sequences
        ]
    
    def detect_texts(
        self,
        texts: List[str],
        tokenizer,
        z_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Detect watermarks in multiple texts.
        
        Args:
            texts: List of texts to analyze
            tokenizer: HuggingFace tokenizer or compatible
            z_threshold: Override z-score threshold
        
        Returns:
            List of DetectionResult objects
        """
        return [
            self.detector.detect_text(text, tokenizer, z_threshold)
            for text in texts
        ]
    
    def compute_statistics(
        self,
        results: List[DetectionResult],
    ) -> dict:
        """Compute aggregate statistics from detection results.
        
        Args:
            results: List of DetectionResult objects
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {"count": 0}
        
        detected = sum(1 for r in results if r.is_watermarked)
        z_scores = [r.z_score for r in results]
        green_ratios = [r.green_ratio for r in results]
        
        return {
            "count": len(results),
            "detected_count": detected,
            "detection_rate": detected / len(results),
            "mean_z_score": sum(z_scores) / len(z_scores),
            "min_z_score": min(z_scores),
            "max_z_score": max(z_scores),
            "mean_green_ratio": sum(green_ratios) / len(green_ratios),
        }

