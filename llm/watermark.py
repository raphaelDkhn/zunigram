from dataclasses import dataclass, field
from typing import List, Optional, Set
import numpy as np

try:
    import zunigram_py
except ImportError:
    raise ImportError(
        "zunigram_py not installed. Build with: cd crates/zunigram-py && maturin develop --release"
    )


@dataclass
class WatermarkConfig:
    """Configuration for watermark generation and detection.
    
    Attributes:
        secret_key: List of 8 u32 values for the Poseidon2 PRF
        gamma: Proportion of vocabulary classified as "green" (default 0.5)
        delta: Logit bias added to green tokens during generation
        z_threshold: Z-score threshold for watermark detection
    """
    secret_key: List[int]
    gamma: float = 0.5
    delta: float = 2.0 
    z_threshold: float = 3.0
    
    def __post_init__(self):
        expected_size = zunigram_py.get_secret_key_size()
        if len(self.secret_key) != expected_size:
            raise ValueError(
                f"secret_key must have exactly {expected_size} elements, got {len(self.secret_key)}"
            )
        
        m31_prime = zunigram_py.get_m31_prime()
        for i, val in enumerate(self.secret_key):
            if not (0 <= val < m31_prime):
                raise ValueError(
                    f"secret_key[{i}] = {val} is out of range [0, {m31_prime})"
                )
    
    @classmethod
    def with_seed(cls, seed: int, **kwargs) -> "WatermarkConfig":
        """Create config with a deterministic secret key from a seed.
        
        Args:
            seed: Integer seed for key derivation
            **kwargs: Additional config parameters
        
        Returns:
            WatermarkConfig with derived secret key
        """
        # Simple deterministic key derivation from seed
        m31_prime = zunigram_py.get_m31_prime()
        key_size = zunigram_py.get_secret_key_size()
        
        # Use a simple hash-like derivation
        state = seed & 0xFFFFFFFFFFFFFFFF
        secret_key = []
        for _ in range(key_size):
            # xorshift64
            state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
            state ^= (state >> 7) & 0xFFFFFFFFFFFFFFFF
            state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
            secret_key.append((state & 0xFFFFFFFF) % m31_prime)
        
        return cls(secret_key=secret_key, **kwargs)


class ZunigramWatermark:
    """Unigram watermarking using Poseidon2 PRF.
    
    This class provides the core functionality for watermark generation and
    detection using the same PRF as the zunigram STARK prover.
    
    Example:
        >>> config = WatermarkConfig.with_seed(42)
        >>> watermark = ZunigramWatermark(config, vocab_size=32000)
        >>> 
        >>> # Check if a token is green
        >>> is_green = watermark.is_green(token_id=1234)
        >>> 
        >>> # Get bias mask for logits
        >>> bias = watermark.get_logit_bias()
        >>> biased_logits = logits + bias
    """
    
    def __init__(self, config: WatermarkConfig, vocab_size: int):
        """Initialize watermark with config and vocabulary size.
        
        Args:
            config: Watermark configuration with secret key
            vocab_size: Size of the tokenizer vocabulary
        """
        self.config = config
        self.vocab_size = vocab_size
        
        # Precompute green token set for efficiency
        self._green_tokens: Optional[Set[int]] = None
        self._green_mask: Optional[np.ndarray] = None
    
    @property
    def green_tokens(self) -> Set[int]:
        """Get the set of green token IDs (lazily computed)."""
        if self._green_tokens is None:
            green_list = zunigram_py.get_green_tokens(
                self.config.secret_key, self.vocab_size
            )
            self._green_tokens = set(green_list)
        return self._green_tokens
    
    @property
    def green_mask(self) -> np.ndarray:
        """Get boolean mask of green tokens (lazily computed)."""
        if self._green_mask is None:
            self._green_mask = np.zeros(self.vocab_size, dtype=bool)
            for token_id in self.green_tokens:
                self._green_mask[token_id] = True
        return self._green_mask
    
    def is_green(self, token_id: int) -> bool:
        """Check if a token is classified as green.
        
        Args:
            token_id: The token ID to check
        
        Returns:
            True if the token is green, False if red
        """
        return zunigram_py.classify_token(self.config.secret_key, token_id)
    
    def classify_tokens(self, token_ids: List[int]) -> List[bool]:
        """Classify multiple tokens as green or red.
        
        Args:
            token_ids: List of token IDs to classify
        
        Returns:
            List of booleans (True = green, False = red)
        """
        return zunigram_py.classify_tokens(self.config.secret_key, token_ids)
    
    def count_green(self, token_ids: List[int]) -> int:
        """Count the number of green tokens in a sequence.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Number of green tokens
        """
        return zunigram_py.count_green(self.config.secret_key, token_ids)
    
    def get_logit_bias(self) -> np.ndarray:
        """Get logit bias array to add to model logits.
        
        Green tokens get +delta bias, red tokens get 0.
        
        Returns:
            NumPy array of shape (vocab_size,) with bias values
        """
        bias = np.zeros(self.vocab_size, dtype=np.float32)
        bias[self.green_mask] = self.config.delta
        return bias
    
    def apply_watermark_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply watermark bias to logits.
        
        Args:
            logits: Model output logits of shape (..., vocab_size)
        
        Returns:
            Biased logits with green tokens promoted
        """
        bias = self.get_logit_bias()
        return logits + bias
    
    def compute_z_score(self, token_ids: List[int]) -> float:
        """Compute the z-score for watermark detection.
        
        Args:
            token_ids: List of token IDs from generated text
        
        Returns:
            Z-score indicating watermark presence
        """
        green_count = self.count_green(token_ids)
        return zunigram_py.compute_z_score(
            green_count, len(token_ids), self.config.gamma
        )
    
    def compute_threshold(self, num_tokens: int) -> int:
        """Compute minimum green count for detection at configured z-threshold.
        
        Args:
            num_tokens: Total number of tokens
        
        Returns:
            Minimum green count threshold
        """
        return zunigram_py.compute_threshold(
            num_tokens, self.config.z_threshold, self.config.gamma
        )
    
    def detect(self, token_ids: List[int]) -> bool:
        """Detect if a token sequence contains a watermark.
        
        Args:
            token_ids: List of token IDs from text
        
        Returns:
            True if watermark detected, False otherwise
        """
        z_score = self.compute_z_score(token_ids)
        return z_score >= self.config.z_threshold
    
    def get_prf_output(self, token_id: int) -> int:
        """Get raw PRF output for a token (for debugging).
        
        Args:
            token_id: The token ID
        
        Returns:
            PRF output value
        """
        return zunigram_py.prf(self.config.secret_key, token_id)

