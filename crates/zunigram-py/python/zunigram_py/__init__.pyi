"""Type stubs for zunigram_py native module."""

from typing import List

def prf(secret_key: List[int], token: int) -> int:
    """Compute PRF(secret_key, token) using Poseidon2.
    
    Args:
        secret_key: List of 8 u32 values representing the secret key
        token: The token ID to hash
    
    Returns:
        The PRF output as u32 (in range [0, 2^31-2])
    """
    ...

def is_green(prf_output: int) -> bool:
    """Check if a PRF output classifies the token as green.
    
    Args:
        prf_output: The PRF output value
    
    Returns:
        True if the token is green, False if red
    """
    ...

def classify_token(secret_key: List[int], token: int) -> bool:
    """Classify a single token as green or red.
    
    Args:
        secret_key: List of 8 u32 values representing the secret key
        token: The token ID to classify
    
    Returns:
        True if the token is green, False if red
    """
    ...

def classify_tokens(secret_key: List[int], tokens: List[int]) -> List[bool]:
    """Classify multiple tokens at once.
    
    Args:
        secret_key: List of 8 u32 values representing the secret key
        tokens: List of token IDs to classify
    
    Returns:
        List of booleans, True for green tokens, False for red
    """
    ...

def get_green_tokens(secret_key: List[int], vocab_size: int) -> List[int]:
    """Get all green token IDs for a given vocabulary size.
    
    Args:
        secret_key: List of 8 u32 values representing the secret key
        vocab_size: The vocabulary size of the tokenizer
    
    Returns:
        List of token IDs that are classified as green
    """
    ...

def count_green(secret_key: List[int], tokens: List[int]) -> int:
    """Count green tokens in a sequence.
    
    Args:
        secret_key: List of 8 u32 values representing the secret key
        tokens: List of token IDs to classify
    
    Returns:
        Number of green tokens in the sequence
    """
    ...

def compute_z_score(green_count: int, total_tokens: int, gamma: float = 0.5) -> float:
    """Compute the z-score for watermark detection.
    
    Args:
        green_count: Number of green tokens observed
        total_tokens: Total number of tokens
        gamma: Expected proportion of green tokens (default 0.5)
    
    Returns:
        The z-score value
    """
    ...

def compute_threshold(total_tokens: int, z_threshold: float, gamma: float = 0.5) -> int:
    """Compute detection threshold from z-score.
    
    Args:
        total_tokens: Total number of tokens
        z_threshold: Required z-score for detection (e.g., 4.0)
        gamma: Expected proportion of green tokens (default 0.5)
    
    Returns:
        Minimum green count threshold as integer
    """
    ...

def generate_secret_key() -> List[int]:
    """Generate a random secret key.
    
    Returns:
        List of 8 u32 values suitable for use as a secret key
    """
    ...

def get_half_prime() -> int:
    """Get the HALF_PRIME constant used for green/red classification.
    
    Tokens with PRF output < HALF_PRIME are classified as green.
    """
    ...

def get_m31_prime() -> int:
    """Get the M31 prime constant (2^31 - 1)."""
    ...

def get_secret_key_size() -> int:
    """Get the required secret key size (8 elements)."""
    ...

