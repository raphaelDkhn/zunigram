# zunigram-py

Python bindings for the zunigram Poseidon2 PRF, enabling STARK-compatible LLM watermark detection.

## Installation

```bash
# From the zunigram-py directory
pip install maturin
maturin develop --release
```

## Usage

```python
import zunigram_py

# Generate or use a fixed secret key (8 x u32)
secret_key = [0x12345678] * 8

# Classify a single token
is_green = zunigram_py.classify_token(secret_key, token_id=42)

# Get all green tokens for a vocabulary
green_tokens = zunigram_py.get_green_tokens(secret_key, vocab_size=32000)

# Count green tokens in a sequence
tokens = [100, 200, 300, 400, 500]
green_count = zunigram_py.count_green(secret_key, tokens)

# Compute z-score for detection
z_score = zunigram_py.compute_z_score(green_count, len(tokens))

# Get threshold for detection at z=4.0
threshold = zunigram_py.compute_threshold(len(tokens), z_threshold=4.0)
```

## Compatibility

This module uses the exact same Poseidon2 implementation as the zunigram STARK prover.
Watermarks generated with this module can be verified using zunigram's zero-knowledge proofs.

