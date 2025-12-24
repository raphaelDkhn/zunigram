# Zunigram

A STARK-based prover for proving unigram LLM-watermark detection using the Stwo prover.

## What is Unigram Watermark Detection?

Unigram watermarking embeds hidden signatures in LLM-generated text. The detection algorithm:

1. Uses a PRF (Poseidon2) with a secret key to classify each token as "green" or "red"
2. Counts the number of green tokens in the text
3. Compares the count against a threshold derived from the z-score

With **γ = 0.5** (half the vocabulary is "green"), unwatermarked text has ~50% green tokens. Watermarked text has significantly more.

This library generates a **zero-knowledge STARK proof** that the green count exceeds the threshold, allowing a public verifier to confirm watermark detection without knowing the secret key.

## CLI Usage

### Installation

**Quick setup** (automated):
```bash
bash cli/setup.sh
source .venv/bin/activate
```

**Requirements**: Python 3.8+, Rust toolchain, transformers, torch

### Commands

Generate, detect, prove, and verify watermarked text:

```bash
# Generate watermarked text
python cli/zunigram_cli.py generate "Write a poem about love" -o output.json --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" -n 200

# Detect watermark
python cli/zunigram_cli.py detect output.json

# Generate STARK proof
python cli/zunigram_cli.py prove output.json -o proof.json

# Verify proof
python cli/zunigram_cli.py verify proof.json
```

## Benchmarks

```bash
$ cargo run -p benchmarks --release
```

| Tokens | Green | Prove Time | Verify Time |
| ------ | ----- | ---------- | ----------- |
| 256    | ~128  | ~16ms      | ~644.54µs   |
| 512    | ~256  | ~30ms      | ~607.63µs   |
| 1024   | ~512  | ~90ms      | ~787.04µs   |
| 2048   | ~1024 | ~320ms     | ~1.12ms     |

## How It Works

### AIR Components

**PRF Component** (`air/prf/`):

- Proves Poseidon2 permutation for each token
- Provides (token, prf_output) via LogUp

**Unigram Component** (`air/unigram/`):

- Consumes (token, prf_output) from PRF
- Computes `is_green` from `prf_output` via range check
- Accumulates green count
- Binds public inputs via LogUp

### Constraints

1. **Boolean**: `is_green ∈ {0, 1}`, `is_padding ∈ {0, 1}`
2. **Range Check** for computing `is_green` from `prf_output`
3. **Accumulator**: `green_count[i] = green_count[i-1] + is_green[i]`
4. **LogUp Balance**: PRF provides, Unigram consumes

## References

- [Unigram Watermark Paper](https://arxiv.org/pdf/2306.17439)
- [Stwo Prover](https://github.com/starkware-libs/stwo)
- [Stwo Documentation](https://zksecurity.github.io/stwo-book/)

## License

MIT
