//! Python bindings for zunigram Poseidon2 PRF.
//!
//! This module exposes the exact same Poseidon2 PRF used in the STARK prover,
//! enabling Python-based LLM watermarking that is compatible with zunigram verification.

use pyo3::prelude::*;
use stwo::core::fields::m31::M31;

use air::{compute_prf, is_green_token, HALF_PRIME};
use common::SECRET_KEY_SIZE;

/// M31 prime constant for Python users
const M31_PRIME: u32 = (1u32 << 31) - 1;

/// Compute PRF(secret_key, token) using Poseidon2.
///
/// Args:
///     secret_key: List of 8 u32 values representing the secret key
///     token: The token ID to hash
///
/// Returns:
///     The PRF output as u32 (in range [0, 2^31-2])
#[pyfunction]
fn prf(secret_key: Vec<u32>, token: u32) -> PyResult<u32> {
    if secret_key.len() != SECRET_KEY_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("secret_key must have exactly {} elements, got {}", SECRET_KEY_SIZE, secret_key.len())
        ));
    }

    let key: [M31; SECRET_KEY_SIZE] = std::array::from_fn(|i| M31::from(secret_key[i]));
    let token_field = M31::from(token);
    let output = compute_prf(&key, token_field);
    
    Ok(output.0)
}

/// Check if a PRF output classifies the token as "green".
///
/// Green tokens have PRF output < HALF_PRIME (gamma = 0.5).
///
/// Args:
///     prf_output: The PRF output value
///
/// Returns:
///     True if the token is green, False if red
#[pyfunction]
fn is_green(prf_output: u32) -> bool {
    is_green_token(M31::from(prf_output))
}

/// Classify a single token as green or red.
///
/// Args:
///     secret_key: List of 8 u32 values representing the secret key
///     token: The token ID to classify
///
/// Returns:
///     True if the token is green, False if red
#[pyfunction]
fn classify_token(secret_key: Vec<u32>, token: u32) -> PyResult<bool> {
    let prf_output = prf(secret_key, token)?;
    Ok(is_green(prf_output))
}

/// Get all green token IDs for a given vocabulary size.
///
/// This precomputes the green list for efficient watermark generation.
///
/// Args:
///     secret_key: List of 8 u32 values representing the secret key
///     vocab_size: The vocabulary size of the tokenizer
///
/// Returns:
///     List of token IDs that are classified as green
#[pyfunction]
fn get_green_tokens(secret_key: Vec<u32>, vocab_size: u32) -> PyResult<Vec<u32>> {
    if secret_key.len() != SECRET_KEY_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("secret_key must have exactly {} elements, got {}", SECRET_KEY_SIZE, secret_key.len())
        ));
    }

    let key: [M31; SECRET_KEY_SIZE] = std::array::from_fn(|i| M31::from(secret_key[i]));
    
    let green_tokens: Vec<u32> = (0..vocab_size)
        .filter(|&token| {
            let token_field = M31::from(token);
            let output = compute_prf(&key, token_field);
            is_green_token(output)
        })
        .collect();
    
    Ok(green_tokens)
}

/// Count green tokens in a sequence.
///
/// Args:
///     secret_key: List of 8 u32 values representing the secret key
///     tokens: List of token IDs to classify
///
/// Returns:
///     Number of green tokens in the sequence
#[pyfunction]
fn count_green(secret_key: Vec<u32>, tokens: Vec<u32>) -> PyResult<u32> {
    if secret_key.len() != SECRET_KEY_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("secret_key must have exactly {} elements, got {}", SECRET_KEY_SIZE, secret_key.len())
        ));
    }

    let key: [M31; SECRET_KEY_SIZE] = std::array::from_fn(|i| M31::from(secret_key[i]));
    
    let count = tokens.iter()
        .filter(|&&token| {
            let token_field = M31::from(token);
            let output = compute_prf(&key, token_field);
            is_green_token(output)
        })
        .count() as u32;
    
    Ok(count)
}

/// Classify multiple tokens at once.
///
/// Args:
///     secret_key: List of 8 u32 values representing the secret key
///     tokens: List of token IDs to classify
///
/// Returns:
///     List of booleans, True for green tokens, False for red
#[pyfunction]
fn classify_tokens(secret_key: Vec<u32>, tokens: Vec<u32>) -> PyResult<Vec<bool>> {
    if secret_key.len() != SECRET_KEY_SIZE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("secret_key must have exactly {} elements, got {}", SECRET_KEY_SIZE, secret_key.len())
        ));
    }

    let key: [M31; SECRET_KEY_SIZE] = std::array::from_fn(|i| M31::from(secret_key[i]));
    
    let classifications: Vec<bool> = tokens.iter()
        .map(|&token| {
            let token_field = M31::from(token);
            let output = compute_prf(&key, token_field);
            is_green_token(output)
        })
        .collect();
    
    Ok(classifications)
}

/// Compute the z-score for watermark detection.
///
/// The z-score measures how many standard deviations the green count
/// is above the expected value for random text.
///
/// Args:
///     green_count: Number of green tokens observed
///     total_tokens: Total number of tokens
///     gamma: Expected proportion of green tokens (default 0.5)
///
/// Returns:
///     The z-score value
#[pyfunction]
#[pyo3(signature = (green_count, total_tokens, gamma=0.5))]
fn compute_z_score(green_count: u32, total_tokens: u32, gamma: f64) -> f64 {
    if total_tokens == 0 {
        return 0.0;
    }
    
    let n = total_tokens as f64;
    let expected = n * gamma;
    let std_dev = (n * gamma * (1.0 - gamma)).sqrt();
    
    if std_dev == 0.0 {
        return 0.0;
    }
    
    (green_count as f64 - expected) / std_dev
}

/// Compute detection threshold from z-score.
///
/// Returns the minimum number of green tokens needed to detect
/// a watermark at the given z-score threshold.
///
/// Args:
///     total_tokens: Total number of tokens
///     z_threshold: Required z-score for detection (e.g., 4.0)
///     gamma: Expected proportion of green tokens (default 0.5)
///
/// Returns:
///     Minimum green count threshold as integer (capped at total_tokens)
#[pyfunction]
#[pyo3(signature = (total_tokens, z_threshold, gamma=0.5))]
fn compute_threshold(total_tokens: u32, z_threshold: f64, gamma: f64) -> u32 {
    let n = total_tokens as f64;
    let expected = n * gamma;
    let std_dev = (n * gamma * (1.0 - gamma)).sqrt();
    
    let threshold = expected + z_threshold * std_dev;
    let threshold_ceil = threshold.ceil() as u32;
    
    // Cap threshold at total_tokens to avoid impossible thresholds
    // When threshold == total_tokens, detection is effectively impossible
    threshold_ceil.min(total_tokens)
}

/// Generate a random secret key.
///
/// Returns:
///     List of 8 u32 values suitable for use as a secret key
#[pyfunction]
fn generate_secret_key() -> Vec<u32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    let mut state = seed;
    let mut key = Vec::with_capacity(SECRET_KEY_SIZE);
    
    for _ in 0..SECRET_KEY_SIZE {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Reduce to M31 range
        key.push((state as u32) % M31_PRIME);
    }
    
    key
}

/// Get the HALF_PRIME constant used for green/red classification.
///
/// Tokens with PRF output < HALF_PRIME are classified as green.
#[pyfunction]
fn get_half_prime() -> u32 {
    HALF_PRIME
}

/// Get the M31 prime constant.
///
/// All field operations are performed modulo this prime (2^31 - 1).
#[pyfunction]
fn get_m31_prime() -> u32 {
    M31_PRIME
}

/// Get the required secret key size (8 elements).
#[pyfunction]
fn get_secret_key_size() -> usize {
    SECRET_KEY_SIZE
}

/// Python module definition
#[pymodule]
fn zunigram_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prf, m)?)?;
    m.add_function(wrap_pyfunction!(is_green, m)?)?;
    m.add_function(wrap_pyfunction!(classify_token, m)?)?;
    m.add_function(wrap_pyfunction!(classify_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(get_green_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(count_green, m)?)?;
    m.add_function(wrap_pyfunction!(compute_z_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(generate_secret_key, m)?)?;
    m.add_function(wrap_pyfunction!(get_half_prime, m)?)?;
    m.add_function(wrap_pyfunction!(get_m31_prime, m)?)?;
    m.add_function(wrap_pyfunction!(get_secret_key_size, m)?)?;
    Ok(())
}

