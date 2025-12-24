//! PRF witness/trace generation.
//!
//! Generates execution traces for Poseidon2 PRF computation.

use itertools::Itertools;
use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::ColumnVec;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES, PackedBaseField};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

use common::SECRET_KEY_SIZE;

use super::component::{
    apply_external_round_matrix, apply_internal_round_matrix, pow5,
    EXTERNAL_ROUND_CONSTS, INTERNAL_ROUND_CONSTS, N_COLUMNS_PER_INSTANCE,
    N_HALF_FULL_ROUNDS, N_PARTIAL_ROUNDS, N_STATE,
};
use super::table::PrfLookupData;
use super::PrfLookupElements;

// ============================================================================
// Poseidon2 Helpers
// ============================================================================

/// Perform the full Poseidon2 permutation on a state.
pub fn poseidon2_permutation(state: &mut [BaseField; N_STATE]) {
    // First half of full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] += EXTERNAL_ROUND_CONSTS[round][i];
        }
        apply_external_round_matrix(state);
        for i in 0..N_STATE {
            state[i] = pow5(state[i]);
        }
    }

    // Partial rounds
    for round in 0..N_PARTIAL_ROUNDS {
        state[0] += INTERNAL_ROUND_CONSTS[round];
        apply_internal_round_matrix(state);
        state[0] = pow5(state[0]);
    }

    // Second half of full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] += EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i];
        }
        apply_external_round_matrix(state);
        for i in 0..N_STATE {
            state[i] = pow5(state[i]);
        }
    }
}

/// Compute PRF(secret_key, token) = Poseidon2(secret_key || token || padding)[0].
pub fn compute_prf(secret_key: &[BaseField; SECRET_KEY_SIZE], token: BaseField) -> BaseField {
    let mut state = [BaseField::zero(); N_STATE];

    // Load secret key into first 8 positions
    for (i, &key_elem) in secret_key.iter().enumerate() {
        state[i] = key_elem;
    }

    // Load token into position 8
    state[SECRET_KEY_SIZE] = token;

    // Positions 9-15 are padding (zeros), with domain separator
    state[N_STATE - 1] = BaseField::one();

    poseidon2_permutation(&mut state);

    state[0]
}

/// Check if a token is green based on PRF output.
/// Green if prf_output < M31_PRIME/2 (gives gamma = 0.5).
pub fn is_green_token(prf_output: BaseField) -> bool {
    const HALF_PRIME: u32 = (1u32 << 31) / 2;
    prf_output.0 < HALF_PRIME
}

// ============================================================================
// Trace Generation
// ============================================================================

/// Generate PRF trace.
/// Each row computes PRF(secret_key, token) for one token.
/// Returns trace columns and lookup data for LogUp.
pub fn gen_prf_trace(
    log_size: u32,
    tokens: &[BaseField],
    secret_key: &[BaseField; SECRET_KEY_SIZE],
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    PrfLookupData,
) {
    assert!(log_size >= LOG_N_LANES);

    let n_rows = 1 << log_size;
    let n_tokens = tokens.len();

    // Initialize columns
    let mut trace = (0..N_COLUMNS_PER_INSTANCE)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(n_rows))
        .collect_vec();

    let mut lookup_data = PrfLookupData::new(n_rows);

    // Process tokens using SIMD (N_LANES tokens per vector)
    let n_vecs = 1 << (log_size - LOG_N_LANES);
    let results: Vec<_> = (0..n_vecs).into_par_iter().map(|vec_index| {
        let mut trace_values = Vec::with_capacity(N_COLUMNS_PER_INSTANCE);

        // Build initial state for N_LANES tokens
        let mut states: [PackedBaseField; N_STATE] = std::array::from_fn(|state_i| {
            PackedBaseField::from_array(std::array::from_fn(|lane| {
                let row_idx = vec_index * N_LANES + lane;
                if state_i < SECRET_KEY_SIZE {
                    secret_key[state_i]
                } else if state_i == SECRET_KEY_SIZE {
                    if row_idx < n_tokens {
                        tokens[row_idx]
                    } else {
                        BaseField::zero()
                    }
                } else if state_i == N_STATE - 1 {
                    BaseField::one()
                } else {
                    BaseField::zero()
                }
            }))
        });

        // Store tokens for lookup
        let token_value = states[SECRET_KEY_SIZE];

        // Write initial state to trace
        states.iter().for_each(|s| {
            trace_values.push(*s);
        });

        // First half of full rounds
        for round in 0..N_HALF_FULL_ROUNDS {
            for i in 0..N_STATE {
                states[i] += PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round][i]);
            }
            apply_external_round_matrix(&mut states);
            states = std::array::from_fn(|i| pow5(states[i]));
            states.iter().for_each(|s| {
                trace_values.push(*s);
            });
        }

        // Partial rounds
        for round in 0..N_PARTIAL_ROUNDS {
            states[0] += PackedBaseField::broadcast(INTERNAL_ROUND_CONSTS[round]);
            apply_internal_round_matrix(&mut states);
            states[0] = pow5(states[0]);
            trace_values.push(states[0]);
        }

        // Second half of full rounds
        for round in 0..N_HALF_FULL_ROUNDS {
            for i in 0..N_STATE {
                states[i] +=
                    PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i]);
            }
            apply_external_round_matrix(&mut states);
            states = std::array::from_fn(|i| pow5(states[i]));
            states.iter().for_each(|s| {
                trace_values.push(*s);
            });
        }

        // Store PRF output for lookup
        let prf_output = states[0];

        (vec_index, token_value, prf_output, trace_values)
    }).collect();

    // Write results sequentially
    for (vec_index, token_value, prf_output, trace_values) in results {
        lookup_data.tokens.data[vec_index] = token_value;
        lookup_data.prf_outputs.data[vec_index] = prf_output;
        for (col_idx, value) in trace_values.iter().enumerate() {
            trace[col_idx].data[vec_index] = *value;
        }
    }

    // Apply bit-reversal for local constraint evaluation
    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_par_iter()
        .map(|mut col| {
            bit_reverse_coset_to_circle_domain_order(&mut col.as_mut_slice());
            CircleEvaluation::new(domain, col)
        })
        .collect();

    (trace, lookup_data)
}

/// Generate interaction trace for PRF component.
/// Produces LogUp columns for (token, prf_output) provision.
pub fn gen_prf_interaction_trace(
    log_size: u32,
    lookup_data: &PrfLookupData,
    lookup_elements: &PrfLookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = unsafe { LogupTraceGenerator::uninitialized(log_size) };

    // Clone and bit-reverse lookup data to match trace order
    let mut tokens_col = lookup_data.tokens.clone();
    let mut prf_outputs_col = lookup_data.prf_outputs.clone();

    bit_reverse_coset_to_circle_domain_order(&mut tokens_col.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(&mut prf_outputs_col.as_mut_slice());

    let n_vecs = 1 << (log_size - LOG_N_LANES);
    let frac_at_row = |vec_row: usize| {
        let token = tokens_col.data[vec_row];
        let prf_output = prf_outputs_col.data[vec_row];
        let denom: PackedSecureField = lookup_elements.combine(&[token, prf_output]);
        (PackedSecureField::one(), denom)
    };

    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(frac_at_row));

    logup_gen.finalize_last()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon2_permutation() {
        let mut state = [BaseField::zero(); N_STATE];
        state[0] = BaseField::one();
        let original = state;
        poseidon2_permutation(&mut state);
        assert_ne!(state, original);
    }

    #[test]
    fn test_prf_deterministic() {
        let secret_key = [BaseField::from(42); SECRET_KEY_SIZE];
        let token = BaseField::from(12345);
        let output1 = compute_prf(&secret_key, token);
        let output2 = compute_prf(&secret_key, token);
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_prf_different_inputs() {
        let secret_key = [BaseField::from(42); SECRET_KEY_SIZE];
        let token1 = BaseField::from(12345);
        let token2 = BaseField::from(12346);
        let output1 = compute_prf(&secret_key, token1);
        let output2 = compute_prf(&secret_key, token2);
        assert_ne!(output1, output2);
    }

    #[test]
    fn test_is_green_distribution() {
        let secret_key = [BaseField::from(42); SECRET_KEY_SIZE];
        let mut green_count = 0;
        let total = 1000;
        for i in 0..total {
            let token = BaseField::from(i);
            let prf_output = compute_prf(&secret_key, token);
            if is_green_token(prf_output) {
                green_count += 1;
            }
        }
        assert!(green_count > 400 && green_count < 600);
    }

    #[test]
    fn test_prf_trace_column_count() {
        let log_size: u32 = 8;
        let n_rows = 1usize << log_size;
        let secret_key = [BaseField::from(42); SECRET_KEY_SIZE];
        let tokens: Vec<BaseField> = (0..n_rows).map(|i| BaseField::from(i as u32)).collect();

        let (trace, _) = gen_prf_trace(log_size, &tokens, &secret_key);
        assert_eq!(trace.len(), N_COLUMNS_PER_INSTANCE);
    }
}

