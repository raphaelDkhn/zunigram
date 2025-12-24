//! Unigram witness/trace generation.
//!
//! Generates execution traces for token classification and green count accumulation.

use common::{UnigramInput, min_log_size_for_tokens};
use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::ColumnVec;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES, PackedM31};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

use crate::prf::{compute_prf, is_green_token, PrfLookupElements};
use super::component::{PublicInputElements, PublicInputs, RangeCheckElements, HALF_PRIME};
use super::table::UnigramLookupData;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute diff value for is_green range check constraint.
/// If is_green is correct, diff will be non-negative (in [0, HALF_PRIME-1]).
fn compute_diff(prf_output: u32, is_green: bool) -> u32 {
    if is_green {
        HALF_PRIME - 1 - prf_output
    } else {
        prf_output - HALF_PRIME
    }
}

/// Decompose a 32-bit value into 4 x 8-bit limbs.
fn decompose_to_limbs(value: u32) -> [u32; 4] {
    [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]
}

// ============================================================================
// Trace Generation
// ============================================================================

/// Generate main trace for Unigram component.
/// Produces columns for token classification, accumulator, and range checks.
pub fn gen_unigram_trace(
    log_size: u32,
    input: &UnigramInput,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    UnigramLookupData,
) {
    let trace_len = 1 << log_size;
    let num_tokens = input.num_tokens();
    let secret_key = input.secret_key().elements();

    // Initialize main trace columns
    let mut token_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut prf_output_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut is_green_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut is_padding_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut diff_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut limb0_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut limb1_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut limb2_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut limb3_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut green_count_col = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let mut multiplicity_col = Col::<SimdBackend, BaseField>::zeros(trace_len);

    // Initialize lookup data columns
    let mut lookup_data = UnigramLookupData::new(trace_len);

    // Count limb value occurrences for range check multiplicity
    let mut limb_counts = [0u32; 256];
    let mut running_green_count = 0u32;

    // Fill actual token rows
    for (row, &token) in input.tokens().iter().enumerate() {
        if row >= trace_len {
            break;
        }

        token_col.set(row, token);
        let prf_output = compute_prf(secret_key, token);
        prf_output_col.set(row, prf_output);

        let is_green_flag = is_green_token(prf_output);
        if is_green_flag {
            running_green_count += 1;
        }
        is_green_col.set(row, BaseField::from(is_green_flag as u32));

        let diff = compute_diff(prf_output.0, is_green_flag);
        let limbs = decompose_to_limbs(diff);
        diff_col.set(row, BaseField::from(diff));
        limb0_col.set(row, BaseField::from(limbs[0]));
        limb1_col.set(row, BaseField::from(limbs[1]));
        limb2_col.set(row, BaseField::from(limbs[2]));
        limb3_col.set(row, BaseField::from(limbs[3]));

        // Count limb values
        limb_counts[limbs[0] as usize] += 1;
        limb_counts[limbs[1] as usize] += 1;
        limb_counts[limbs[2] as usize] += 1;
        limb_counts[limbs[3] as usize] += 1;

        is_padding_col.set(row, BaseField::zero());
        green_count_col.set(row, BaseField::from(running_green_count));
    }

    // Fill padding rows
    let padding_prf = compute_prf(secret_key, BaseField::zero());
    let padding_is_green = is_green_token(padding_prf);
    let padding_diff = compute_diff(padding_prf.0, padding_is_green);
    let padding_limbs = decompose_to_limbs(padding_diff);
    let n_padding = trace_len - num_tokens;

    // Count padding limb values
    limb_counts[padding_limbs[0] as usize] += n_padding as u32;
    limb_counts[padding_limbs[1] as usize] += n_padding as u32;
    limb_counts[padding_limbs[2] as usize] += n_padding as u32;
    limb_counts[padding_limbs[3] as usize] += n_padding as u32;

    for row in num_tokens..trace_len {
        token_col.set(row, BaseField::zero());
        prf_output_col.set(row, padding_prf);
        is_green_col.set(row, BaseField::from(padding_is_green as u32));
        diff_col.set(row, BaseField::from(padding_diff));
        limb0_col.set(row, BaseField::from(padding_limbs[0]));
        limb1_col.set(row, BaseField::from(padding_limbs[1]));
        limb2_col.set(row, BaseField::from(padding_limbs[2]));
        limb3_col.set(row, BaseField::from(padding_limbs[3]));
        is_padding_col.set(row, BaseField::one());
        green_count_col.set(row, BaseField::from(running_green_count));
    }

    // Fill multiplicity column (only first occurrence of each value)
    for v in 0..trace_len.min(256) {
        multiplicity_col.set(v, BaseField::from(limb_counts[v]));
    }

    // Fill lookup data
    let n_vecs = 1 << (log_size - LOG_N_LANES);
    let lookup_results: Vec<_> = (0..n_vecs).into_par_iter().map(|vec_idx| {
        (
            vec_idx,
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                if row < num_tokens { input.tokens()[row] } else { BaseField::zero() }
            })),
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                let token = if row < num_tokens { input.tokens()[row] } else { BaseField::zero() };
                compute_prf(secret_key, token)
            })),
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                BaseField::from((row >= num_tokens) as u32)
            })),
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                let mut count = 0u32;
                for i in 0..=row.min(num_tokens.saturating_sub(1)) {
                    if i < num_tokens {
                        let token = input.tokens()[i];
                        let prf = compute_prf(secret_key, token);
                        if is_green_token(prf) { count += 1; }
                    }
                }
                BaseField::from(count)
            })),
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                let (prf, is_green_flag) = if row < num_tokens {
                    let token = input.tokens()[row];
                    let prf = compute_prf(secret_key, token);
                    (prf.0, is_green_token(prf))
                } else {
                    (padding_prf.0, padding_is_green)
                };
                let diff = compute_diff(prf, is_green_flag);
                BaseField::from(diff & 0xFF)
            })),
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                let (prf, is_green_flag) = if row < num_tokens {
                    let token = input.tokens()[row];
                    let prf = compute_prf(secret_key, token);
                    (prf.0, is_green_token(prf))
                } else {
                    (padding_prf.0, padding_is_green)
                };
                let diff = compute_diff(prf, is_green_flag);
                BaseField::from((diff >> 8) & 0xFF)
            })),
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                let (prf, is_green_flag) = if row < num_tokens {
                    let token = input.tokens()[row];
                    let prf = compute_prf(secret_key, token);
                    (prf.0, is_green_token(prf))
                } else {
                    (padding_prf.0, padding_is_green)
                };
                let diff = compute_diff(prf, is_green_flag);
                BaseField::from((diff >> 16) & 0xFF)
            })),
            PackedM31::from_array(std::array::from_fn(|lane| {
                let row = vec_idx * N_LANES + lane;
                let (prf, is_green_flag) = if row < num_tokens {
                    let token = input.tokens()[row];
                    let prf = compute_prf(secret_key, token);
                    (prf.0, is_green_token(prf))
                } else {
                    (padding_prf.0, padding_is_green)
                };
                let diff = compute_diff(prf, is_green_flag);
                BaseField::from((diff >> 24) & 0xFF)
            })),
        )
    }).collect();

    // Write lookup data sequentially
    for (vec_idx, tokens_val, prf_outputs_val, is_padding_val, green_counts_val, limb0_val, limb1_val, limb2_val, limb3_val) in lookup_results {
        lookup_data.tokens.data[vec_idx] = tokens_val;
        lookup_data.prf_outputs.data[vec_idx] = prf_outputs_val;
        lookup_data.is_padding.data[vec_idx] = is_padding_val;
        lookup_data.green_counts.data[vec_idx] = green_counts_val;
        lookup_data.limb0.data[vec_idx] = limb0_val;
        lookup_data.limb1.data[vec_idx] = limb1_val;
        lookup_data.limb2.data[vec_idx] = limb2_val;
        lookup_data.limb3.data[vec_idx] = limb3_val;
    }

    // Multiplicity lookup data
    for v in 0..trace_len.min(256) {
        lookup_data.multiplicity.set(v, BaseField::from(limb_counts[v]));
    }

    // Apply bit-reversal to all main trace columns
    [
        token_col.as_mut_slice(), prf_output_col.as_mut_slice(), is_green_col.as_mut_slice(),
        is_padding_col.as_mut_slice(), diff_col.as_mut_slice(), limb0_col.as_mut_slice(),
        limb1_col.as_mut_slice(), limb2_col.as_mut_slice(), limb3_col.as_mut_slice(),
        green_count_col.as_mut_slice(), multiplicity_col.as_mut_slice(),
    ].par_iter_mut().for_each(|slice| {
        bit_reverse_coset_to_circle_domain_order(*slice);
    });

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = vec![
        CircleEvaluation::new(domain, token_col),
        CircleEvaluation::new(domain, prf_output_col),
        CircleEvaluation::new(domain, is_green_col),
        CircleEvaluation::new(domain, is_padding_col),
        CircleEvaluation::new(domain, diff_col),
        CircleEvaluation::new(domain, limb0_col),
        CircleEvaluation::new(domain, limb1_col),
        CircleEvaluation::new(domain, limb2_col),
        CircleEvaluation::new(domain, limb3_col),
        CircleEvaluation::new(domain, multiplicity_col),
        CircleEvaluation::new(domain, green_count_col),
    ];

    (trace, lookup_data)
}

/// Generate interaction trace for Unigram component.
/// Produces LogUp columns for PRF, public inputs, and range checks.
pub fn gen_unigram_interaction_trace(
    log_size: u32,
    num_tokens: usize,
    lookup_data: &UnigramLookupData,
    lookup_elements: &PrfLookupElements,
    public_input_elements: &PublicInputElements,
    range_check_elements: &RangeCheckElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = unsafe { LogupTraceGenerator::uninitialized(log_size) };
    let trace_len = 1usize << log_size;
    let n_vecs = 1 << (log_size - LOG_N_LANES);

    // Clone and bit-reverse all lookup data columns
    let mut tokens = lookup_data.tokens.clone();
    let mut prf_outputs = lookup_data.prf_outputs.clone();
    let mut is_padding = lookup_data.is_padding.clone();
    let mut green_counts = lookup_data.green_counts.clone();
    let mut limb0 = lookup_data.limb0.clone();
    let mut limb1 = lookup_data.limb1.clone();
    let mut limb2 = lookup_data.limb2.clone();
    let mut limb3 = lookup_data.limb3.clone();
    let mut multiplicity = lookup_data.multiplicity.clone();

    // Bit-reverse lookup data columns
    [
        tokens.as_mut_slice(), prf_outputs.as_mut_slice(), is_padding.as_mut_slice(),
        green_counts.as_mut_slice(), limb0.as_mut_slice(), limb1.as_mut_slice(),
        limb2.as_mut_slice(), limb3.as_mut_slice(), multiplicity.as_mut_slice(),
    ].par_iter_mut().for_each(|slice| {
        bit_reverse_coset_to_circle_domain_order(*slice);
    });

    // Create is_last_token mask
    let mut is_last_token = BaseColumn::zeros(trace_len);
    if num_tokens > 0 {
        is_last_token.set(num_tokens - 1, BaseField::one());
    }
    bit_reverse_coset_to_circle_domain_order(&mut is_last_token.as_mut_slice());

    // Range check table
    let mut range_check_table = BaseColumn::zeros(trace_len);
    let n_vecs_table = 1 << (log_size - LOG_N_LANES);
    let table_data: Vec<_> = (0..n_vecs_table).into_par_iter().map(|vec_idx| {
        (0..N_LANES).map(move |lane| {
            let row = vec_idx * N_LANES + lane;
            if row < trace_len {
                Some((row, BaseField::from((row % 256) as u32)))
            } else {
                None
            }
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>().into_iter().flatten().flatten().collect();
    for (row, value) in table_data {
        range_check_table.set(row, value);
    }
    bit_reverse_coset_to_circle_domain_order(&mut range_check_table.as_mut_slice());

    // Column 1: PRF lookup
    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let token = tokens.data[vec_row];
        let prf_output = prf_outputs.data[vec_row];
        let denom: PackedSecureField = lookup_elements.combine(&[token, prf_output]);
        (-PackedSecureField::one(), denom)
    }));

    // Column 2: Token public input binding
    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let token = tokens.data[vec_row];
        let padding = is_padding.data[vec_row];
        let denom: PackedSecureField = public_input_elements.combine(&[token]);
        let is_actual = PackedSecureField::one() - PackedSecureField::from(padding);
        (-is_actual, denom)
    }));

    // Column 3: Green count binding
    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let green_count = green_counts.data[vec_row];
        let is_last = is_last_token.data[vec_row];
        let denom: PackedSecureField = public_input_elements.combine(&[green_count]);
        (-PackedSecureField::from(is_last), denom)
    }));

    // Columns 4-7: Range check limbs
    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let limb = limb0.data[vec_row];
        let denom: PackedSecureField = range_check_elements.combine(&[limb]);
        (PackedSecureField::one(), denom)
    }));

    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let limb = limb1.data[vec_row];
        let denom: PackedSecureField = range_check_elements.combine(&[limb]);
        (PackedSecureField::one(), denom)
    }));

    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let limb = limb2.data[vec_row];
        let denom: PackedSecureField = range_check_elements.combine(&[limb]);
        (PackedSecureField::one(), denom)
    }));

    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let limb = limb3.data[vec_row];
        let denom: PackedSecureField = range_check_elements.combine(&[limb]);
        (PackedSecureField::one(), denom)
    }));

    // Column 8: Range check table
    logup_gen.col_from_par_iter((0..n_vecs).into_par_iter().map(|vec_row| {
        let value = range_check_table.data[vec_row];
        let mult = multiplicity.data[vec_row];
        let denom: PackedSecureField = range_check_elements.combine(&[value]);
        (-PackedSecureField::from(mult), denom)
    }));

    logup_gen.finalize_last()
}

// ============================================================================
// Public Interface
// ============================================================================

/// Count green tokens in input.
pub fn count_green_tokens(input: &UnigramInput) -> u32 {
    let secret_key = input.secret_key().elements();
    input.tokens().par_iter()
        .filter(|&&token| is_green_token(compute_prf(secret_key, token)))
        .count() as u32
}

/// Create public inputs from input data.
pub fn create_public_inputs(input: &UnigramInput, threshold: u32) -> PublicInputs {
    let num_tokens = input.num_tokens();
    let log_size = min_log_size_for_tokens(num_tokens);
    let green_count = count_green_tokens(input);

    PublicInputs::new(
        log_size,
        num_tokens,
        threshold,
        input.tokens().to_vec(),
        BaseField::from(green_count),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::SECRET_KEY_SIZE;
    use stwo::core::fields::m31::M31;

    #[test]
    fn test_unigram_trace_column_count() {
        let tokens = vec![M31::from(1), M31::from(2), M31::from(3)];
        let secret_key = [M31::from(42); SECRET_KEY_SIZE];
        let input = UnigramInput::new(tokens, secret_key);
        let (trace, _) = gen_unigram_trace(4, &input);
        assert_eq!(trace.len(), 11);
    }

    #[test]
    fn test_count_green_tokens() {
        let tokens: Vec<M31> = (0..100).map(|i| M31::from(i)).collect();
        let secret_key = [M31::from(12345); SECRET_KEY_SIZE];
        let input = UnigramInput::new(tokens, secret_key);
        let count = count_green_tokens(&input);
        assert!(count > 30 && count < 70);
    }

    #[test]
    fn test_diff_computation() {
        let prf_green = HALF_PRIME - 100;
        let diff_green = compute_diff(prf_green, true);
        assert_eq!(diff_green, HALF_PRIME - 1 - prf_green);

        let prf_red = HALF_PRIME + 100;
        let diff_red = compute_diff(prf_red, false);
        assert_eq!(diff_red, prf_red - HALF_PRIME);
    }

    #[test]
    fn test_limb_decomposition() {
        let value = 0x12345678u32;
        let limbs = decompose_to_limbs(value);
        assert_eq!(limbs[0], 0x78);
        assert_eq!(limbs[1], 0x56);
        assert_eq!(limbs[2], 0x34);
        assert_eq!(limbs[3], 0x12);
        let recomposed = limbs[0] + limbs[1] * 256 + limbs[2] * 65536 + limbs[3] * 16777216;
        assert_eq!(recomposed, value);
    }
}

