//! Preprocessed column generation.
//!
//! Generates static columns computed before proof generation:
//! - is_first: marks the first row
//! - is_last_token: marks the last token row
//! - range_check: lookup table values 0-255 (for limb range checks)

use stwo::core::ColumnVec;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES, PackedM31};
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

/// Preprocessed column identifier wrapper.
#[derive(Clone)]
pub struct PreprocessedColumn {
    pub name: String,
}

impl PreprocessedColumn {
    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!("unigram_{}", self.name).into(),
        }
    }
}

/// Get preprocessed column ID for is_first indicator.
pub fn is_first_col_id() -> PreProcessedColumnId {
    PreprocessedColumn { name: "is_first".to_string() }.id()
}

/// Get preprocessed column ID for is_last_token indicator.
pub fn is_last_token_col_id() -> PreProcessedColumnId {
    PreprocessedColumn { name: "is_last_token".to_string() }.id()
}

/// Get preprocessed column ID for range check table.
pub fn range_check_col_id() -> PreProcessedColumnId {
    PreprocessedColumn { name: "range_check".to_string() }.id()
}

/// Get all preprocessed column IDs.
pub fn preprocessed_column_ids() -> Vec<PreProcessedColumnId> {
    vec![is_first_col_id(), is_last_token_col_id(), range_check_col_id()]
}

/// Generate preprocessed trace columns.
/// 
/// Columns:
/// - is_first: 1 at row 0, 0 elsewhere
/// - is_last_token: 1 at row (num_tokens-1), 0 elsewhere  
/// - range_check: values 0-255 repeated to fill trace
pub fn gen_preprocessed_trace(
    log_size: u32,
    num_tokens: usize,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let trace_len = 1 << log_size;
    let n_vecs = 1 << (log_size - LOG_N_LANES);

    // is_first column: 1 at row 0
    let mut is_first = Col::<SimdBackend, BaseField>::zeros(trace_len);
    is_first.set(0, BaseField::from(1));

    // is_last_token column: 1 at row (num_tokens - 1)
    let mut is_last_token = Col::<SimdBackend, BaseField>::zeros(trace_len);
    if num_tokens > 0 {
        is_last_token.set(num_tokens - 1, BaseField::from(1));
    }

    // range_check column: values 0-255 repeated
    use rayon::prelude::*;
    let mut range_check = Col::<SimdBackend, BaseField>::zeros(trace_len);
    let range_check_data: Vec<_> = (0..n_vecs).into_par_iter().map(|vec_idx| {
        (vec_idx, PackedM31::from_array(std::array::from_fn(|lane| {
            let row = vec_idx * N_LANES + lane;
            BaseField::from((row % 256) as u32)
        })))
    }).collect();
    for (vec_idx, value) in range_check_data {
        range_check.data[vec_idx] = value;
    }

    // Apply bit-reversal for local-row constraint
    bit_reverse_coset_to_circle_domain_order(&mut is_first.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(&mut is_last_token.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(&mut range_check.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![
        CircleEvaluation::new(domain, is_first),
        CircleEvaluation::new(domain, is_last_token),
        CircleEvaluation::new(domain, range_check),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessed_trace_length() {
        let trace = gen_preprocessed_trace(8, 100);
        assert_eq!(trace.len(), 3);
    }

    #[test]
    fn test_column_ids() {
        let ids = preprocessed_column_ids();
        assert_eq!(ids.len(), 3);
    }
}

