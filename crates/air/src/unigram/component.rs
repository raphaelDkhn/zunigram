//! Unigram AIR component for watermark detection.
//!
//! Defines constraints for proving correct token classification and green count
//! accumulation. Uses LogUp for PRF lookup, public input binding, and range checks.

use common::M31;
use num_traits::One;
use stwo::core::channel::Channel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::proof::StarkProof;
use stwo::core::vcs::MerkleHasher;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry, ORIGINAL_TRACE_IDX,
};

use crate::prf::PrfLookupElements;

// ============================================================================
// LogUp Relations
// ============================================================================

// LogUp relation for binding public inputs (tokens and green_count).
relation!(PublicInputElements, 1);

// LogUp relation for 8-bit range check lookups.
relation!(RangeCheckElements, 1);

// ============================================================================
// Constants
// ============================================================================

/// Half of M31 prime for green/red classification threshold.
/// is_green = 1 if prf_output < HALF_PRIME, else 0.
pub const HALF_PRIME: u32 = 1073741823;

// ============================================================================
// Public Inputs
// ============================================================================

/// Public inputs for unigram watermark proof.
#[derive(Clone, Debug)]
pub struct PublicInputs {
    /// Log of trace size.
    pub log_n_rows: u32,
    /// Number of actual tokens (not padding).
    pub num_tokens: usize,
    /// Detection threshold.
    pub threshold: u32,
    /// Token sequence (public inputs).
    pub tokens: Vec<M31>,
    /// Final green count (public output).
    pub green_count: M31,
}

impl PublicInputs {
    /// Create new public inputs.
    pub fn new(
        log_n_rows: u32,
        num_tokens: usize,
        threshold: u32,
        tokens: Vec<M31>,
        green_count: M31,
    ) -> Self {
        Self {
            log_n_rows,
            num_tokens,
            threshold,
            tokens,
            green_count,
        }
    }

    /// Mix public inputs into Fiat-Shamir channel.
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_n_rows as u64);
        channel.mix_u64(self.num_tokens as u64);
        channel.mix_u64(self.threshold as u64);
        for token in &self.tokens {
            channel.mix_u64(token.0 as u64);
        }
        channel.mix_u64(self.green_count.0 as u64);
    }
}

// ============================================================================
// Preprocessed Column IDs
// ============================================================================

/// Preprocessed column identifier wrapper.
#[derive(Clone)]
pub struct UnigramPreprocessedColumn {
    pub name: String,
}

impl UnigramPreprocessedColumn {
    pub const fn new(name: String) -> Self {
        Self { name }
    }

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!("unigram_{}", self.name).into(),
        }
    }
}

/// Column ID for is_first indicator (1 at row 0).
pub fn is_first_col_id() -> PreProcessedColumnId {
    UnigramPreprocessedColumn::new("is_first".to_string()).id()
}

/// Column ID for is_last_token indicator (1 at row num_tokens-1).
pub fn is_last_token_col_id() -> PreProcessedColumnId {
    UnigramPreprocessedColumn::new("is_last_token".to_string()).id()
}

/// Column ID for range check table (values 0-255 repeated).
pub fn range_check_col_id() -> PreProcessedColumnId {
    UnigramPreprocessedColumn::new("range_check".to_string()).id()
}

/// Get all preprocessed column IDs.
pub fn preprocessed_column_ids() -> Vec<PreProcessedColumnId> {
    vec![is_first_col_id(), is_last_token_col_id(), range_check_col_id()]
}

// ============================================================================
// Unigram Component
// ============================================================================

/// Unigram AIR component type alias.
pub type UnigramComponent = FrameworkComponent<UnigramEval>;

/// Evaluator for unigram watermark detection AIR.
#[derive(Clone)]
pub struct UnigramEval {
    /// Log of number of rows.
    pub log_n_rows: u32,
    /// Number of tokens.
    pub num_tokens: usize,
    /// Detection threshold.
    pub threshold: u32,
    /// Lookup elements for PRF connection.
    pub lookup_elements: PrfLookupElements,
    /// Lookup elements for public input binding.
    pub public_input_elements: PublicInputElements,
    /// Lookup elements for range checks.
    pub range_check_elements: RangeCheckElements,
    /// Claimed sum for LogUp verification.
    pub claimed_sum: SecureField,
}

impl UnigramEval {
    /// Create new evaluator with all parameters.
    pub fn new(
        log_n_rows: u32,
        num_tokens: usize,
        threshold: u32,
        lookup_elements: PrfLookupElements,
        public_input_elements: PublicInputElements,
        range_check_elements: RangeCheckElements,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            log_n_rows,
            num_tokens,
            threshold,
            lookup_elements,
            public_input_elements,
            range_check_elements,
            claimed_sum,
        }
    }

    /// Number of main trace columns.
    pub fn n_main_trace_columns() -> usize {
        11 // token, prf_output, is_green, is_padding, diff, limb0-3, green_count, multiplicity
    }

    /// Number of preprocessed columns.
    pub fn n_preprocessed_columns() -> usize {
        3 // is_first, is_last_token, range_check
    }
}

impl FrameworkEval for UnigramEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // =====================
        // PREPROCESSED COLUMNS
        // =====================
        let is_first = eval.get_preprocessed_column(is_first_col_id());
        let is_last_token = eval.get_preprocessed_column(is_last_token_col_id());
        let range_check = eval.get_preprocessed_column(range_check_col_id());

        // =====================
        // MAIN TRACE COLUMNS
        // =====================
        let token = eval.next_trace_mask();
        let prf_output = eval.next_trace_mask();
        let is_green = eval.next_trace_mask();
        let is_padding = eval.next_trace_mask();
        let diff = eval.next_trace_mask();
        let limb0 = eval.next_trace_mask();
        let limb1 = eval.next_trace_mask();
        let limb2 = eval.next_trace_mask();
        let limb3 = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();
        let [green_count_prev, green_count] =
            eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);

        // =====================
        // BOOLEAN CONSTRAINTS
        // =====================
        eval.add_constraint(is_green.clone() * (is_green.clone() - E::F::one()));
        eval.add_constraint(is_padding.clone() * (is_padding.clone() - E::F::one()));

        // =====================
        // ACCUMULATOR CONSTRAINTS
        // =====================
        let is_not_first = E::F::one() - is_first.clone();
        let is_actual = E::F::one() - is_padding.clone();

        // Non-first actual rows: green_count = green_count_prev + is_green
        eval.add_constraint(
            is_not_first.clone() * is_actual.clone() 
                * (green_count.clone() - green_count_prev.clone() - is_green.clone()),
        );
        // First actual row: green_count = is_green
        eval.add_constraint(
            is_first.clone() * is_actual.clone() 
                * (green_count.clone() - is_green.clone())
        );
        // Padding rows: green_count stays constant
        eval.add_constraint(
            is_padding.clone() * (green_count.clone() - green_count_prev.clone())
        );

        // =====================
        // RANGE CHECK CONSTRAINTS
        // =====================
        let half_prime = E::F::from(M31::from(HALF_PRIME));
        let two_pow_8 = E::F::from(M31::from(256u32));
        let two_pow_16 = E::F::from(M31::from(65536u32));
        let two_pow_24 = E::F::from(M31::from(16777216u32));

        // diff = is_green*(HALF_PRIME-1-prf_output) + (1-is_green)*(prf_output-HALF_PRIME)
        let expected_diff = is_green.clone() * (half_prime.clone() - E::F::one() - prf_output.clone())
            + (E::F::one() - is_green.clone()) * (prf_output.clone() - half_prime);
        eval.add_constraint(diff.clone() - expected_diff);

        // Limb recomposition: diff = limb0 + 256*limb1 + 65536*limb2 + 16777216*limb3
        let recomposed = limb0.clone()
            + limb1.clone() * two_pow_8
            + limb2.clone() * two_pow_16
            + limb3.clone() * two_pow_24;
        eval.add_constraint(diff - recomposed);

        // =====================
        // PRF LOOKUP
        // =====================
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::one(),
            &[token.clone(), prf_output],
        ));

        // =====================
        // PUBLIC INPUT BINDING
        // =====================
        let is_actual_token = E::F::one() - is_padding.clone();
        eval.add_to_relation(RelationEntry::new(
            &self.public_input_elements,
            -E::EF::from(is_actual_token),
            &[token],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.public_input_elements,
            -E::EF::from(is_last_token),
            &[green_count],
        ));

        // =====================
        // RANGE CHECK LOOKUPS
        // =====================
        eval.add_to_relation(RelationEntry::new(
            &self.range_check_elements,
            E::EF::one(),
            &[limb0],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.range_check_elements,
            E::EF::one(),
            &[limb1],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.range_check_elements,
            E::EF::one(),
            &[limb2],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.range_check_elements,
            E::EF::one(),
            &[limb3],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.range_check_elements,
            -E::EF::from(multiplicity),
            &[range_check],
        ));

        eval.finalize_logup();
        eval
    }
}

// ============================================================================
// Proof Structure
// ============================================================================

/// Complete proof for unigram watermark verification.
#[derive(Clone)]
pub struct UnigramProof<H: MerkleHasher> {
    /// Public inputs bound via LogUp.
    pub public_inputs: PublicInputs,
    /// STARK proof.
    pub stark_proof: StarkProof<H>,
    /// PCS configuration.
    pub pcs_config: PcsConfig,
    /// PRF LogUp claimed sum.
    pub prf_claimed_sum: SecureField,
    /// Unigram LogUp claimed sum.
    pub unigram_claimed_sum: SecureField,
}

// ============================================================================
// Trace Sizes
// ============================================================================

/// Compute trace log degree bounds for Unigram component.
pub fn compute_unigram_trace_sizes(log_size: u32) -> [Vec<u32>; 2] {
    let preprocessed = vec![log_size; UnigramEval::n_preprocessed_columns()];
    let main = vec![log_size; UnigramEval::n_main_trace_columns()];
    [preprocessed, main]
}

/// Combined trace sizes for both PRF and Unigram components.
pub struct CombinedTraceSizes {
    pub preprocessed: Vec<u32>,
    pub prf_main: Vec<u32>,
    pub unigram_main: Vec<u32>,
    pub prf_interaction: Vec<u32>,
    pub unigram_interaction: Vec<u32>,
}

impl CombinedTraceSizes {
    /// Create combined trace sizes for given log size.
    pub fn new(log_size: u32, prf_n_cols: usize) -> Self {
        Self {
            preprocessed: vec![log_size; UnigramEval::n_preprocessed_columns()],
            prf_main: vec![log_size; prf_n_cols],
            unigram_main: vec![log_size; UnigramEval::n_main_trace_columns()],
            prf_interaction: vec![log_size; 4],
            unigram_interaction: vec![log_size; 32],
        }
    }

    /// Get all sizes as vectors for commitment scheme.
    pub fn as_vecs(&self) -> Vec<Vec<u32>> {
        vec![
            self.preprocessed.clone(),
            [self.prf_main.clone(), self.unigram_main.clone()].concat(),
            [self.prf_interaction.clone(), self.unigram_interaction.clone()].concat(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessed_column_ids() {
        let ids = preprocessed_column_ids();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_trace_sizes() {
        let sizes = compute_unigram_trace_sizes(10);
        assert_eq!(sizes[0].len(), 3);
        assert!(sizes[1].len() > 0);
    }

    #[test]
    fn test_public_inputs_creation() {
        let tokens = vec![M31::from(1), M31::from(2), M31::from(3)];
        let pi = PublicInputs::new(10, 3, 2, tokens, M31::from(2));
        assert_eq!(pi.num_tokens, 3);
        assert_eq!(pi.threshold, 2);
    }

    #[test]
    fn test_column_counts() {
        assert_eq!(UnigramEval::n_main_trace_columns(), 11);
        assert_eq!(UnigramEval::n_preprocessed_columns(), 3);
    }

    #[test]
    fn test_half_prime() {
        assert_eq!(HALF_PRIME, 1073741823);
    }
}

