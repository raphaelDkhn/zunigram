//! Unigram lookup table definitions.
//!
//! Defines data structures for LogUp lookups in the Unigram component:
//! PRF consumption, public input binding, and range check tables.

use num_traits::Zero;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::Column;
use stwo_constraint_framework::Relation;

use super::component::{PublicInputElements, PublicInputs};

/// Lookup data for Unigram component.
/// Contains columns needed for LogUp interaction trace generation.
#[derive(Clone)]
pub struct UnigramLookupData {
    /// Token values.
    pub tokens: BaseColumn,
    /// PRF output values.
    pub prf_outputs: BaseColumn,
    /// Is padding flag.
    pub is_padding: BaseColumn,
    /// Green count values.
    pub green_counts: BaseColumn,
    /// Limb 0 values (for range check).
    pub limb0: BaseColumn,
    /// Limb 1 values (for range check).
    pub limb1: BaseColumn,
    /// Limb 2 values (for range check).
    pub limb2: BaseColumn,
    /// Limb 3 values (for range check).
    pub limb3: BaseColumn,
    /// Range check multiplicity.
    pub multiplicity: BaseColumn,
}

impl UnigramLookupData {
    /// Create new lookup data with pre-allocated columns.
    pub fn new(n_rows: usize) -> Self {
        Self {
            tokens: BaseColumn::zeros(n_rows),
            prf_outputs: BaseColumn::zeros(n_rows),
            is_padding: BaseColumn::zeros(n_rows),
            green_counts: BaseColumn::zeros(n_rows),
            limb0: BaseColumn::zeros(n_rows),
            limb1: BaseColumn::zeros(n_rows),
            limb2: BaseColumn::zeros(n_rows),
            limb3: BaseColumn::zeros(n_rows),
            multiplicity: BaseColumn::zeros(n_rows),
        }
    }

    /// Get the number of rows.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.len() == 0
    }
}

/// Compute positive LogUp sum for public inputs (verifier side).
/// Adds +1 multiplicity for each public value to cancel trace's -1.
pub fn compute_public_inputs_logup_sum(
    public_inputs: &PublicInputs,
    public_input_elements: &PublicInputElements,
) -> SecureField {
    let mut sum = SecureField::zero();

    // Add +1 for each actual token
    for token in &public_inputs.tokens {
        let denom: SecureField = public_input_elements.combine(&[*token]);
        sum += denom.inverse();
    }

    // Add +1 for the final green_count
    let denom: SecureField = public_input_elements.combine(&[public_inputs.green_count]);
    sum += denom.inverse();

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_data_creation() {
        let data = UnigramLookupData::new(256);
        assert_eq!(data.len(), 256);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_empty_lookup_data() {
        let data = UnigramLookupData::new(0);
        assert!(data.is_empty());
    }
}

