//! PRF lookup table definitions.
//!
//! Defines data structures for LogUp lookups between PRF and Unigram components.
//! The PRF provides (token, prf_output) tuples that Unigram consumes.

use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::Column;

/// Lookup data for PRF component.
/// Stores (token, prf_output) pairs for LogUp verification.
#[derive(Clone)]
pub struct PrfLookupData {
    /// Token values for each row.
    pub tokens: BaseColumn,
    /// PRF output values for each row.
    pub prf_outputs: BaseColumn,
}

impl PrfLookupData {
    /// Create new lookup data with pre-allocated columns.
    pub fn new(n_rows: usize) -> Self {
        Self {
            tokens: BaseColumn::zeros(n_rows),
            prf_outputs: BaseColumn::zeros(n_rows),
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

