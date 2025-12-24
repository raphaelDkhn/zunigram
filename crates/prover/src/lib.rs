//! Prover for unigram LLM-watermark detection proofs.
//!
//! Generates STARK proofs that a token sequence contains a valid unigram watermark.
//! The proof proves correct PRF computation, green token classification, and green count accumulation
//! without revealing the secret key.

use air::{
    PublicInputElements, RangeCheckElements, UnigramComponent, UnigramEval,
    PrfComponent, PrfEval, PrfLookupElements,
    gen_prf_trace, gen_prf_interaction_trace, gen_unigram_trace, gen_unigram_interaction_trace,
    gen_preprocessed_trace, preprocessed_column_ids, create_public_inputs,
    LOG_EXPAND,
};
pub use air::UnigramProof;
use common::{UnigramInput, min_log_size_for_tokens};
use stwo::core::channel::Blake2sChannel;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{CommitmentSchemeProver, prove};
use stwo_constraint_framework::TraceLocationAllocator;
use thiserror::Error;

/// Error type for proof generation failures.
#[derive(Debug, Error)]
pub enum ProofError {
    #[error("Cannot prove with zero tokens")]
    EmptyInput,
    
    #[error("Proof generation failed: {0}")]
    ProvingFailed(String),
}

/// Builder for generating unigram watermark detection proofs.
/// 
/// # Example
/// ```ignore
/// let input = UnigramInput::new(tokens, secret_key);
/// let proof = Prover::new().prove(&input, 55)?;
/// ```
#[derive(Clone)]
pub struct Prover {
    pcs_config: PcsConfig,
}

impl Default for Prover {
    fn default() -> Self {
        Self::new()
    }
}

impl Prover {
    /// Create a new Prover with default configuration.
    pub fn new() -> Self {
        Self {
            pcs_config: PcsConfig::default(),
        }
    }

    /// Set the Polynomial Commitment Scheme configuration.
    pub fn with_pcs_config(mut self, config: PcsConfig) -> Self {
        self.pcs_config = config;
        self
    }

    /// Generate a proof that the token sequence contains a watermark.
    ///
    /// # Arguments
    /// * `input` - Token sequence and secret key
    /// * `threshold` - Minimum green count for detection
    ///
    /// # Returns
    /// A proof that can be verified to confirm watermark detection.
    pub fn prove(
        &self,
        input: &UnigramInput,
        threshold: u32,
    ) -> Result<UnigramProof<Blake2sMerkleHasher>, ProofError> {
        let num_tokens = input.num_tokens();
        if num_tokens == 0 {
            return Err(ProofError::EmptyInput);
        }

        let log_size = min_log_size_for_tokens(num_tokens);
        let pcs_config = self.pcs_config;

        // Precompute twiddles
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(log_size + LOG_EXPAND + pcs_config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        let channel = &mut Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(pcs_config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();

        // Phase 0: Preprocessed trace
        let preprocessed_trace = gen_preprocessed_trace(log_size, num_tokens);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(preprocessed_trace);
        tree_builder.commit(channel);

        // Create and mix public inputs
        let public_inputs = create_public_inputs(input, threshold);
        public_inputs.mix_into(channel);

        // Phase 1: Main traces
        let (prf_trace, prf_lookup_data) =
            gen_prf_trace(log_size, input.tokens(), input.secret_key().elements());
        let (unigram_trace, unigram_lookup_data) = gen_unigram_trace(log_size, input);

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(prf_trace);
        tree_builder.extend_evals(unigram_trace);
        tree_builder.commit(channel);

        // Phase 2: Draw lookup elements
        let lookup_elements = PrfLookupElements::draw(channel);
        let public_input_elements = PublicInputElements::draw(channel);
        let range_check_elements = RangeCheckElements::draw(channel);

        // Phase 3: Interaction traces
        let (prf_interaction_trace, prf_claimed_sum) =
            gen_prf_interaction_trace(log_size, &prf_lookup_data, &lookup_elements);

        let (unigram_interaction_trace, unigram_claimed_sum) = gen_unigram_interaction_trace(
            log_size,
            num_tokens,
            &unigram_lookup_data,
            &lookup_elements,
            &public_input_elements,
            &range_check_elements,
        );

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(prf_interaction_trace);
        tree_builder.extend_evals(unigram_interaction_trace);
        tree_builder.commit(channel);

        // Create components
        let mut trace_alloc =
            TraceLocationAllocator::new_with_preprocessed_columns(&preprocessed_column_ids());

        let prf_component = PrfComponent::new(
            &mut trace_alloc,
            PrfEval {
                log_n_rows: log_size,
                lookup_elements: lookup_elements.clone(),
                claimed_sum: prf_claimed_sum,
            },
            prf_claimed_sum,
        );

        let unigram_component = UnigramComponent::new(
            &mut trace_alloc,
            UnigramEval::new(
                log_size,
                num_tokens,
                threshold,
                lookup_elements,
                public_input_elements,
                range_check_elements,
                unigram_claimed_sum,
            ),
            unigram_claimed_sum,
        );

        // Generate STARK proof
        let stark_proof = prove(
            &[&prf_component, &unigram_component],
            channel,
            commitment_scheme,
        )
        .map_err(|e| ProofError::ProvingFailed(format!("{:?}", e)))?;

        Ok(UnigramProof {
            public_inputs,
            stark_proof,
            pcs_config,
            prf_claimed_sum,
            unigram_claimed_sum,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use common::M31;

    #[test]
    fn test_prover_creation() {
        let prover = Prover::default();
        assert!(prover.pcs_config.fri_config.log_blowup_factor > 0);
    }

    #[test]
    fn test_empty_input_rejected() {
        let input = UnigramInput::new(vec![], [M31::from(0); 8]);
        let result = Prover::default().prove(&input, 0);
        assert!(matches!(result, Err(ProofError::EmptyInput)));
    }
}
