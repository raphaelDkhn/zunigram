//! Verifier for unigram watermark detection proofs.
//!
//! Validates STARK proofs of unigram watermark detection. Checks that the proof
//! is valid and the claimed green count meets the detection threshold.

use air::{
    CombinedTraceSizes, PublicInputElements, RangeCheckElements, UnigramComponent, UnigramEval,
    UnigramProof, PrfComponent, PrfEval, PrfLookupElements,
    compute_public_inputs_logup_sum, preprocessed_column_ids, N_COLUMNS_PER_INSTANCE,
};
use num_traits::Zero;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify;
use stwo_constraint_framework::TraceLocationAllocator;
use thiserror::Error;

/// Verification error type.
#[derive(Debug, Error)]
pub enum VerificationError {
    #[error("STARK verification failed: {0}")]
    StarkError(#[from] stwo::core::verifier::VerificationError),
    
    #[error("LogUp claimed sums don't cancel: PRF={prf:?}, Unigram={unigram:?}, PublicInputs={public:?}")]
    LogupMismatch {
        prf: SecureField,
        unigram: SecureField,
        public: SecureField,
    },
    
    #[error("Threshold {threshold} exceeds green count {green_count}")]
    ThresholdNotMet { threshold: u32, green_count: u32 },
}

/// Verifier for unigram watermark detection proofs.
/// 
/// # Example
/// ```ignore
/// Verifier::new().verify(proof)?;
/// ```
#[derive(Clone, Default)]
pub struct Verifier {
    pcs_config_override: Option<PcsConfig>,
}

impl Verifier {
    /// Create a new Verifier with default configuration.
    pub fn new() -> Self {
        Self {
            pcs_config_override: None,
        }
    }

    /// Override the PCS configuration from the proof.
    pub fn with_pcs_config(mut self, config: PcsConfig) -> Self {
        self.pcs_config_override = Some(config);
        self
    }

    /// Verify a unigram watermark detection proof.
    ///
    /// # Returns
    /// Ok(()) if valid, or an error describing why verification failed.
    pub fn verify(
        &self,
        proof: UnigramProof<Blake2sMerkleHasher>,
    ) -> Result<(), VerificationError> {
        let pcs_config = self.pcs_config_override.unwrap_or(proof.pcs_config);
        let log_size = proof.public_inputs.log_n_rows;
        let num_tokens = proof.public_inputs.num_tokens;
        let threshold = proof.public_inputs.threshold;

        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

        let sizes = CombinedTraceSizes::new(log_size, N_COLUMNS_PER_INSTANCE);

        // Phase 0: Preprocessed trace
        commitment_scheme.commit(
            proof.stark_proof.commitments[0],
            &sizes.preprocessed,
            channel,
        );

        // Mix public inputs before drawing lookup elements
        proof.public_inputs.mix_into(channel);

        // Phase 1: Main traces
        let main_sizes = [sizes.prf_main.clone(), sizes.unigram_main.clone()].concat();
        commitment_scheme.commit(proof.stark_proof.commitments[1], &main_sizes, channel);

        // Phase 2: Draw lookup elements
        let lookup_elements = PrfLookupElements::draw(channel);
        let public_input_elements = PublicInputElements::draw(channel);
        let range_check_elements = RangeCheckElements::draw(channel);

        // Phase 3: Interaction traces
        let interaction_sizes = [
            sizes.prf_interaction.clone(),
            sizes.unigram_interaction.clone(),
        ]
        .concat();
        commitment_scheme.commit(
            proof.stark_proof.commitments[2],
            &interaction_sizes,
            channel,
        );

        // Create components
        let mut trace_alloc =
            TraceLocationAllocator::new_with_preprocessed_columns(&preprocessed_column_ids());

        let prf_component = PrfComponent::new(
            &mut trace_alloc,
            PrfEval {
                log_n_rows: log_size,
                lookup_elements: lookup_elements.clone(),
                claimed_sum: proof.prf_claimed_sum,
            },
            proof.prf_claimed_sum,
        );

        let unigram_component = UnigramComponent::new(
            &mut trace_alloc,
            UnigramEval::new(
                log_size,
                num_tokens,
                threshold,
                lookup_elements,
                public_input_elements.clone(),
                range_check_elements,
                proof.unigram_claimed_sum,
            ),
            proof.unigram_claimed_sum,
        );

        // Verify STARK proof
        verify(
            &[&prf_component, &unigram_component],
            channel,
            commitment_scheme,
            proof.stark_proof,
        )?;

        // Verify LogUp soundness
        let public_logup_sum =
            compute_public_inputs_logup_sum(&proof.public_inputs, &public_input_elements);

        let total_claimed_sum =
            proof.prf_claimed_sum + proof.unigram_claimed_sum + public_logup_sum;
        
        if total_claimed_sum != SecureField::zero() {
            return Err(VerificationError::LogupMismatch {
                prf: proof.prf_claimed_sum,
                unigram: proof.unigram_claimed_sum,
                public: public_logup_sum,
            });
        }

        // Verify threshold
        if proof.public_inputs.green_count.0 < threshold {
            return Err(VerificationError::ThresholdNotMet {
                threshold,
                green_count: proof.public_inputs.green_count.0,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_creation() {
        let verifier = Verifier::new();
        assert!(verifier.pcs_config_override.is_none());
    }
}
