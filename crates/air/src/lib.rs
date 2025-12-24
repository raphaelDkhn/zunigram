//! AIR components for unigram watermark detection.
//!
//! This crate provides the AIR for proving
//! unigram watermark detection using the Stwo prover. It includes two components:
//! - PRF component: Proves Poseidon2 hash computation H(secret_key, token)
//! - Unigram component: Proves token classification and green count accumulation

pub mod prf;
pub mod unigram;
pub mod preprocessed;

pub use prf::{
    PrfComponent, PrfEval, PrfLookupElements, PrfLookupData,
    gen_prf_trace, gen_prf_interaction_trace,
    compute_prf, is_green_token, poseidon2_permutation,
    N_STATE, N_COLUMNS_PER_INSTANCE, LOG_EXPAND,
    N_HALF_FULL_ROUNDS, N_PARTIAL_ROUNDS, RATE, CAPACITY,
    EXTERNAL_ROUND_CONSTS, INTERNAL_ROUND_CONSTS,
};

pub use unigram::{
    UnigramComponent, UnigramEval, UnigramLookupData,
    PublicInputs, PublicInputElements, RangeCheckElements,
    gen_unigram_trace, gen_unigram_interaction_trace,
    count_green_tokens, create_public_inputs, compute_public_inputs_logup_sum,
    UnigramProof, CombinedTraceSizes, compute_unigram_trace_sizes,
    HALF_PRIME,
};

pub use preprocessed::{
    gen_preprocessed_trace, preprocessed_column_ids,
    is_first_col_id, is_last_token_col_id, range_check_col_id,
};
