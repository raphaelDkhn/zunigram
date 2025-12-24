//! PRF component for proving Poseidon2 hash computation.
//!
//! This module implements the PRF used for token
//! classification in unigram watermark detection. It uses Poseidon2
//! to compute H(secret_key, token).

pub mod component;
pub mod table;
pub mod witness;

pub use component::*;
pub use table::*;
pub use witness::*;

