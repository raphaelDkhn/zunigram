//! Unigram component for proving watermark detection.
//!
//! This module implements the unigram watermark detection AIR. It verifies
//! that tokens are correctly classified as green/red based on PRF output,
//! and accumulates the green token count with public input binding via LogUp.

pub mod component;
pub mod table;
pub mod witness;

pub use component::*;
pub use table::*;
pub use witness::*;

