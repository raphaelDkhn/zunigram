//! PRF AIR component for Poseidon2 hash computation.
//!
//! Defines constraints for proving correct computation of PRF(secret_key, token).

use std::ops::{Add, AddAssign, Mul, Sub};

use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry,
};

use common::SECRET_KEY_SIZE;

// ============================================================================
// Poseidon2 Parameters
// ============================================================================

/// State size for Poseidon2 (16 M31 elements).
pub const N_STATE: usize = 16;

/// Rate of the sponge (elements absorbed per permutation).
pub const RATE: usize = 8;

/// Capacity of the sponge.
pub const CAPACITY: usize = N_STATE - RATE;

/// Number of partial rounds.
pub const N_PARTIAL_ROUNDS: usize = 14;

/// Number of full rounds on each side.
pub const N_HALF_FULL_ROUNDS: usize = 4;

/// Total full rounds.
pub const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;

/// Log of constraint degree bound for Poseidon2 (x^5 S-box).
pub const LOG_EXPAND: u32 = 2;

/// Minimum log size for trace.
pub const MIN_LOG_N_ROWS: u32 = 8;

/// Number of columns per PRF instance.
/// - Initial state: N_STATE (16)
/// - After each full round: N_STATE * N_HALF_FULL_ROUNDS (64)
/// - After each partial round: 1 * N_PARTIAL_ROUNDS (14)
/// - After final full rounds: N_STATE * N_HALF_FULL_ROUNDS (64)
pub const N_COLUMNS_PER_INSTANCE: usize =
    N_STATE + N_STATE * N_HALF_FULL_ROUNDS + N_PARTIAL_ROUNDS + N_STATE * N_HALF_FULL_ROUNDS;

// ============================================================================
// Round Constants
// ============================================================================

/// External round constants (for full rounds).
/// Derived from nothing-up-my-sleeve values.
pub const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 2 * N_HALF_FULL_ROUNDS] = {
    let mut consts = [[BaseField::from_u32_unchecked(0); N_STATE]; 2 * N_HALF_FULL_ROUNDS];
    let mut round = 0;
    while round < 2 * N_HALF_FULL_ROUNDS {
        let mut i = 0;
        while i < N_STATE {
            let val = ((round * 16 + i + 1) as u64 * 0x9e3779b9u64) % ((1u64 << 31) - 1);
            consts[round][i] = BaseField::from_u32_unchecked(val as u32);
            i += 1;
        }
        round += 1;
    }
    consts
};

/// Internal round constants (for partial rounds).
pub const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] = {
    let mut consts = [BaseField::from_u32_unchecked(0); N_PARTIAL_ROUNDS];
    let mut i = 0;
    while i < N_PARTIAL_ROUNDS {
        let val = ((i + 100) as u64 * 0x9e3779b9u64) % ((1u64 << 31) - 1);
        consts[i] = BaseField::from_u32_unchecked(val as u32);
        i += 1;
    }
    consts
};

// ============================================================================
// Poseidon2 Matrix Operations
// ============================================================================

/// Apply the 4x4 MDS matrix M4.
#[inline(always)]
pub fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t12.clone() + t12.clone() + t3.clone();
    let t5 = t02.clone() + t02.clone() + t2.clone();
    let t6 = t3.clone() + t5.clone();
    let t7 = t2.clone() + t4.clone();
    [t6, t5, t7, t4]
}

/// Apply the external round matrix (used in full rounds).
pub fn apply_external_round_matrix<F>(state: &mut [F; N_STATE])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Apply M4 to each 4-element chunk
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i].clone(),
            state[4 * i + 1].clone(),
            state[4 * i + 2].clone(),
            state[4 * i + 3].clone(),
        ]);
    }
    // Diffusion across chunks
    for j in 0..4 {
        let s =
            state[j].clone() + state[j + 4].clone() + state[j + 8].clone() + state[j + 12].clone();
        for i in 0..4 {
            state[4 * i + j] += s.clone();
        }
    }
}

/// Apply the internal round matrix (used in partial rounds).
pub fn apply_internal_round_matrix<F>(state: &mut [F; N_STATE])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);
    state.iter_mut().enumerate().for_each(|(i, s)| {
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

/// S-box: x^5.
#[inline(always)]
pub fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

// ============================================================================
// Lookup Relation
// ============================================================================

// LogUp relation for connecting PRF and Unigram components.
// Carries (token, prf_output) tuples.
relation!(PrfLookupElements, 2);

// ============================================================================
// PRF Component
// ============================================================================

/// PRF component type alias.
pub type PrfComponent = FrameworkComponent<PrfEval>;

/// Evaluator for PRF AIR constraints.
/// Verifies correct hash computation for each token in the trace.
#[derive(Clone)]
pub struct PrfEval {
    /// Log of number of rows in the trace.
    pub log_n_rows: u32,
    /// Lookup elements for connecting to Unigram component.
    pub lookup_elements: PrfLookupElements,
    /// Claimed sum for LogUp verification.
    pub claimed_sum: SecureField,
}

impl FrameworkEval for PrfEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_EXPAND
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_prf_constraints(&mut eval, &self.lookup_elements);
        eval
    }
}

/// Evaluate PRF constraints for a single row.
/// Reads initial state, constrains Poseidon2 permutation, adds lookup entry.
pub fn eval_prf_constraints<E: EvalAtRow>(eval: &mut E, lookup_elements: &PrfLookupElements) {
    // Read initial state from trace
    let mut state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

    // Store token (at position 8) for lookup
    let token = state[SECRET_KEY_SIZE].clone();

    // First half of full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] += EXTERNAL_ROUND_CONSTS[round][i];
        }
        apply_external_round_matrix(&mut state);
        state = std::array::from_fn(|i| pow5(state[i].clone()));
        state.iter_mut().for_each(|s| {
            let m = eval.next_trace_mask();
            eval.add_constraint(s.clone() - m.clone());
            *s = m;
        });
    }

    // Partial rounds
    for round in 0..N_PARTIAL_ROUNDS {
        state[0] += INTERNAL_ROUND_CONSTS[round];
        apply_internal_round_matrix(&mut state);
        state[0] = pow5(state[0].clone());
        let m = eval.next_trace_mask();
        eval.add_constraint(state[0].clone() - m.clone());
        state[0] = m;
    }

    // Second half of full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] += EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i];
        }
        apply_external_round_matrix(&mut state);
        state = std::array::from_fn(|i| pow5(state[i].clone()));
        state.iter_mut().for_each(|s| {
            let m = eval.next_trace_mask();
            eval.add_constraint(s.clone() - m.clone());
            *s = m;
        });
    }

    // Add lookup: PRF provides (token, prf_output)
    let prf_output = state[0].clone();
    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::one(),
        &[token, prf_output],
    ));

    eval.finalize_logup();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_count() {
        // 16 initial + 64 first full + 14 partial + 64 second full
        assert_eq!(N_COLUMNS_PER_INSTANCE, 158);
    }

    #[test]
    fn test_round_constants_initialized() {
        // Verify round constants are non-zero
        assert_ne!(EXTERNAL_ROUND_CONSTS[0][0], BaseField::from(0));
        assert_ne!(INTERNAL_ROUND_CONSTS[0], BaseField::from(0));
    }
}

