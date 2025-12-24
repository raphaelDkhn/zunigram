use common::{M31, UnigramInput};
use air::{compute_prf, count_green_tokens, is_green_token, UnigramProof, HALF_PRIME};
use prover::Prover;
use verifier::Verifier;

/// Helper to create test input with a specific seed.
fn create_test_input(num_tokens: usize, seed: u32) -> UnigramInput {
    let tokens: Vec<M31> = (0..num_tokens)
        .map(|i| M31::from((i as u32 * 7 + seed) % 50000))
        .collect();
    let secret_key = [M31::from(seed); 8];
    UnigramInput::new(tokens, secret_key)
}

/// Helper to find a valid threshold for an input.
fn find_valid_threshold(input: &UnigramInput) -> u32 {
    let green_count = count_green_tokens(input);
    green_count.saturating_sub(5).max(1)
}

// ============================================================================
// PRF COMPONENT TESTS
// ============================================================================

#[test]
fn test_prf_deterministic() {
    let secret_key = [M31::from(42); 8];
    let token = M31::from(12345);
    let output1 = compute_prf(&secret_key, token);
    let output2 = compute_prf(&secret_key, token);
    assert_eq!(output1, output2, "PRF should be deterministic");
}

#[test]
fn test_prf_different_keys() {
    let token = M31::from(12345);
    let key1 = [M31::from(1); 8];
    let key2 = [M31::from(2); 8];
    let output1 = compute_prf(&key1, token);
    let output2 = compute_prf(&key2, token);
    assert_ne!(output1, output2, "Different keys should give different outputs");
}

#[test]
fn test_prf_different_tokens() {
    let secret_key = [M31::from(42); 8];
    let output1 = compute_prf(&secret_key, M31::from(1));
    let output2 = compute_prf(&secret_key, M31::from(2));
    assert_ne!(output1, output2, "Different tokens should give different outputs");
}

#[test]
fn test_green_classification() {
    let secret_key = [M31::from(42); 8];
    let mut green_count = 0;
    let total = 1000;
    for i in 0..total {
        let token = M31::from(i);
        let prf_output = compute_prf(&secret_key, token);
        if is_green_token(prf_output) {
            green_count += 1;
        }
    }
    assert!(green_count > 400 && green_count < 600, "Expected roughly 50% green");
}

// ============================================================================
// VALID PROOF TESTS
// ============================================================================

#[test]
fn test_valid_proof_small() {
    let input = create_test_input(32, 42);
    let threshold = find_valid_threshold(&input);
    let proof = Prover::default().prove(&input, threshold).expect("proving failed");
    Verifier::new().verify(proof).expect("verification should succeed");
}

#[test]
fn test_valid_proof_medium() {
    let input = create_test_input(128, 123);
    let threshold = find_valid_threshold(&input);
    let proof = Prover::default().prove(&input, threshold).expect("proving failed");
    Verifier::new().verify(proof).expect("verification should succeed");
}

#[test]
fn test_valid_proof_larger() {
    let input = create_test_input(256, 456);
    let threshold = find_valid_threshold(&input);
    let proof = Prover::default().prove(&input, threshold).expect("proving failed");
    Verifier::new().verify(proof).expect("verification should succeed");
}

#[test]
fn test_valid_proof_exact_threshold() {
    let input = create_test_input(100, 999);
    let green_count = count_green_tokens(&input);
    let proof = Prover::default().prove(&input, green_count).expect("proving failed");
    Verifier::new().verify(proof).expect("verification should succeed");
}

// ============================================================================
// REJECTION TESTS
// ============================================================================

#[test]
fn test_reject_empty_input() {
    let input = UnigramInput::new(vec![], [M31::from(0); 8]);
    let result = Prover::default().prove(&input, 0);
    assert!(result.is_err(), "Empty input should be rejected");
}

#[test]
fn test_reject_threshold_too_high() {
    let input = create_test_input(100, 111);
    let green_count = count_green_tokens(&input);
    let proof = Prover::default().prove(&input, green_count + 1).expect("proving failed");
    let result = Verifier::new().verify(proof);
    assert!(result.is_err(), "Threshold above green count should fail");
}

// ============================================================================
// ATTACK TESTS
// ============================================================================

#[test]
fn test_attack_tampered_public_tokens() {
    let input = create_test_input(64, 333);
    let threshold = find_valid_threshold(&input);
    let mut proof = Prover::default().prove(&input, threshold).expect("proving failed");
    if !proof.public_inputs.tokens.is_empty() {
        proof.public_inputs.tokens[0] = M31::from(99999);
    }
    let result = Verifier::new().verify(proof);
    assert!(result.is_err(), "Tampered tokens should fail");
}

#[test]
fn test_attack_tampered_threshold() {
    let input = create_test_input(64, 444);
    let threshold = find_valid_threshold(&input);
    let mut proof = Prover::default().prove(&input, threshold).expect("proving failed");
    proof.public_inputs.threshold = threshold + 10;
    let result = Verifier::new().verify(proof);
    assert!(result.is_err(), "Tampered threshold should fail");
}

#[test]
fn test_attack_tampered_num_tokens() {
    let input = create_test_input(64, 555);
    let threshold = find_valid_threshold(&input);
    let mut proof = Prover::default().prove(&input, threshold).expect("proving failed");
    proof.public_inputs.num_tokens = 128;
    let result = Verifier::new().verify(proof);
    assert!(result.is_err(), "Tampered num_tokens should fail");
}

#[test]
fn test_attack_tampered_green_count() {
    let input = create_test_input(64, 666);
    let threshold = find_valid_threshold(&input);
    let mut proof = Prover::default().prove(&input, threshold).expect("proving failed");
    proof.public_inputs.green_count = M31::from(proof.public_inputs.green_count.0 + 10);
    let result = Verifier::new().verify(proof);
    assert!(result.is_err(), "Tampered green_count should fail");
}

#[test]
fn test_attack_proof_swap() {
    let input1 = create_test_input(64, 888);
    let threshold1 = find_valid_threshold(&input1);
    let proof1 = Prover::default().prove(&input1, threshold1).expect("proving failed");

    let input2 = create_test_input(64, 999);
    let threshold2 = find_valid_threshold(&input2);
    let proof2 = Prover::default().prove(&input2, threshold2).expect("proving failed");

    let fake_proof = UnigramProof {
        public_inputs: proof2.public_inputs.clone(),
        stark_proof: proof1.stark_proof,
        pcs_config: proof1.pcs_config,
        prf_claimed_sum: proof1.prf_claimed_sum,
        unigram_claimed_sum: proof1.unigram_claimed_sum,
    };

    let result = Verifier::new().verify(fake_proof);
    assert!(result.is_err(), "Proof swap should fail");
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_minimum_tokens() {
    let input = create_test_input(16, 1000);
    let threshold = 1;
    let proof = Prover::default().prove(&input, threshold).expect("proving failed");
    Verifier::new().verify(proof).expect("verification should succeed");
}

#[test]
fn test_power_of_two_tokens() {
    for log_size in [4, 5, 6] {
        let num_tokens = 1 << log_size;
        let input = create_test_input(num_tokens, log_size * 100);
        let threshold = find_valid_threshold(&input);
        let proof = Prover::default().prove(&input, threshold).expect("proving failed");
        Verifier::new().verify(proof).expect("verification should succeed");
    }
}

#[test]
fn test_non_power_of_two_tokens() {
    for num_tokens in [17, 31, 50, 100] {
        let input = create_test_input(num_tokens, num_tokens as u32 * 7);
        let threshold = find_valid_threshold(&input);
        let proof = Prover::default().prove(&input, threshold).expect("proving failed");
        Verifier::new().verify(proof).expect("verification should succeed");
    }
}

// ============================================================================
// RANGE CHECK TESTS
// ============================================================================

#[test]
fn test_is_green_constraint_enforced() {
    for (num_tokens, seed) in [(64, 12345), (100, 999), (256, 42), (17, 31415)] {
        let input = create_test_input(num_tokens, seed);
        let green_count = count_green_tokens(&input);
        let threshold = green_count.saturating_sub(5).max(1);
        let proof = Prover::default()
            .prove(&input, threshold)
            .expect(&format!("proving failed for {} tokens", num_tokens));
        Verifier::new()
            .verify(proof)
            .expect(&format!("verification failed for {} tokens", num_tokens));
    }
}

#[test]
fn test_is_green_classification_consistency() {
    let input = create_test_input(256, 54321);
    let secret_key = [M31::from(54321); 8];
    
    let mut manual_green = 0u32;
    for &token in input.tokens() {
        let prf = compute_prf(&secret_key, token);
        if prf.0 < HALF_PRIME {
            manual_green += 1;
        }
    }
    
    let counted_green = count_green_tokens(&input);
    assert_eq!(manual_green, counted_green, "Manual count should match");
    
    let threshold = counted_green.saturating_sub(5).max(1);
    let proof = Prover::default().prove(&input, threshold).expect("proving failed");
    assert_eq!(proof.public_inputs.green_count.0, counted_green);
    Verifier::new().verify(proof).expect("verification should succeed");
}

