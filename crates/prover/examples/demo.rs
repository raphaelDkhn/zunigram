use common::{UnigramInput, M31};
use std::time::Instant;
use air::count_green_tokens;
use prover::Prover;
use verifier::Verifier;

fn main() {
    println!("Zunigram - Unigram Watermark Detection Prover");
    println!("============================================\n");

    // Generate tokens for demonstration
    let num_tokens = 256;
    let secret_key = [M31::from(0x12345678_u32); 8];
    
    // Create tokens (simulating a watermarked text)
    let tokens: Vec<M31> = (0..num_tokens)
        .map(|i| M31::from((i * 7 + 13) as u32 % 50000))
        .collect();

    let input = UnigramInput::new(tokens.clone(), secret_key);
    
    // Count actual green tokens
    let green_count = count_green_tokens(&input);
    
    // Use threshold slightly below actual count
    let threshold = green_count.saturating_sub(5).max(1);

    println!("Configuration:");
    println!("  - Number of tokens: {}", num_tokens);
    println!("  - Green count: {}", green_count);
    println!("  - Threshold T: {}", threshold);
    println!("  - Gamma: 0.5\n");

    // Proof generation
    println!("Generating proof...");
    let start = Instant::now();
    
    let proof = Prover::default()
        .prove(&input, threshold)
        .expect("proving failed");
    
    let prove_time = start.elapsed();
    println!("  Proof generation: {:?}", prove_time);
    println!("  Public inputs bound via LogUp:");
    println!("    - {} tokens", proof.public_inputs.num_tokens);
    println!("    - green_count = {}", proof.public_inputs.green_count.0);
    println!("    - threshold = {}\n", proof.public_inputs.threshold);

    // Verification
    println!("Verifying proof...");
    let start = Instant::now();
    
    Verifier::new()
        .verify(proof)
        .expect("verification failed");
    
    let verify_time = start.elapsed();
    println!("  Verification: {:?}\n", verify_time);

    println!("Summary:");
    println!("  Proof generation:  {:?}", prove_time);
    println!("  Verification:      {:?}", verify_time);
    println!("  Total time:        {:?}", prove_time + verify_time);
}

