//! Example: Generate STARK proof from JSON input.
//!
//! This example accepts token data from a JSON file, enabling integration
//! with Python-generated watermarked text.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p prover --example prove_from_json --release -- input.json
//! ```
//!
//! # JSON Format
//!
//! ```json
//! {
//!     "tokens": [1234, 5678, ...],
//!     "secret_key": [305419896, 305419896, ...],
//!     "threshold": 55
//! }
//! ```

use common::{UnigramInput, M31};
use std::fs;
use std::time::Instant;
use air::count_green_tokens;
use prover::Prover;
use verifier::Verifier;
use serde::Deserialize;

#[derive(Deserialize)]
struct ProofInput {
    tokens: Vec<u32>,
    secret_key: Vec<u32>,
    threshold: Option<u32>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <input.json>", args[0]);
        eprintln!("\nJSON format:");
        eprintln!("{{");
        eprintln!("    \"tokens\": [1234, 5678, ...],");
        eprintln!("    \"secret_key\": [305419896, 305419896, ...],");
        eprintln!("    \"threshold\": 55");
        eprintln!("}}");
        std::process::exit(1);
    }
    
    let input_path = &args[1];
    
    println!("Zunigram - STARK Proof from JSON Input");
    println!("=====================================\n");
    
    // Read and parse JSON
    let json_content = fs::read_to_string(input_path)
        .expect("Failed to read input file");
    
    let proof_input: ProofInput = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON");
    
    // Validate secret key
    if proof_input.secret_key.len() != 8 {
        eprintln!("Error: secret_key must have exactly 8 elements");
        std::process::exit(1);
    }
    
    // Convert to M31 types
    let tokens: Vec<M31> = proof_input.tokens
        .iter()
        .map(|&t| M31::from(t))
        .collect();
    
    let secret_key: [M31; 8] = std::array::from_fn(|i| M31::from(proof_input.secret_key[i]));
    
    let input = UnigramInput::new(tokens.clone(), secret_key);
    
    // Count green tokens
    let green_count = count_green_tokens(&input);
    
    // Determine threshold
    let threshold = proof_input.threshold.unwrap_or_else(|| {
        // Default: set threshold slightly below actual count
        green_count.saturating_sub(5).max(1)
    });
    
    println!("Input:");
    println!("  - File: {}", input_path);
    println!("  - Number of tokens: {}", tokens.len());
    println!("  - Green count: {}", green_count);
    println!("  - Threshold T: {}", threshold);
    println!("  - Gamma: 0.5\n");
    
    // Check if proof is possible
    if green_count < threshold {
        eprintln!("Error: green_count ({}) < threshold ({})", green_count, threshold);
        eprintln!("Cannot generate proof for this input.");
        std::process::exit(1);
    }
    
    // Generate proof
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
    
    // Verify proof
    println!("Verifying proof...");
    let start = Instant::now();
    
    Verifier::new()
        .verify(proof)
        .expect("verification failed");
    
    let verify_time = start.elapsed();
    println!("  Verification: {:?}\n", verify_time);
    
    println!("SUCCESS!");
    println!("========");
    println!("  Proof generation:  {:?}", prove_time);
    println!("  Verification:      {:?}", verify_time);
    println!("  Total time:        {:?}", prove_time + verify_time);
}

