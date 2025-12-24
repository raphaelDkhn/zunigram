//! Benchmarks for Zunigram unigram watermark detection.

use common::{M31, UnigramInput};
use std::time::Instant;
use air::count_green_tokens;
use prover::Prover;
use verifier::Verifier;

fn main() {
    println!("Zunigram Benchmarks");
    println!("=================\n");

    let sizes = [256, 512, 1024, 2048];
    let secret_key = [M31::from(0x12345678_u32); 8];

    println!(
        "{:<12} {:>12} {:>15} {:>15}",
        "Tokens", "Green", "Prove Time", "Verify Time"
    );
    println!("{}", "-".repeat(60));

    for &num_tokens in &sizes {
        let tokens: Vec<M31> = (0..num_tokens)
            .map(|i| M31::from((i * 7 + 13) as u32 % 50000))
            .collect();

        let input = UnigramInput::new(tokens, secret_key);

        // Use actual green count minus margin as threshold
        let green_count = count_green_tokens(&input);
        let threshold = green_count.saturating_sub(5).max(1);

        let start = Instant::now();
        let proof = Prover::default()
            .prove(&input, threshold)
            .expect("proving failed");
        let prove_time = start.elapsed();

        let start = Instant::now();
        Verifier::new().verify(proof).expect("verification failed");
        let verify_time = start.elapsed();

        println!(
            "{:<12} {:>12} {:>15.2?} {:>15.2?}",
            num_tokens, green_count, prove_time, verify_time
        );
    }
}
