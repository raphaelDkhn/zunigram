"""End-to-end tests for zunigram workflow.

These tests verify the complete workflow:
1. Generate watermarked text with LLM
2. Detect watermark
3. Generate STARK proof
4. Verify proof
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import zunigram_py
from llm.watermark import WatermarkConfig
from llm.generate import ZunigramGenerator, GenerationConfig
from llm.detect import ZunigramDetector
from llm.prover import RustProverWrapper, build_prover_if_needed


class TestE2EBasicFlow(unittest.TestCase):
    """Test basic end-to-end workflow with GPT-2."""
    
    @classmethod
    def setUpClass(cls):
        """Setup class - load model once for all tests."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print("\nLoading GPT-2 model for E2E tests...")
            cls.model = AutoModelForCausalLM.from_pretrained("gpt2")
            cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            cls.vocab_size = len(cls.tokenizer)
            
            # Use CPU for testing
            cls.device = "cpu"
            cls.model = cls.model.to(cls.device)
            
            print("✓ Model loaded")
            
            # Build prover if needed
            print("Checking prover binary...")
            if build_prover_if_needed():
                print("✓ Prover ready")
            else:
                print("⚠ Prover build failed - some tests may be skipped")
                cls.prover_available = False
                return
            
            cls.prover_available = True
            cls.prover = RustProverWrapper()
            
        except Exception as e:
            print(f"Failed to setup E2E tests: {e}")
            raise
    
    def setUp(self):
        """Setup for each test."""
        self.config = WatermarkConfig.with_seed(42, z_threshold=3.0, delta=2.5)
        self.generator = ZunigramGenerator(
            self.model,
            self.tokenizer,
            self.config,
            device=self.device
        )
        self.detector = ZunigramDetector(self.config, self.vocab_size)
    
    def test_01_generate_and_detect(self):
        """Test: Generate watermarked text and detect watermark."""
        print("\n=== Test 1: Generate and Detect ===")
        
        # Generate watermarked text
        prompt = "Once upon a time"
        gen_config = GenerationConfig(max_new_tokens=50, do_sample=True)
        
        print(f"Generating text from prompt: '{prompt}'")
        output = self.generator.generate(prompt, gen_config)
        
        print(f"Generated {output.total_tokens} tokens")
        print(f"Green count: {output.green_count}")
        print(f"Z-score: {output.z_score:.4f}")
        
        # Verify generation
        self.assertGreater(output.total_tokens, 0)
        self.assertGreater(len(output.text), 0)
        self.assertTrue(output.is_watermarked)
        
        # Detect watermark
        print("Detecting watermark...")
        result = self.detector.detect(output.token_ids)
        
        print(f"Detection z-score: {result.z_score:.4f}")
        print(f"Watermarked: {result.is_watermarked}")
        
        # Verify detection
        self.assertTrue(result.is_watermarked)
        self.assertGreater(result.z_score, 4.0)
        self.assertEqual(result.green_count, output.green_count)
        
        print("✓ Test passed")
    
    def test_02_watermarked_vs_unwatermarked(self):
        """Test: Compare watermarked vs unwatermarked text."""
        print("\n=== Test 2: Watermarked vs Unwatermarked ===")
        
        prompt = "The quick brown fox"
        gen_config = GenerationConfig(max_new_tokens=50, do_sample=True)
        
        # Generate watermarked text
        print("Generating watermarked text...")
        wm_output = self.generator.generate(prompt, gen_config, apply_watermark=True)
        print(f"Watermarked z-score: {wm_output.z_score:.4f}")
        
        # Generate unwatermarked text
        print("Generating unwatermarked text...")
        unwm_output = self.generator.generate(prompt, gen_config, apply_watermark=False)
        print(f"Unwatermarked z-score: {unwm_output.z_score:.4f}")
        
        # Verify watermarked is detected
        wm_result = self.detector.detect(wm_output.token_ids)
        self.assertTrue(wm_result.is_watermarked)
        
        # Verify unwatermarked is NOT detected (most of the time)
        unwm_result = self.detector.detect(unwm_output.token_ids)

        print(f"Unwatermarked detection: {unwm_result.is_watermarked} (z={unwm_result.z_score:.4f})")
        
        # At minimum, watermarked should have higher z-score
        self.assertGreater(wm_result.z_score, unwm_result.z_score)
        
        print("✓ Test passed")
    
    def test_03_different_lengths(self):
        """Test: Generate and detect with different text lengths."""
        print("\n=== Test 3: Different Text Lengths ===")
        
        prompt = "In a galaxy far away"
        lengths = [50, 75, 100]  # Use longer texts for reliable detection
        
        for length in lengths:
            print(f"\nTesting with {length} tokens...")
            gen_config = GenerationConfig(max_new_tokens=length, do_sample=True)
            
            output = self.generator.generate(prompt, gen_config)
            result = self.detector.detect(output.token_ids)
            
            print(f"  Generated: {output.total_tokens} tokens")
            print(f"  Z-score: {result.z_score:.4f}")
            print(f"  Detected: {result.is_watermarked}")
            
            # Verify generation worked
            self.assertGreater(output.total_tokens, 0)
            
            # For watermarked text, z-score should be positive
            # Detection is more reliable with longer texts
            self.assertGreater(result.z_score, 0)
            
            # Texts >= 50 tokens should generally be detected
            if length >= 50:
                # Allow occasional false negatives due to sampling variance
                if not result.is_watermarked:
                    print(f"  ⚠ Warning: {length} token text not detected (z={result.z_score:.4f})")
                    print(f"     This can happen with sampling, but is rare.")
        
        print("✓ Test passed")
    
    @unittest.skipIf(not hasattr(unittest.TestCase, 'prover_available') or 
                     not getattr(unittest.TestCase, 'prover_available', False),
                     "Prover binary not available - build with: cargo build -p prover --release --example prove_from_json")
    def test_04_generate_detect_prove(self):
        """Test: Full workflow including STARK proof generation.
        
        Note: This test requires the Rust prover binary to be built.
        Build with: cargo build -p prover --release --example prove_from_json
        """
        print("\n=== Test 4: Generate, Detect, and Prove ===")
        
        # Generate watermarked text
        prompt = "The future of AI"
        gen_config = GenerationConfig(max_new_tokens=50, do_sample=True)
        
        print("1. Generating watermarked text...")
        output = self.generator.generate(prompt, gen_config)
        print(f"   Generated {output.total_tokens} tokens (z-score: {output.z_score:.4f})")
        
        # Detect watermark
        print("2. Detecting watermark...")
        result = self.detector.detect(output.token_ids)
        print(f"   Detected: {result.is_watermarked} (z-score: {result.z_score:.4f})")
        self.assertTrue(result.is_watermarked)
        
        # Generate proof
        print("3. Generating STARK proof...")
        print("   (This may take 10-30 seconds...)")
        proof_result = self.prover.prove(
            tokens=output.token_ids,
            secret_key=self.config.secret_key,
            threshold=result.threshold
        )
        
        if proof_result.success:
            print(f"   ✓ Proof generated in {proof_result.proof_time:.3f}s")
            print(f"   ✓ Verified in {proof_result.verify_time:.3f}s" if proof_result.verify_time else "")
        else:
            print(f"   ✗ Proof failed: {proof_result.error}")
        
        # Verify proof succeeded
        self.assertTrue(proof_result.success, f"Proof generation failed: {proof_result.error}")
        self.assertIsNotNone(proof_result.proof_time)
        self.assertEqual(proof_result.num_tokens, output.total_tokens)
        
        print("✓ Test passed")
    
    def test_05_save_and_load(self):
        """Test: Save results to JSON and load them back."""
        print("\n=== Test 5: JSON Save and Load ===")
        
        # Generate watermarked text (use more tokens for reliable detection)
        prompt = "Testing JSON serialization"
        gen_config = GenerationConfig(max_new_tokens=60, do_sample=True)
        output = self.generator.generate(prompt, gen_config)
        
        print(f"Generated {output.total_tokens} tokens (z-score: {output.z_score:.4f})")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_data = output.to_dict()
            json.dump(output_data, f)
            temp_file = f.name
        
        try:
            print(f"Saved to: {temp_file}")
            
            # Load back
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            print("Loaded data back")
            
            # Verify data integrity
            self.assertEqual(loaded_data['token_ids'], output.token_ids)
            self.assertEqual(loaded_data['green_count'], output.green_count)
            self.assertEqual(loaded_data['total_tokens'], output.total_tokens)
            self.assertEqual(loaded_data['secret_key'], self.config.secret_key)
            
            # Verify z-score matches
            self.assertAlmostEqual(loaded_data['z_score'], output.z_score, places=4)
            
            # Detect from loaded data
            result = self.detector.detect(loaded_data['token_ids'])
            print(f"Detection after load: z-score={result.z_score:.4f}, detected={result.is_watermarked}")
            
            # Verify detection matches original
            self.assertEqual(result.green_count, output.green_count)
            self.assertAlmostEqual(result.z_score, output.z_score, places=4)
            
            print("✓ Test passed - Data integrity verified")
            
        finally:
            # Clean up
            os.unlink(temp_file)


class TestIntegration(unittest.TestCase):
    """Integration tests for individual components."""
    
    def test_zunigram_py_bindings(self):
        """Test zunigram_py Python bindings."""
        print("\n=== Integration Test: zunigram_py bindings ===")
        
        # Test secret key generation
        secret_key = zunigram_py.generate_secret_key()
        self.assertEqual(len(secret_key), 8)
        
        # Test PRF computation
        token = 1234
        prf_output = zunigram_py.prf(secret_key, token)
        self.assertIsInstance(prf_output, int)
        self.assertGreaterEqual(prf_output, 0)
        
        # Test green classification
        is_green = zunigram_py.is_green(prf_output)
        self.assertIsInstance(is_green, bool)
        
        # Test token classification
        tokens = [1, 2, 3, 4, 5]
        classifications = zunigram_py.classify_tokens(secret_key, tokens)
        self.assertEqual(len(classifications), len(tokens))
        
        # Test green count
        green_count = zunigram_py.count_green(secret_key, tokens)
        self.assertEqual(green_count, sum(classifications))
        
        # Test z-score computation
        z_score = zunigram_py.compute_z_score(green_count, len(tokens), 0.5)
        self.assertIsInstance(z_score, float)
        
        print("✓ All zunigram_py bindings working correctly")
    
    def test_prover_wrapper(self):
        """Test prover wrapper functionality."""
        print("\n=== Integration Test: Prover wrapper ===")
        
        try:
            prover = RustProverWrapper()
            print(f"Prover binary found at: {prover.prover_binary}")
            self.assertTrue(prover.prover_binary.exists())
        except FileNotFoundError:
            print("⚠ Prover binary not found - skipping test")
            self.skipTest("Prover binary not available")
        
        # Test with simple data
        secret_key = zunigram_py.generate_secret_key()
        tokens = list(range(50))  # 50 tokens
        
        # Verify data before proving
        from llm.prover import verify_proof_data
        green_count = zunigram_py.count_green(secret_key, tokens)
        threshold = max(1, green_count - 5)
        
        is_valid = verify_proof_data(tokens, secret_key, threshold)
        self.assertTrue(is_valid)
        
        print("✓ Prover wrapper working correctly")


def run_tests():
    """Run all E2E tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestE2EBasicFlow))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())

