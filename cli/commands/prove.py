import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.prover import RustProverWrapper, build_prover_if_needed
from llm.utils import (
    load_json, save_json, load_secret_key,
    print_section, print_result, print_success, print_error,
    format_duration
)

try:
    import zunigram_py
except ImportError:
    zunigram_py = None


def setup_prove_parser(subparsers) -> argparse.ArgumentParser:
    """Setup argument parser for prove command.
    
    Args:
        subparsers: Subparsers from main argparse
        
    Returns:
        Prove command parser
    """
    parser = subparsers.add_parser(
        "prove",
        help="Generate STARK proof for watermarked text",
        description="Generate a zero-knowledge STARK proof that text contains a watermark"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input file (JSON from generate command or custom format)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for proof metadata (JSON format)"
    )
    
    parser.add_argument(
        "--secret-key-file",
        type=str,
        help="Secret key file (if not in input file)"
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        help="Green count threshold (computed automatically if not provided)"
    )
    
    parser.add_argument(
        "--auto-build",
        action="store_true",
        help="Automatically build prover binary if not found"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    return parser


def prove_command(args: argparse.Namespace) -> int:
    """Execute prove command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        if zunigram_py is None:
            print_error("zunigram_py not installed")
            print("Build with: cd crates/zunigram-py && maturin develop --release")
            return 1
        
        # Load input file
        data = load_json(args.input_file)
        
        # Extract token IDs
        if "token_ids" in data:
            token_ids = data["token_ids"]
        elif "tokens" in data:
            token_ids = data["tokens"]
        else:
            print_error("Input file must contain 'token_ids' or 'tokens' field")
            return 1
        
        # Extract secret key
        if "secret_key" in data:
            secret_key = data["secret_key"]
        elif args.secret_key_file:
            secret_key = load_secret_key(args.secret_key_file)
        else:
            print_error("Secret key not found. Provide via input file or --secret-key-file")
            return 1
        
        # Extract or compute threshold
        threshold = args.threshold
        if threshold is None and "threshold" in data:
            threshold = data["threshold"]
        
        if not args.quiet:
            print_section("Zunigram STARK Proof Generation", "blue")
            print_result("Input file", args.input_file)
            print_result("Total tokens", len(token_ids))
        
        # Check if prover binary exists
        try:
            prover = RustProverWrapper()
        except FileNotFoundError as e:
            if args.auto_build:
                if not args.quiet:
                    print("\nProver binary not found. Building...")
                if not build_prover_if_needed():
                    print_error("Failed to build prover binary")
                    return 1
                prover = RustProverWrapper()
            else:
                print_error(str(e))
                print("Use --auto-build to build automatically")
                return 1
        
        # Compute threshold if not provided
        if threshold is None:
            green_count = zunigram_py.count_green(secret_key, token_ids)
            threshold = max(1, green_count - 5)
            if not args.quiet:
                print_result("Green count", green_count)
                print_result("Threshold", threshold)
        else:
            if not args.quiet:
                print_result("Threshold", threshold)
        
        # Generate proof
        if not args.quiet:
            print("\nGenerating STARK proof...")
            print("(This may take a few seconds to minutes depending on text length)")
        
        result = prover.prove(
            tokens=token_ids,
            secret_key=secret_key,
            threshold=threshold,
            auto_threshold=False
        )
        
        if not result.success:
            print_error(f"Proof generation failed: {result.error}")
            if result.output and not args.quiet:
                print("\nProver output:")
                print(result.output)
            return 1
        
        # Display results
        if args.quiet:
            print("PROOF SUCCESS")
        else:
            print_section("Proof Results", "green")
            print_result("Status", "✓ SUCCESS")
            if result.proof_time:
                print_result("Proof generation time", format_duration(result.proof_time))
            if result.verify_time:
                print_result("Verification time", format_duration(result.verify_time))
            print_result("Tokens proved", result.num_tokens)
            print_result("Green count", result.green_count)
            print_result("Threshold", result.threshold)
            
            print_success("\nProof generated and verified successfully!")
        
        # Save proof metadata if requested
        if args.output:
            output_data = {
                "success": True,
                "proof_time": result.proof_time,
                "verify_time": result.verify_time,
                "num_tokens": result.num_tokens,
                "green_count": result.green_count,
                "threshold": result.threshold,
                "input_file": args.input_file,
            }
            save_json(output_data, args.output)
            
            if not args.quiet:
                print(f"\nProof metadata saved to {args.output}")
        
        return 0
        
    except FileNotFoundError as e:
        print_error(str(e))
        return 1
    except Exception as e:
        print_error(f"Proof generation failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_prove_parser(subparsers)
    args = parser.parse_args()
    sys.exit(prove_command(args))

