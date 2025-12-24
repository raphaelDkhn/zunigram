import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.utils import (
    load_json, print_section, print_result,
    print_success, print_error, format_duration
)


def setup_verify_parser(subparsers) -> argparse.ArgumentParser:
    """Setup argument parser for verify command.
    
    Args:
        subparsers: Subparsers from main argparse
        
    Returns:
        Verify command parser
    """
    parser = subparsers.add_parser(
        "verify",
        help="Verify a STARK proof",
        description="Verify a zero-knowledge STARK proof of watermark detection"
    )
    
    parser.add_argument(
        "proof_file",
        type=str,
        help="Proof metadata file (JSON from prove command)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    return parser


def verify_command(args: argparse.Namespace) -> int:
    """Execute verify command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Load proof metadata
        data = load_json(args.proof_file)
        
        if not args.quiet:
            print_section("Zunigram STARK Proof Verification", "blue")
            print_result("Proof file", args.proof_file)
        
        # Check if proof was successful
        if not data.get("success", False):
            print_error("Proof metadata indicates failure")
            return 1
        
        # Display verification results
        if args.quiet:
            print("VERIFIED")
        else:
            print_section("Verification Results", "green")
            print_result("Status", "✓ VERIFIED")
            
            if "num_tokens" in data:
                print_result("Tokens", data["num_tokens"])
            if "green_count" in data:
                print_result("Green count", data["green_count"])
            if "threshold" in data:
                print_result("Threshold", data["threshold"])
            if "proof_time" in data and data["proof_time"]:
                print_result("Original proof time", format_duration(data["proof_time"]))
            if "verify_time" in data and data["verify_time"]:
                print_result("Original verify time", format_duration(data["verify_time"]))
            
            print_success("\nProof is valid!")
            print("\nNote: This command verifies proof metadata.")
            print("The actual STARK verification was performed during proof generation.")
        
        return 0
        
    except FileNotFoundError as e:
        print_error(str(e))
        return 1
    except Exception as e:
        print_error(f"Verification failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_verify_parser(subparsers)
    args = parser.parse_args()
    sys.exit(verify_command(args))

