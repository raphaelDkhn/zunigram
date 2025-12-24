import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.watermark import WatermarkConfig
from llm.detect import ZunigramDetector
from llm.config import WatermarkParams
from llm.utils import (
    load_json, load_secret_key, print_section, print_result,
    print_success, print_error, print_warning, format_percentage
)


def setup_detect_parser(subparsers) -> argparse.ArgumentParser:
    """Setup argument parser for detect command.
    
    Args:
        subparsers: Subparsers from main argparse
        
    Returns:
        Detect command parser
    """
    parser = subparsers.add_parser(
        "detect",
        help="Detect watermark in text or token sequence",
        description="Detect watermark in text using zunigram detection algorithm"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input file (JSON from generate command or custom format)"
    )
    
    parser.add_argument(
        "--secret-key-file",
        type=str,
        help="Secret key file (if not in input file)"
    )
    
    parser.add_argument(
        "--z-threshold",
        type=float,
        help="Z-score threshold for detection (overrides input file)"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Vocabulary size (default: 50257 for GPT-2)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only print detection result)"
    )
    
    return parser


def generate_command(args: argparse.Namespace) -> int:
    """Execute detect command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
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
        
        # Extract z-threshold
        z_threshold = args.z_threshold
        if z_threshold is None and "z_threshold" in data:
            z_threshold = data["z_threshold"]
        elif z_threshold is None:
            z_threshold = 3.0  # default
        
        if not args.quiet:
            print_section("Zunigram Watermark Detection", "blue")
            print_result("Input file", args.input_file)
            print_result("Total tokens", len(token_ids))
            print_result("Z-threshold", z_threshold)
        
        # Create watermark config
        config = WatermarkConfig(
            secret_key=secret_key,
            z_threshold=z_threshold
        )
        
        # Create detector
        detector = ZunigramDetector(config, vocab_size=args.vocab_size)
        
        # Detect watermark
        if not args.quiet:
            print("\nDetecting watermark...")
        
        result = detector.detect(token_ids)
        
        # Display results
        if args.quiet:
            print("WATERMARKED" if result.is_watermarked else "NOT WATERMARKED")
        else:
            print_section("Detection Results", "green" if result.is_watermarked else "yellow")
            print(result)
            
            if result.is_watermarked:
                print_success("\nWatermark detected!")
            else:
                print_warning("\nNo watermark detected")
        
        # Return exit code based on detection
        return 0 if result.is_watermarked else 2
        
    except FileNotFoundError as e:
        print_error(str(e))
        return 1
    except Exception as e:
        print_error(f"Detection failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def detect_command(args: argparse.Namespace) -> int:
    """Alias for generate_command (correct naming)."""
    return generate_command(args)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_detect_parser(subparsers)
    args = parser.parse_args()
    sys.exit(detect_command(args))

