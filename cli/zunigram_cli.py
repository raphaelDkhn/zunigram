#!/usr/bin/env python3
"""Zunigram CLI - Command-line interface for watermark generation, detection, and proving.

Usage:
    zunigram generate "prompt" [options]
    zunigram detect input.json [options]
    zunigram prove input.json [options]
    zunigram verify proof.json [options]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.commands import (
    generate_command, detect_command, prove_command, verify_command
)
from cli.commands.generate import setup_generate_parser
from cli.commands.detect import setup_detect_parser
from cli.commands.prove import setup_prove_parser
from cli.commands.verify import setup_verify_parser


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="zunigram",
        description="Zunigram - LLM watermark generation, detection, and zero-knowledge proving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate watermarked text
  zunigram generate "Once upon a time" -o output.json

  # Detect watermark in generated text
  zunigram detect output.json

  # Generate STARK proof for watermarked text
  zunigram prove output.json -o proof.json

  # Verify the proof
  zunigram verify proof.json

  # Complete workflow
  zunigram generate "Hello world" -o text.json && \\
  zunigram detect text.json && \\
  zunigram prove text.json -o proof.json && \\
  zunigram verify proof.json
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="zunigram 0.1.0"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        help="Command to execute"
    )
    
    # Setup command parsers
    generate_parser = setup_generate_parser(subparsers)
    detect_parser = setup_detect_parser(subparsers)
    prove_parser = setup_prove_parser(subparsers)
    verify_parser = setup_verify_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "generate":
        return generate_command(args)
    elif args.command == "detect":
        return detect_command(args)
    elif args.command == "prove":
        return prove_command(args)
    elif args.command == "verify":
        return verify_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

