import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.watermark import WatermarkConfig
from llm.generate import ZunigramGenerator, GenerationConfig
from llm.utils import save_json, print_section, print_result, print_success, print_error, format_duration, format_percentage


def setup_generate_parser(subparsers) -> argparse.ArgumentParser:
    """Setup argument parser for generate command.
    
    Args:
        subparsers: Subparsers from main argparse
        
    Returns:
        Generate command parser
    """
    parser = subparsers.add_parser(
        "generate",
        help="Generate watermarked text from a prompt",
        description="Generate watermarked text using an LLM with zunigram watermarking"
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt for generation"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model to use for generation (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0). Any HuggingFace causal LM can be used."
    )
    
    parser.add_argument(
        "-n", "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=50,
        help="Top-K sampling"
    )
    
    parser.add_argument(
        "-p", "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold (default: 0.95)"
    )
    
    parser.add_argument(
        "--no-watermark",
        action="store_true",
        help="Generate without watermark (for comparison)"
    )
    
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable auto-detection of chat templates (use raw prompt)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for secret key generation (default: 42)"
    )
    
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Watermark strength (logit bias) (default: 2.0)"
    )
    
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for detection"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (JSON format)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device for model inference (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only print generated text)"
    )
    
    return parser


def generate_command(args: argparse.Namespace) -> int:
    """Execute generate command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Import here to avoid slow startup if not needed
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        if not args.quiet:
            print_section("Zunigram Text Generation", "blue")
            print_result("Model", args.model)
            print_result("Prompt", f'"{args.prompt}"')
            print_result("Max tokens", args.max_tokens)
            print_result("Watermark", "disabled" if args.no_watermark else "enabled")
            if not args.no_watermark:
                print_result("  Seed", args.seed, indent=1)
                print_result("  Delta", args.delta, indent=1)
                print_result("  Z-threshold", args.z_threshold, indent=1)
        
        # Load model and tokenizer
        if not args.quiet:
            print("\nLoading model...")
        
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Set device
        if args.device:
            device = args.device
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        model = model.to(device)
        
        if not args.quiet:
            print_success(f"Model loaded on {device}")
        
        # Create watermark config
        watermark_config = WatermarkConfig.with_seed(
            args.seed,
            delta=args.delta,
            z_threshold=args.z_threshold
        )
        
        # Create generator
        generator = ZunigramGenerator(
            model=model,
            tokenizer=tokenizer,
            watermark_config=watermark_config,
            device=device
        )
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            use_chat_template=False if args.no_chat_template else None  # None = auto-detect
        )
        
        # Generate text with streaming
        start_time = time.time()
        
        if not args.quiet:
            # Streaming mode - display tokens as they're generated
            print("\nGenerating text...")
            print("\n" + "=" * 60)
            print(f"{args.prompt}", end="", flush=True)
            
            generated_tokens = []
            for token_id, new_text in generator.generate_stream(
                args.prompt,
                generation_config=gen_config,
                apply_watermark=not args.no_watermark
            ):
                generated_tokens.append(token_id)
                print(new_text, end="", flush=True)
            
            print()  # New line after streaming
            print("=" * 60 + "\n")
            
            # Calculate statistics from generated tokens
            green_count = generator.watermark.count_green(generated_tokens)
            z_score = generator.watermark.compute_z_score(generated_tokens)
            output_text = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_text = args.prompt + output_text
            
            from llm.generate import GenerationOutput
            output = GenerationOutput(
                text=output_text,
                full_text=full_text,
                prompt=args.prompt,
                token_ids=generated_tokens,
                green_count=green_count,
                total_tokens=len(generated_tokens),
                z_score=z_score,
                is_watermarked=not args.no_watermark,
                secret_key=watermark_config.secret_key,
            )
        else:
            # Quiet mode - use non-streaming for efficiency
            output = generator.generate(
                args.prompt,
                generation_config=gen_config,
                apply_watermark=not args.no_watermark
            )
            print(output.text)
        
        generation_time = time.time() - start_time
        
        # Display statistics (if not quiet)
        if not args.quiet:
            print_section("Statistics", "blue")
            print_result("Generation time", format_duration(generation_time))
            print_result("Total tokens", output.total_tokens)
            print_result("Green tokens", output.green_count)
            print_result("Green ratio", format_percentage(output.green_ratio))
            print_result("Z-score", f"{output.z_score:.4f}")
            print_result("Watermarked", "Yes" if output.is_watermarked else "No")
        
        # Save to file if requested
        if args.output:
            output_data = output.to_dict()
            output_data["prompt"] = args.prompt
            output_data["model"] = args.model
            output_data["generation_time"] = generation_time
            
            save_json(output_data, args.output)
            
            if not args.quiet:
                print_success(f"Results saved to {args.output}")
        
        return 0
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print("Install with: pip install transformers torch")
        return 1
    except Exception as e:
        print_error(f"Generation failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_generate_parser(subparsers)
    args = parser.parse_args()
    sys.exit(generate_command(args))

