import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import zunigram_py
except ImportError:
    raise ImportError(
        "zunigram_py not installed. Build with: cd crates/zunigram-py && maturin develop --release"
    )


@dataclass
class ProofResult:
    """Result from proof generation.
    
    Attributes:
        success: Whether proof generation succeeded
        proof_time: Time taken to generate proof (seconds)
        verify_time: Time taken to verify proof (seconds)
        num_tokens: Number of tokens in the proof
        green_count: Number of green tokens
        threshold: Threshold used for proof
        output: Raw output from prover
        error: Error message if failed
    """
    success: bool
    proof_time: Optional[float] = None
    verify_time: Optional[float] = None
    num_tokens: Optional[int] = None
    green_count: Optional[int] = None
    threshold: Optional[int] = None
    output: Optional[str] = None
    error: Optional[str] = None


class RustProverWrapper:
    """Wrapper for calling Rust prover via subprocess."""
    
    def __init__(self, prover_binary: Optional[str] = None):
        """Initialize prover wrapper.
        
        Args:
            prover_binary: Path to prover binary (auto-detected if None)
        """
        if prover_binary is None:
            self.prover_binary = self._find_prover_binary()
        else:
            self.prover_binary = Path(prover_binary)
        
        if not self.prover_binary.exists():
            raise FileNotFoundError(
                f"Prover binary not found at {self.prover_binary}. "
                "Build with: cargo build -p prover --release --example prove_from_json"
            )
    
    def _find_prover_binary(self) -> Path:
        """Find the prover binary in standard locations."""
        # Try multiple possible locations
        workspace_root = Path(__file__).parent.parent
        
        possible_paths = [
            workspace_root / "target" / "release" / "examples" / "prove_from_json",
            workspace_root / "target" / "debug" / "examples" / "prove_from_json",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Default to release path
        return workspace_root / "target" / "release" / "examples" / "prove_from_json"
    
    def prove(
        self,
        tokens: List[int],
        secret_key: List[int],
        threshold: Optional[int] = None,
        auto_threshold: bool = True,
    ) -> ProofResult:
        """Generate a STARK proof for watermarked text.
        
        Args:
            tokens: List of token IDs
            secret_key: List of 8 u32 values for secret key
            threshold: Green count threshold (computed if None)
            auto_threshold: Compute threshold automatically if None
            
        Returns:
            ProofResult with proof generation results
        """
        # Validate inputs
        if len(secret_key) != 8:
            return ProofResult(
                success=False,
                error=f"Secret key must have 8 elements, got {len(secret_key)}"
            )
        
        if len(tokens) == 0:
            return ProofResult(
                success=False,
                error="Cannot prove with zero tokens"
            )
        
        # Compute threshold if not provided
        if threshold is None and auto_threshold:
            green_count = zunigram_py.count_green(secret_key, tokens)
            # Set threshold slightly below actual count for safety
            threshold = max(1, green_count - 5)
        elif threshold is None:
            return ProofResult(
                success=False,
                error="threshold must be provided or auto_threshold must be True"
            )
        
        # Create temporary file for input
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            input_data = {
                "tokens": tokens,
                "secret_key": secret_key,
                "threshold": threshold
            }
            json.dump(input_data, f)
            input_file = f.name
        
        try:
            # Run prover
            start_time = time.time()
            result = subprocess.run(
                [str(self.prover_binary), input_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            total_time = time.time() - start_time
            
            if result.returncode != 0:
                return ProofResult(
                    success=False,
                    error=f"Prover failed: {result.stderr}",
                    output=result.stdout
                )
            
            # Parse output for timing information
            proof_time, verify_time = self._parse_timing(result.stdout)
            
            return ProofResult(
                success=True,
                proof_time=proof_time or total_time,
                verify_time=verify_time,
                num_tokens=len(tokens),
                green_count=zunigram_py.count_green(secret_key, tokens),
                threshold=threshold,
                output=result.stdout
            )
            
        except subprocess.TimeoutExpired:
            return ProofResult(
                success=False,
                error="Proof generation timed out (5 minutes)"
            )
        except Exception as e:
            return ProofResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
        finally:
            # Clean up temp file
            Path(input_file).unlink(missing_ok=True)
    
    def _parse_timing(self, output: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse timing information from prover output.
        
        Args:
            output: Prover output text
            
        Returns:
            Tuple of (proof_time, verify_time) in seconds
        """
        proof_time = None
        verify_time = None
        
        for line in output.split('\n'):
            if "Proof generation:" in line:
                # Extract time like "21.729083ms" or "1.234s"
                time_str = line.split(":")[-1].strip()
                proof_time = self._parse_duration(time_str)
            elif "Verification:" in line:
                time_str = line.split(":")[-1].strip()
                verify_time = self._parse_duration(time_str)
        
        return proof_time, verify_time
    
    def _parse_duration(self, duration_str: str) -> Optional[float]:
        """Parse duration string to seconds.
        
        Args:
            duration_str: Duration like "21.729083ms" or "1.234s"
            
        Returns:
            Duration in seconds or None if parsing fails
        """
        try:
            duration_str = duration_str.strip()
            if duration_str.endswith('ms'):
                return float(duration_str[:-2]) / 1000
            elif duration_str.endswith('µs') or duration_str.endswith('us'):
                return float(duration_str[:-2]) / 1_000_000
            elif duration_str.endswith('s'):
                return float(duration_str[:-1])
            else:
                # Try parsing as float (assume seconds)
                return float(duration_str)
        except (ValueError, IndexError):
            return None


def build_prover_if_needed(force: bool = False) -> bool:
    """Build the prover binary if it doesn't exist.
    
    Args:
        force: Force rebuild even if binary exists
        
    Returns:
        True if build succeeded, False otherwise
    """
    wrapper = RustProverWrapper()
    
    if wrapper.prover_binary.exists() and not force:
        return True
    
    print("Building Rust prover binary...")
    workspace_root = Path(__file__).parent.parent
    
    try:
        result = subprocess.run(
            ["cargo", "build", "-p", "prover", "--release", "--example", "prove_from_json"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for build
        )
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False
        
        print("✓ Prover binary built successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("Build timed out")
        return False
    except Exception as e:
        print(f"Build failed: {e}")
        return False


def verify_proof_data(
    tokens: List[int],
    secret_key: List[int],
    threshold: int,
) -> bool:
    """Verify that proof data is valid without generating full STARK proof.
    
    This is a fast pre-check that validates:
    - Secret key is correct format
    - Green count meets threshold
    - Tokens are valid
    
    Args:
        tokens: List of token IDs
        secret_key: List of 8 u32 values
        threshold: Required green count threshold
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check secret key length
        if len(secret_key) != 8:
            return False
        
        # Check tokens
        if len(tokens) == 0:
            return False
        
        # Check green count meets threshold
        green_count = zunigram_py.count_green(secret_key, tokens)
        return green_count >= threshold
        
    except Exception:
        return False

