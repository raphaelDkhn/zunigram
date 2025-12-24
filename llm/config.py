from dataclasses import dataclass
from typing import Optional

# Default watermark parameters
DEFAULT_Z_THRESHOLD = 3.0
DEFAULT_DELTA = 2.0
DEFAULT_GAMMA = 0.5

# Default generation parameters
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95


# Paths
DEFAULT_OUTPUT_DIR = "outputs"


@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    name: str
    device: Optional[str] = None
    
    @property
    def model_id(self) -> str:
        """Get HuggingFace model identifier."""
        return self.name


@dataclass
class WatermarkParams:
    """Parameters for watermark generation and detection."""
    z_threshold: float = DEFAULT_Z_THRESHOLD
    delta: float = DEFAULT_DELTA
    gamma: float = DEFAULT_GAMMA
    
    def __post_init__(self):
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if not (0 < self.gamma < 1):
            raise ValueError("gamma must be between 0 and 1")


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    do_sample: bool = True
    
    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")

