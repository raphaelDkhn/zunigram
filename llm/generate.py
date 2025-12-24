from dataclasses import dataclass
from typing import List, Optional, Union, Callable
import numpy as np

from .watermark import ZunigramWatermark, WatermarkConfig


@dataclass
class GenerationConfig:
    """Configuration for text generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no scaling)
        top_k: Top-K sampling (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        do_sample: Whether to sample (False = greedy)
        use_chat_template: Whether to use chat template (None = auto-detect)
    """
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    use_chat_template: Optional[bool] = None


class ZunigramGenerator:
    """LLM text generator with Unigram watermarking.
    
    This generator applies watermark bias during token sampling to embed
    a detectable watermark in the generated text.
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> 
        >>> config = WatermarkConfig.with_seed(42)
        >>> generator = ZunigramGenerator(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     watermark_config=config
        ... )
        >>> 
        >>> output = generator.generate("Once upon a time")
        >>> print(output.text)
    """
    
    def __init__(
        self,
        model,  # HuggingFace model or compatible
        tokenizer,  # HuggingFace tokenizer or compatible
        watermark_config: WatermarkConfig,
        device: Optional[str] = None,
    ):
        """Initialize generator with model and watermark config.
        
        Args:
            model: Language model with forward() method
            tokenizer: Tokenizer for encoding/decoding
            watermark_config: Watermark configuration
            device: Device for inference (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.watermark_config = watermark_config
        
        # Determine device
        if device is None:
            import torch
            if hasattr(model, 'device'):
                self.device = model.device
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Get vocabulary size from tokenizer
        self.vocab_size = len(tokenizer)
        
        # Create watermark instance
        self.watermark = ZunigramWatermark(watermark_config, self.vocab_size)
        
        # Precompute logit bias tensor
        import torch
        self._logit_bias = torch.tensor(
            self.watermark.get_logit_bias(),
            dtype=torch.float32,
            device=self.device
        )
        
        # Check if tokenizer has chat template support
        self._has_chat_template = self._detect_chat_template()
    
    def _detect_chat_template(self) -> bool:
        """Detect if the tokenizer supports chat templates.
        
        Returns:
            True if chat template is available
        """
        # Check for chat_template attribute (modern transformers)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            return True
        
        # Check for apply_chat_template method
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Try to see if it works (some tokenizers have the method but no template)
            try:
                test_messages = [{"role": "user", "content": "test"}]
                self.tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
                return True
            except Exception:
                return False
        
        return False
    
    def _format_prompt(self, prompt: str, use_chat_template: Optional[bool] = None) -> str:
        """Format prompt with chat template if available.
        
        Args:
            prompt: Raw user prompt
            use_chat_template: Override auto-detection (None = auto)
        
        Returns:
            Formatted prompt string
        """
        # Determine whether to use chat template
        should_use_template = use_chat_template if use_chat_template is not None else self._has_chat_template
        
        if not should_use_template:
            return prompt
        
        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception:
            # Fall back to raw prompt if template fails
            return prompt
    
    def generate_stream(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        apply_watermark: bool = True,
    ):
        """Generate watermarked text from a prompt with streaming output.
        
        Args:
            prompt: Input text prompt
            generation_config: Generation parameters
            apply_watermark: Whether to apply watermark (for comparison)
        
        Yields:
            Tuple of (token_id, decoded_text) as tokens are generated
        """
        import torch
        
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Format prompt with chat template if available
        formatted_prompt = self._format_prompt(prompt, generation_config.use_chat_template)
        
        # Encode prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        
        # Generate tokens one at a time with watermark bias
        generated_ids = input_ids.clone()
        generated_tokens = []
        last_decoded_length = 0
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(generation_config.max_new_tokens):
                # Get model outputs
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply watermark bias
                if apply_watermark:
                    next_token_logits = next_token_logits + self._logit_bias
                
                # Apply temperature
                if generation_config.temperature != 1.0:
                    next_token_logits = next_token_logits / generation_config.temperature
                
                # Apply top-k filtering
                if generation_config.top_k > 0:
                    next_token_logits = self._top_k_filtering(
                        next_token_logits, generation_config.top_k
                    )
                
                # Apply top-p (nucleus) filtering
                if generation_config.top_p < 1.0:
                    next_token_logits = self._top_p_filtering(
                        next_token_logits, generation_config.top_p
                    )
                
                # Sample or greedy decode
                if generation_config.do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append token
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                token_id = next_token.item()
                generated_tokens.append(token_id)
                
                # Decode current state and yield new text
                current_output_ids = generated_ids[0, prompt_length:].tolist()
                current_decoded = self.tokenizer.decode(current_output_ids, skip_special_tokens=True)
                
                # Yield only the new portion
                new_text = current_decoded[last_decoded_length:]
                last_decoded_length = len(current_decoded)
                
                yield token_id, new_text
                
                # Check for EOS
                if hasattr(self.tokenizer, 'eos_token_id'):
                    if token_id == self.tokenizer.eos_token_id:
                        break
    
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        apply_watermark: bool = True,
    ) -> "GenerationOutput":
        """Generate watermarked text from a prompt.
        
        Args:
            prompt: Input text prompt
            generation_config: Generation parameters
            apply_watermark: Whether to apply watermark (for comparison)
        
        Returns:
            GenerationOutput with text and metadata
        """
        import torch
        
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Format prompt with chat template if available
        formatted_prompt = self._format_prompt(prompt, generation_config.use_chat_template)
        
        # Encode prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        
        # Generate tokens one at a time with watermark bias
        generated_ids = input_ids.clone()
        generated_tokens = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(generation_config.max_new_tokens):
                # Get model outputs
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply watermark bias
                if apply_watermark:
                    next_token_logits = next_token_logits + self._logit_bias
                
                # Apply temperature
                if generation_config.temperature != 1.0:
                    next_token_logits = next_token_logits / generation_config.temperature
                
                # Apply top-k filtering
                if generation_config.top_k > 0:
                    next_token_logits = self._top_k_filtering(
                        next_token_logits, generation_config.top_k
                    )
                
                # Apply top-p (nucleus) filtering
                if generation_config.top_p < 1.0:
                    next_token_logits = self._top_p_filtering(
                        next_token_logits, generation_config.top_p
                    )
                
                # Sample or greedy decode
                if generation_config.do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append token
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Check for EOS
                if hasattr(self.tokenizer, 'eos_token_id'):
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
        
        # Decode output
        output_ids = generated_ids[0, prompt_length:].tolist()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Compute watermark statistics
        green_count = self.watermark.count_green(output_ids)
        z_score = self.watermark.compute_z_score(output_ids)
        
        return GenerationOutput(
            text=output_text,
            full_text=full_text,
            prompt=prompt,
            token_ids=output_ids,
            green_count=green_count,
            total_tokens=len(output_ids),
            z_score=z_score,
            is_watermarked=apply_watermark,
            secret_key=self.watermark_config.secret_key,
        )
    
    def _top_k_filtering(self, logits, top_k: int):
        """Filter logits to keep only top-k tokens."""
        import torch
        
        if top_k <= 0:
            return logits
        
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_value = values[:, -1].unsqueeze(-1)
        return torch.where(
            logits < min_value,
            torch.full_like(logits, float('-inf')),
            logits
        )
    
    def _top_p_filtering(self, logits, top_p: float):
        """Filter logits using nucleus (top-p) sampling."""
        import torch
        
        if top_p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        
        return logits


@dataclass
class GenerationOutput:
    """Output from watermarked text generation.
    
    Attributes:
        text: Generated text (excluding prompt)
        full_text: Full text including prompt
        prompt: Original prompt
        token_ids: List of generated token IDs
        green_count: Number of green tokens generated
        total_tokens: Total tokens generated
        z_score: Watermark detection z-score
        is_watermarked: Whether watermark was applied
        secret_key: Secret key used for verification
    """
    text: str
    full_text: str
    prompt: str
    token_ids: List[int]
    green_count: int
    total_tokens: int
    z_score: float
    is_watermarked: bool
    secret_key: List[int]
    
    @property
    def green_ratio(self) -> float:
        """Proportion of green tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.green_count / self.total_tokens
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "full_text": self.full_text,
            "prompt": self.prompt,
            "token_ids": self.token_ids,
            "green_count": self.green_count,
            "total_tokens": self.total_tokens,
            "green_ratio": self.green_ratio,
            "z_score": self.z_score,
            "is_watermarked": self.is_watermarked,
            "secret_key": self.secret_key,
        }