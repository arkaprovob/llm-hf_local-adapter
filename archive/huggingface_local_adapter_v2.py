"""
HuggingFace Local Adapter V2 for LiteLLM

A more robust implementation inspired by LlamaIndex's HuggingFaceLLM class,
providing better message formatting, tokenizer handling, and model management.

This adapter follows LiteLLM's CustomLLM interface while incorporating best practices
from LlamaIndex's mature implementation.
"""

import asyncio
import time
import json
import logging
from threading import Thread
from typing import Iterator, AsyncIterator, Any, Optional, Union, Dict, List, Callable, Sequence
from dataclasses import dataclass, field

import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig
)
from torch import bfloat16

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse, Message, Choices, Usage
from litellm.llms.base import BaseLLM

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for HuggingFace model initialization."""
    model_id: str
    device: str = "cuda:0"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: Optional[Union[str, Dict]] = "auto"
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    revision: Optional[str] = None
    
    # Quantization config
    quantization_config: Optional[Dict[str, Any]] = None
    
    # Model-specific kwargs
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Tokenizer-specific kwargs
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for specific token IDs."""
    
    def __init__(self, stop_ids: List[int]):
        self.stop_ids = stop_ids
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class MessageFormatter:
    """
    Handles conversion of messages to prompts with support for various chat templates.
    Inspired by LlamaIndex's approach but adapted for LiteLLM's message format.
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        default_system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the message formatter.
        
        Args:
            tokenizer: The tokenizer to use for chat template application
            default_system_prompt: Default system prompt if none provided
            messages_to_prompt: Custom function to convert messages to prompt
            completion_to_prompt: Custom function to convert completion to prompt
        """
        self.tokenizer = tokenizer
        self.default_system_prompt = default_system_prompt
        self._messages_to_prompt = messages_to_prompt
        self._completion_to_prompt = completion_to_prompt
        
        # Check if tokenizer supports chat templates
        self.supports_chat_template = hasattr(tokenizer, 'apply_chat_template')
        
        logger.info(
            f"MessageFormatter initialized. "
            f"Chat template support: {self.supports_chat_template}"
        )
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of messages to a prompt string.
        
        This method attempts multiple strategies:
        1. Use custom messages_to_prompt if provided
        2. Use tokenizer's apply_chat_template if available
        3. Use model-specific formatting based on known patterns
        4. Fall back to generic formatting
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        # Strategy 1: Use custom formatter if provided
        if self._messages_to_prompt:
            return self._messages_to_prompt(messages)
        
        # Strategy 2: Use tokenizer's chat template if available
        if self.supports_chat_template:
            try:
                # Ensure messages are in the correct format
                formatted_messages = [
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    for msg in messages
                ]
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}. "
                    f"Falling back to model-specific formatting."
                )
        
        # Strategy 3: Model-specific formatting
        model_name_lower = self.tokenizer.name_or_path.lower()
        
        # Llama-2 style formatting
        if "llama-2" in model_name_lower or "llama2" in model_name_lower:
            return self._format_llama2_style(messages)
        
        # Mistral/Mixtral style formatting
        elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
            return self._format_mistral_style(messages)
        
        # Falcon style formatting
        elif "falcon" in model_name_lower:
            return self._format_falcon_style(messages)
        
        # Alpaca style formatting
        elif "alpaca" in model_name_lower:
            return self._format_alpaca_style(messages)
        
        # ChatML style (used by many models including Qwen, Yi, etc.)
        elif any(name in model_name_lower for name in ["qwen", "yi", "deepseek"]):
            return self._format_chatml_style(messages)
        
        # Strategy 4: Generic formatting
        return self._format_generic(messages)
    
    def format_completion(self, text: str) -> str:
        """
        Format a completion prompt.
        
        Args:
            text: The completion text
            
        Returns:
            Formatted prompt string
        """
        if self._completion_to_prompt:
            return self._completion_to_prompt(text)
        
        # Add system prompt if available
        if self.default_system_prompt:
            return f"{self.default_system_prompt}\n\n{text}"
        
        return text
    
    def _format_llama2_style(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in Llama-2 chat style."""
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        prompt = ""
        
        for i, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                if i == 0:  # Only use first system message
                    prompt = f"{B_INST} {B_SYS}{content}{E_SYS}"
            elif role == "user":
                if i == 0 and not prompt:  # No system message
                    prompt = f"{B_INST} {content} {E_INST}"
                else:
                    prompt += f"{B_INST} {content} {E_INST}"
            elif role == "assistant":
                prompt += f" {content} "
        
        return prompt
    
    def _format_mistral_style(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in Mistral instruction style."""
        prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                prompt += f"[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content}</s>"
            elif role == "system":
                # Mistral doesn't have explicit system role, prepend to first user message
                if not prompt:
                    prompt = f"[INST] {content}\n\n"
        
        return prompt
    
    def _format_falcon_style(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in Falcon style."""
        prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        return prompt
    
    def _format_alpaca_style(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in Alpaca instruction style."""
        instruction = ""
        input_text = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                instruction = content
            elif role == "user":
                if instruction:
                    input_text = content
                else:
                    instruction = content
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return prompt
    
    def _format_chatml_style(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in ChatML style."""
        prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def _format_generic(self, messages: List[Dict[str, str]]) -> str:
        """Generic message formatting as fallback."""
        prompt = ""
        
        # Add system message if present
        system_messages = [m for m in messages if m.get("role") == "system"]
        if system_messages:
            prompt += f"System: {system_messages[0].get('content', '')}\n\n"
        elif self.default_system_prompt:
            prompt += f"System: {self.default_system_prompt}\n\n"
        
        # Add conversation history
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        # Add final assistant prompt
        prompt += "Assistant: "
        return prompt


class HuggingFaceLocalAdapterV2(CustomLLM):
    """
    Advanced HuggingFace Local Adapter for LiteLLM.
    
    This adapter provides a robust interface for running HuggingFace models locally
    through LiteLLM's unified API. It includes advanced features like:
    
    - Automatic chat template detection and application
    - Model-specific prompt formatting
    - Efficient memory management with quantization support
    - Streaming support for real-time generation
    - Comprehensive error handling and logging
    - Token counting and usage tracking
    
    The implementation is inspired by LlamaIndex's mature HuggingFaceLLM class
    while maintaining full compatibility with LiteLLM's interface.
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        # Generation parameters
        context_window: int = 4096,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        # Stopping criteria
        stopping_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        # Custom formatters
        messages_to_prompt: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        default_system_prompt: Optional[str] = None,
        # Model and tokenizer instances (optional)
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs
    ):
        """
        Initialize the HuggingFace Local Adapter.
        
        Args:
            model_config: ModelConfig object with model initialization parameters
            model_id: Model ID (used if model_config not provided)
            device: Device to run on (used if model_config not provided)
            cache_dir: Cache directory (used if model_config not provided)
            context_window: Maximum context length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            stopping_ids: List of token IDs that stop generation
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            messages_to_prompt: Custom function to format messages
            completion_to_prompt: Custom function to format completions
            default_system_prompt: Default system prompt
            model: Pre-initialized model (optional)
            tokenizer: Pre-initialized tokenizer (optional)
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Initialize model config
        if model_config is None:
            if model_id is None:
                raise ValueError("Either model_config or model_id must be provided")
            
            model_config = ModelConfig(
                model_id=model_id,
                device=device or "cuda:0",
                cache_dir=cache_dir,
                **kwargs
            )
        
        self.model_config = model_config
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        
        # Generation parameters
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        }
        
        # Initialize model and tokenizer
        self._initialize_model_and_tokenizer(model, tokenizer)
        
        # Set up stopping criteria
        self._setup_stopping_criteria(stopping_ids, eos_token_id, pad_token_id)
        
        # Initialize message formatter
        self.message_formatter = MessageFormatter(
            tokenizer=self.tokenizer,
            default_system_prompt=default_system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt
        )
        
        logger.info(
            f"HuggingFaceLocalAdapterV2 initialized with model: {self.model_config.model_id}"
        )
    
    def _initialize_model_and_tokenizer(
        self, 
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """Initialize the model and tokenizer with proper configuration."""
        
        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            logger.info(f"Loading tokenizer: {self.model_config.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_id,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code,
                revision=self.model_config.revision,
                **self.model_config.tokenizer_kwargs
            )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            logger.info(f"Loading model: {self.model_config.model_id}")
            
            # Prepare model kwargs
            model_kwargs = self.model_config.model_kwargs.copy()
            
            # Set up quantization if requested
            if self.model_config.load_in_4bit or self.model_config.load_in_8bit:
                if self.model_config.quantization_config:
                    quantization_config = BitsAndBytesConfig(
                        **self.model_config.quantization_config
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=self.model_config.load_in_4bit,
                        load_in_8bit=self.model_config.load_in_8bit,
                        bnb_4bit_compute_dtype=self.model_config.torch_dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                model_kwargs["quantization_config"] = quantization_config
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code,
                device_map=self.model_config.device_map,
                max_memory=self.model_config.max_memory,
                offload_folder=self.model_config.offload_folder,
                revision=self.model_config.revision,
                torch_dtype=self.model_config.torch_dtype,
                **model_kwargs
            )
        
        # Validate and adjust context window
        self._validate_context_window()
    
    def _validate_context_window(self):
        """Validate and adjust context window based on model configuration."""
        try:
            model_max_length = self.tokenizer.model_max_length
            if model_max_length and model_max_length < self.context_window:
                logger.warning(
                    f"Reducing context_window from {self.context_window} to "
                    f"model's max length {model_max_length}"
                )
                self.context_window = model_max_length
        except Exception as e:
            logger.warning(f"Could not validate context window: {e}")
    
    def _setup_stopping_criteria(
        self,
        stopping_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        """Set up stopping criteria for generation."""
        stop_ids = stopping_ids or []
        
        # Add EOS token if specified
        if eos_token_id is not None:
            stop_ids.append(eos_token_id)
        elif self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        
        # Create stopping criteria
        if stop_ids:
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
        else:
            self.stopping_criteria = StoppingCriteriaList([])
        
        # Set pad token ID for generation
        if pad_token_id is not None:
            self.generation_config["pad_token_id"] = pad_token_id
        elif self.tokenizer.pad_token_id is not None:
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
    
    def _prepare_generation_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare generation kwargs by merging defaults with provided kwargs."""
        gen_kwargs = self.generation_config.copy()
        
        # Override with any provided kwargs
        for key in ["temperature", "top_p", "top_k", "max_new_tokens", 
                    "repetition_penalty", "do_sample"]:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]
        
        # Add stopping criteria
        gen_kwargs["stopping_criteria"] = self.stopping_criteria
        
        # Add any additional kwargs from optional_params
        optional_params = kwargs.get("optional_params", {})
        gen_kwargs.update(optional_params)
        
        return gen_kwargs
    
    def _tokenize_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize prompt and prepare inputs for model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window
        )
        
        # Move to model device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        return inputs
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def completion(self, *args, **kwargs) -> ModelResponse:
        """
        Generate a completion for the given prompt.
        
        Args:
            *args: Positional arguments (unused, for compatibility)
            **kwargs: Keyword arguments including:
                - messages: List of message dictionaries
                - prompt: Direct prompt string (if messages not provided)
                - temperature, top_p, top_k, etc.: Generation parameters
                
        Returns:
            ModelResponse with generated text and usage statistics
        """
        # Extract messages or prompt
        messages = kwargs.get("messages", [])
        prompt = kwargs.get("prompt", "")
        
        # Format messages into prompt
        if messages:
            prompt = self.message_formatter.format_messages(messages)
        elif not prompt:
            raise ValueError("Either 'messages' or 'prompt' must be provided")
        
        # Tokenize input
        inputs = self._tokenize_prompt(prompt)
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Prepare generation kwargs
        gen_kwargs = self._prepare_generation_kwargs(**kwargs)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        generation_time = time.time() - start_time
        
        # Extract generated tokens (excluding prompt)
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Count tokens
        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        # Create response
        response = ModelResponse(
            id=f"chatcmpl-{int(time.time())}",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=generated_text,
                        role="assistant"
                    )
                )
            ],
            created=int(time.time()),
            model=self.model_config.model_id,
            object="chat.completion",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        logger.debug(
            f"Generated {completion_tokens} tokens in {generation_time:.2f}s "
            f"({completion_tokens/generation_time:.2f} tokens/s)"
        )
        
        return response
    
    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        """
        Stream tokens as they are generated.
        
        Args:
            *args: Positional arguments (unused, for compatibility)
            **kwargs: Keyword arguments including messages and generation parameters
            
        Yields:
            GenericStreamingChunk objects containing generated tokens
        """
        # Extract messages or prompt
        messages = kwargs.get("messages", [])
        prompt = kwargs.get("prompt", "")
        
        # Format messages into prompt
        if messages:
            prompt = self.message_formatter.format_messages(messages)
        elif not prompt:
            raise ValueError("Either 'messages' or 'prompt' must be provided")
        
        # Tokenize input
        inputs = self._tokenize_prompt(prompt)
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Prepare generation kwargs
        gen_kwargs = self._prepare_generation_kwargs(**kwargs)
        gen_kwargs["streamer"] = streamer
        
        # Start generation in separate thread
        generation_thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs}
        )
        generation_thread.start()
        
        # Stream tokens
        generated_text = ""
        token_count = 0
        
        for new_text in streamer:
            generated_text += new_text
            token_count += self._count_tokens(new_text)
            
            chunk: GenericStreamingChunk = {
                "text": new_text,
                "is_finished": False,
                "finish_reason": None,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": token_count,
                    "total_tokens": prompt_tokens + token_count
                },
                "index": 0,
                "tool_use": None
            }
            yield chunk
        
        # Wait for generation to complete
        generation_thread.join()
        
        # Send final chunk
        final_chunk: GenericStreamingChunk = {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count
            },
            "index": 0,
            "tool_use": None
        }
        yield final_chunk
    
    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """
        Async version of completion.
        
        Runs the synchronous completion in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.completion, *args, **kwargs)
    
    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """
        Async streaming completion.
        
        Yields tokens asynchronously as they are generated.
        """
        # Extract messages or prompt
        messages = kwargs.get("messages", [])
        prompt = kwargs.get("prompt", "")
        
        # Format messages into prompt
        if messages:
            prompt = self.message_formatter.format_messages(messages)
        elif not prompt:
            raise ValueError("Either 'messages' or 'prompt' must be provided")
        
        # Tokenize input
        inputs = self._tokenize_prompt(prompt)
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Prepare generation kwargs
        gen_kwargs = self._prepare_generation_kwargs(**kwargs)
        gen_kwargs["streamer"] = streamer
        
        # Start generation in separate thread
        generation_thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs}
        )
        generation_thread.start()
        
        # Stream tokens asynchronously
        generated_text = ""
        token_count = 0
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Get next token with timeout
                new_text = await loop.run_in_executor(
                    None,
                    lambda: next(iter(streamer), None)
                )
                
                if new_text is None:
                    break
                
                generated_text += new_text
                token_count += self._count_tokens(new_text)
                
                chunk: GenericStreamingChunk = {
                    "text": new_text,
                    "is_finished": False,
                    "finish_reason": None,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": token_count,
                        "total_tokens": prompt_tokens + token_count
                    },
                    "index": 0,
                    "tool_use": None
                }
                yield chunk
                
            except StopIteration:
                break
        
        # Wait for generation to complete
        await loop.run_in_executor(None, generation_thread.join)
        
        # Send final chunk
        final_chunk: GenericStreamingChunk = {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count
            },
            "index": 0,
            "tool_use": None
        }
        yield final_chunk


# Convenience functions for creating adapters with common configurations

def create_llama2_adapter(
    model_id: str = "meta-llama/Llama-2-7b-chat-hf",
    **kwargs
) -> HuggingFaceLocalAdapterV2:
    """Create an adapter configured for Llama-2 models."""
    
    def llama2_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Llama-2 specific message formatting."""
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        prompt = ""
        for i, message in enumerate(messages):
            if message["role"] == "system" and i == 0:
                prompt = f"{B_INST} {B_SYS}{message['content']}{E_SYS}"
            elif message["role"] == "user":
                if i == 0 and not prompt:
                    prompt = f"{B_INST} {message['content']} {E_INST}"
                else:
                    prompt += f"{B_INST} {message['content']} {E_INST}"
            elif message["role"] == "assistant":
                prompt += f" {message['content']} "
        
        return prompt
    
    return HuggingFaceLocalAdapterV2(
        model_id=model_id,
        messages_to_prompt=llama2_messages_to_prompt,
        **kwargs
    )


def create_mistral_adapter(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
    **kwargs
) -> HuggingFaceLocalAdapterV2:
    """Create an adapter configured for Mistral models."""
    
    def mistral_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Mistral specific message formatting."""
        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"[INST] {message['content']} [/INST]"
            elif message["role"] == "assistant":
                prompt += f" {message['content']}</s>"
        return prompt
    
    return HuggingFaceLocalAdapterV2(
        model_id=model_id,
        messages_to_prompt=mistral_messages_to_prompt,
        **kwargs
    )


def create_falcon_adapter(
    model_id: str = "tiiuae/falcon-7b-instruct",
    **kwargs
) -> HuggingFaceLocalAdapterV2:
    """Create an adapter configured for Falcon models."""
    
    def falcon_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Falcon specific message formatting."""
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n"
        prompt += "Assistant:"
        return prompt
    
    return HuggingFaceLocalAdapterV2(
        model_id=model_id,
        messages_to_prompt=falcon_messages_to_prompt,
        **kwargs
    ) 