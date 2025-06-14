import asyncio
import time
import json
from threading import Thread
from typing import Iterator, AsyncIterator, Any, Optional, Union, Dict, List
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from torch import bfloat16

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse, Message, Choices, Usage
from litellm.llms.base import BaseLLM


class HuggingFaceLocalAdapter(CustomLLM):
    """
    Custom LiteLLM adapter for running Hugging Face models locally.
    Supports both streaming and non-streaming completions with async variants.
    """
    
    def __init__(
        self,
        model_id: str = "microsoft/phi-2",
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        
        # Initialize model and tokenizer
        self._initialize_model(load_in_4bit, torch_dtype, **kwargs)
        
    def _initialize_model(self, load_in_4bit: bool, torch_dtype: torch.dtype, **kwargs):
        """Initialize the Hugging Face model and tokenizer."""
        # Configure quantization if needed
        bnb_config = None
        if load_in_4bit:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )
        
        # Load model config
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            config=model_config,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=False,
            padding_side='left',
            cache_dir=self.cache_dir
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        # This is a simple implementation - you may want to customize based on model
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        # Add final assistant prompt
        prompt += "Assistant: "
        return prompt
    
    def _extract_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generation parameters from kwargs."""
        generation_params = {
            "do_sample": kwargs.get("temperature", 1.0) > 0,
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 50),
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Add any additional parameters from optional_params
        optional_params = kwargs.get("optional_params", {})
        generation_params.update(optional_params)
        
        return generation_params
    
    def completion(self, *args, **kwargs) -> ModelResponse:
        """Non-streaming completion."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self.model_id)
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Get generation parameters
        generation_params = self._extract_generation_params(kwargs)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_params
            )
        
        # Decode output
        generated_ids = outputs[0][len(input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate tokens
        prompt_tokens = len(input_ids[0])
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
            model=model,
            object="chat.completion",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        return response
    
    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        """Streaming completion."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self.model_id)
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Get generation parameters
        generation_params = self._extract_generation_params(kwargs)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Add streamer to generation params
        generation_params["streamer"] = streamer
        
        # Start generation in a separate thread
        generation_thread = Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **generation_params
            }
        )
        generation_thread.start()
        
        # Stream tokens
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            
            chunk: GenericStreamingChunk = {
                "text": new_text,
                "is_finished": False,
                "finish_reason": None,
                "usage": {
                    "prompt_tokens": len(input_ids[0]),
                    "completion_tokens": 0,  # Will be updated at the end
                    "total_tokens": len(input_ids[0])
                },
                "index": 0,
                "tool_use": None
            }
            yield chunk
        
        # Wait for generation to complete
        generation_thread.join()
        
        # Send final chunk with usage info
        final_chunk: GenericStreamingChunk = {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(self.tokenizer.encode(generated_text)),
                "total_tokens": len(input_ids[0]) + len(self.tokenizer.encode(generated_text))
            },
            "index": 0,
            "tool_use": None
        }
        yield final_chunk
    
    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """Async non-streaming completion."""
        # Run the synchronous completion in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.completion, *args, **kwargs)
    
    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming completion."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self.model_id)
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Get generation parameters
        generation_params = self._extract_generation_params(kwargs)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Add streamer to generation params
        generation_params["streamer"] = streamer
        
        # Start generation in a separate thread
        generation_thread = Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **generation_params
            }
        )
        generation_thread.start()
        
        # Stream tokens asynchronously
        generated_text = ""
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
                
                chunk: GenericStreamingChunk = {
                    "text": new_text,
                    "is_finished": False,
                    "finish_reason": None,
                    "usage": {
                        "prompt_tokens": len(input_ids[0]),
                        "completion_tokens": 0,
                        "total_tokens": len(input_ids[0])
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
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(self.tokenizer.encode(generated_text)),
                "total_tokens": len(input_ids[0]) + len(self.tokenizer.encode(generated_text))
            },
            "index": 0,
            "tool_use": None
        }
        yield final_chunk


# Factory function to create adapters for different models
def create_huggingface_adapter(
    model_id: str,
    device: str = "cuda:0",
    cache_dir: Optional[str] = None,
    **kwargs
) -> HuggingFaceLocalAdapter:
    """
    Factory function to create a HuggingFace adapter instance.
    
    Args:
        model_id: The Hugging Face model ID (e.g., "microsoft/phi-2")
        device: Device to run the model on (default: "cuda:0")
        cache_dir: Directory to cache the model files
        **kwargs: Additional arguments passed to the adapter
    
    Returns:
        HuggingFaceLocalAdapter instance
    """
    return HuggingFaceLocalAdapter(
        model_id=model_id,
        device=device,
        cache_dir=cache_dir,
        **kwargs
    ) 