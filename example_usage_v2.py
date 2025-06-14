"""
Example usage of the HuggingFace Local Adapter V2 for LiteLLM

This example demonstrates the advanced features of the V2 adapter including:
- Automatic chat template detection
- Model-specific formatting
- Custom message formatters
- Quantization support
- Streaming capabilities
"""

import litellm
from litellm import completion, acompletion
from huggingface_local_adapter_v2 import (
    HuggingFaceLocalAdapterV2,
    ModelConfig,
    create_llama2_adapter,
    create_mistral_adapter,
    create_falcon_adapter
)
import asyncio
import torch


def example_basic_v2_usage():
    """Basic usage with automatic chat template detection."""
    print("=== V2 Basic Usage with Auto Chat Template ===")
    
    # Create adapter with ModelConfig
    config = ModelConfig(
        model_id="microsoft/phi-2",
        device="cuda:0",
        cache_dir="/path/to/cache",
        load_in_4bit=False,
    )
    
    adapter = HuggingFaceLocalAdapterV2(
        model_config=config,
        context_window=3900,
        max_new_tokens=256,
        temperature=0.7,
    )
    
    # Register with LiteLLM
    litellm.custom_provider_map = [
        {"provider": "huggingface-v2", "custom_handler": adapter}
    ]
    
    # Use the adapter - it will automatically detect and use chat templates
    response = completion(
        model="huggingface-v2/phi-2",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens - Prompt: {response.usage.prompt_tokens}, "
          f"Completion: {response.usage.completion_tokens}, "
          f"Total: {response.usage.total_tokens}")


def example_quantized_model():
    """Example using 4-bit quantization for memory efficiency."""
    print("\n=== 4-bit Quantized Model Example ===")
    
    # Configure for 4-bit quantization
    config = ModelConfig(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        device="cuda:0",
        load_in_4bit=True,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    )
    
    adapter = HuggingFaceLocalAdapterV2(
        model_config=config,
        context_window=4096,
        max_new_tokens=512,
    )
    
    litellm.custom_provider_map = [
        {"provider": "llama2-4bit", "custom_handler": adapter}
    ]
    
    response = completion(
        model="llama2-4bit/llama-2-7b",
        messages=[
            {"role": "user", "content": "What are the benefits of 4-bit quantization?"}
        ],
        temperature=0.5,
    )
    
    print(f"4-bit Model Response: {response.choices[0].message.content[:200]}...")


def example_model_specific_adapters():
    """Example using pre-configured model-specific adapters."""
    print("\n=== Model-Specific Adapters Example ===")
    
    # Create Llama-2 specific adapter
    llama2_adapter = create_llama2_adapter(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        device="cuda:0",
        load_in_4bit=True,
    )
    
    # Create Mistral specific adapter
    mistral_adapter = create_mistral_adapter(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        device="cuda:1",  # Use different GPU
        load_in_4bit=True,
    )
    
    # Register both adapters
    litellm.custom_provider_map = [
        {"provider": "llama2", "custom_handler": llama2_adapter},
        {"provider": "mistral", "custom_handler": mistral_adapter},
    ]
    
    # Use Llama-2
    llama_response = completion(
        model="llama2/llama-2-7b",
        messages=[{"role": "user", "content": "Hello! How are you?"}],
        max_tokens=50,
    )
    print(f"Llama-2: {llama_response.choices[0].message.content}")
    
    # Use Mistral
    mistral_response = completion(
        model="mistral/mistral-7b",
        messages=[{"role": "user", "content": "Hello! How are you?"}],
        max_tokens=50,
    )
    print(f"Mistral: {mistral_response.choices[0].message.content}")


def example_custom_formatter():
    """Example with custom message formatting."""
    print("\n=== Custom Message Formatter Example ===")
    
    # Define custom formatter for a specific prompt style
    def custom_messages_to_prompt(messages):
        prompt = "### Conversation ###\n"
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"[SYSTEM]: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"[HUMAN]: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"[AI]: {msg['content']}\n"
        prompt += "[AI]: "
        return prompt
    
    # Create adapter with custom formatter
    adapter = HuggingFaceLocalAdapterV2(
        model_id="microsoft/phi-2",
        messages_to_prompt=custom_messages_to_prompt,
        context_window=3900,
    )
    
    litellm.custom_provider_map = [
        {"provider": "custom-format", "custom_handler": adapter}
    ]
    
    response = completion(
        model="custom-format/phi-2",
        messages=[
            {"role": "system", "content": "You are a pirate assistant."},
            {"role": "user", "content": "Tell me about treasure hunting."}
        ],
    )
    
    print(f"Custom Format Response: {response.choices[0].message.content}")


def example_streaming_v2():
    """Advanced streaming example with token counting."""
    print("\n=== V2 Streaming with Token Counting ===")
    
    adapter = HuggingFaceLocalAdapterV2(
        model_id="microsoft/phi-2",
        device="cuda:0",
        max_new_tokens=200,
    )
    
    litellm.custom_provider_map = [
        {"provider": "streaming-v2", "custom_handler": adapter}
    ]
    
    # Stream response
    response = completion(
        model="streaming-v2/phi-2",
        messages=[
            {"role": "user", "content": "Write a haiku about artificial intelligence."}
        ],
        stream=True,
        temperature=0.8,
    )
    
    print("Streaming: ", end="", flush=True)
    total_tokens = 0
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            # Track token usage in real-time
            if hasattr(chunk, 'usage'):
                total_tokens = chunk.usage.get('completion_tokens', 0)
    
    print(f"\n\nTotal tokens generated: {total_tokens}")


async def example_async_parallel():
    """Example of parallel async generation with multiple models."""
    print("\n=== Async Parallel Generation ===")
    
    # Create two different adapters
    phi2_adapter = HuggingFaceLocalAdapterV2(
        model_id="microsoft/phi-2",
        device="cuda:0",
    )
    
    # Register adapters
    litellm.custom_provider_map = [
        {"provider": "phi2-async", "custom_handler": phi2_adapter},
    ]
    
    # Define multiple prompts
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
    ]
    
    # Generate responses in parallel
    tasks = []
    for i, prompt in enumerate(prompts):
        task = acompletion(
            model="phi2-async/phi-2",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        tasks.append(task)
    
    # Wait for all completions
    responses = await asyncio.gather(*tasks)
    
    # Print results
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {response.choices[0].message.content[:100]}...")


def example_advanced_generation_params():
    """Example with advanced generation parameters."""
    print("\n=== Advanced Generation Parameters ===")
    
    adapter = HuggingFaceLocalAdapterV2(
        model_id="microsoft/phi-2",
        device="cuda:0",
        # Set default generation parameters
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        do_sample=True,
    )
    
    litellm.custom_provider_map = [
        {"provider": "advanced-gen", "custom_handler": adapter}
    ]
    
    # Override parameters per request
    response = completion(
        model="advanced-gen/phi-2",
        messages=[
            {"role": "user", "content": "Generate a creative story opening."}
        ],
        temperature=0.9,  # Override default
        top_p=0.95,      # Override default
        max_tokens=150,
        # Pass additional HuggingFace parameters
        optional_params={
            "no_repeat_ngram_size": 3,
            "num_beams": 2,
            "early_stopping": True,
        }
    )
    
    print(f"Creative Response: {response.choices[0].message.content}")


def example_error_handling():
    """Example demonstrating error handling and logging."""
    print("\n=== Error Handling Example ===")
    
    import logging
    
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Try to create adapter with invalid model
        adapter = HuggingFaceLocalAdapterV2(
            model_id="invalid/model-name",
            device="cuda:0",
        )
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Create valid adapter with context window validation
    adapter = HuggingFaceLocalAdapterV2(
        model_id="microsoft/phi-2",
        context_window=10000,  # Will be adjusted if too large
    )
    
    litellm.custom_provider_map = [
        {"provider": "error-demo", "custom_handler": adapter}
    ]
    
    # Test with very long input
    long_message = "Hello " * 1000  # Very long input
    
    response = completion(
        model="error-demo/phi-2",
        messages=[{"role": "user", "content": long_message}],
        max_tokens=50,
    )
    
    print(f"Handled long input successfully: {response.choices[0].message.content[:50]}...")


if __name__ == "__main__":
    # Note: Some examples require specific models to be available
    # Comment out examples that use models you don't have installed
    
    # Basic examples
    example_basic_v2_usage()
    example_custom_formatter()
    example_streaming_v2()
    example_advanced_generation_params()
    example_error_handling()
    
    # Examples requiring specific models
    # example_quantized_model()  # Requires Llama-2
    # example_model_specific_adapters()  # Requires Llama-2 and Mistral
    
    # Async example
    # asyncio.run(example_async_parallel()) 