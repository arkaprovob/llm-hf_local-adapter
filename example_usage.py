"""
Example usage of the HuggingFace Local Adapter for LiteLLM
"""

import litellm
from litellm import completion, acompletion
from huggingface_local_adapter import HuggingFaceLocalAdapter, create_huggingface_adapter
import asyncio


def example_basic_usage():
    """Basic usage example with direct instantiation."""
    print("=== Basic Usage Example ===")
    
    # Create adapter instance
    adapter = HuggingFaceLocalAdapter(
        model_id="microsoft/phi-2",
        device="cuda:0",
        cache_dir="/path/to/cache",  # Optional
        load_in_4bit=False,  # Set to True for 4-bit quantization
    )
    
    # Register the adapter with LiteLLM
    litellm.custom_provider_map = [
        {"provider": "huggingface-local", "custom_handler": adapter}
    ]
    
    # Use the adapter
    response = completion(
        model="huggingface-local/phi-2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")


def example_streaming():
    """Streaming example."""
    print("\n=== Streaming Example ===")
    
    # Create adapter
    adapter = create_huggingface_adapter(
        model_id="microsoft/phi-2",
        device="cuda:0"
    )
    
    # Register adapter
    litellm.custom_provider_map = [
        {"provider": "huggingface-local", "custom_handler": adapter}
    ]
    
    # Stream response
    response = completion(
        model="huggingface-local/phi-2",
        messages=[
            {"role": "user", "content": "Write a short story about a robot."}
        ],
        stream=True,
        temperature=0.8,
        max_tokens=200
    )
    
    print("Streaming response: ", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


async def example_async_usage():
    """Async usage example."""
    print("\n=== Async Usage Example ===")
    
    # Create adapter
    adapter = HuggingFaceLocalAdapter(
        model_id="microsoft/phi-2",
        device="cuda:0"
    )
    
    # Register adapter
    litellm.custom_provider_map = [
        {"provider": "huggingface-local", "custom_handler": adapter}
    ]
    
    # Async completion
    response = await acompletion(
        model="huggingface-local/phi-2",
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        temperature=0.5,
        max_tokens=150
    )
    
    print(f"Async Response: {response.choices[0].message.content}")


async def example_async_streaming():
    """Async streaming example."""
    print("\n=== Async Streaming Example ===")
    
    # Create adapter
    adapter = create_huggingface_adapter(
        model_id="microsoft/phi-2",
        device="cuda:0"
    )
    
    # Register adapter
    litellm.custom_provider_map = [
        {"provider": "huggingface-local", "custom_handler": adapter}
    ]
    
    # Async stream
    response = await acompletion(
        model="huggingface-local/phi-2",
        messages=[
            {"role": "user", "content": "List 5 interesting facts about space."}
        ],
        stream=True,
        temperature=0.7,
        max_tokens=200
    )
    
    print("Async streaming response: ", end="", flush=True)
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def example_multiple_models():
    """Example with multiple models."""
    print("\n=== Multiple Models Example ===")
    
    # Create adapters for different models
    phi2_adapter = create_huggingface_adapter(
        model_id="microsoft/phi-2",
        device="cuda:0"
    )
    
    # You can create more adapters for different models
    # llama_adapter = create_huggingface_adapter(
    #     model_id="meta-llama/Llama-2-7b-chat-hf",
    #     device="cuda:1",
    #     load_in_4bit=True
    # )
    
    # Register multiple adapters
    litellm.custom_provider_map = [
        {"provider": "phi2-local", "custom_handler": phi2_adapter},
        # {"provider": "llama-local", "custom_handler": llama_adapter},
    ]
    
    # Use different models
    response1 = completion(
        model="phi2-local/phi-2",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50
    )
    print(f"Phi-2 says: {response1.choices[0].message.content}")
    
    # response2 = completion(
    #     model="llama-local/llama-2-7b",
    #     messages=[{"role": "user", "content": "Hello!"}],
    #     max_tokens=50
    # )
    # print(f"Llama-2 says: {response2.choices[0].message.content}")


def example_with_custom_params():
    """Example with custom generation parameters."""
    print("\n=== Custom Parameters Example ===")
    
    # Create adapter
    adapter = HuggingFaceLocalAdapter(
        model_id="microsoft/phi-2",
        device="cuda:0"
    )
    
    # Register adapter
    litellm.custom_provider_map = [
        {"provider": "huggingface-local", "custom_handler": adapter}
    ]
    
    # Use with custom parameters
    response = completion(
        model="huggingface-local/phi-2",
        messages=[
            {"role": "user", "content": "Generate a creative product name for a smart water bottle."}
        ],
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        max_tokens=50,
        repetition_penalty=1.2,
        # Additional HuggingFace parameters can be passed via optional_params
        optional_params={
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }
    )
    
    print(f"Creative response: {response.choices[0].message.content}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_streaming()
    
    # Run async examples
    asyncio.run(example_async_usage())
    asyncio.run(example_async_streaming())
    
    example_multiple_models()
    example_with_custom_params() 