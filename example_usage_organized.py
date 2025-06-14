"""
Example usage of the organized HuggingFace Local Adapter package.

This demonstrates how to use the new clean package structure with separated concerns.
"""

import litellm
from litellm import completion

# Import from the new organized package
from src import (
    HuggingFaceLocalAdapterV2,
    ModelConfig,
    create_llama2_adapter,
    create_mistral_adapter,
    create_falcon_adapter
)


def example_basic_usage():
    """Basic usage with the organized package."""
    print("=== Basic Usage with Organized Package ===")
    
    # Create adapter using ModelConfig
    config = ModelConfig(
        model_id="microsoft/phi-2",
        device="cuda:0",
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
    
    # Use the adapter
    response = completion(
        model="huggingface-v2/phi-2",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )
    
    print(f"Response: {response.choices[0].message.content}")


def example_factory_usage():
    """Example using factory functions."""
    print("\n=== Factory Function Usage ===")
    
    # Create a Llama-2 adapter using factory
    llama2_adapter = create_llama2_adapter(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        device="cuda:0",
        load_in_4bit=True,
        context_window=4096,
    )
    
    # Register with LiteLLM
    litellm.custom_provider_map = [
        {"provider": "llama2", "custom_handler": llama2_adapter}
    ]
    
    # Use the adapter
    response = completion(
        model="llama2/llama-2-7b",
        messages=[
            {"role": "user", "content": "Hello! How are you today?"}
        ],
        max_tokens=100,
    )
    
    print(f"Llama-2 Response: {response.choices[0].message.content}")


def example_custom_formatter():
    """Example with custom message formatter."""
    print("\n=== Custom Formatter Example ===")
    
    # Define custom formatter
    def my_custom_formatter(messages):
        prompt = "### Conversation ###\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"[{role.upper()}]: {content}\n"
        prompt += "[ASSISTANT]: "
        return prompt
    
    # Create adapter with custom formatter
    adapter = HuggingFaceLocalAdapterV2(
        model_id="microsoft/phi-2",
        messages_to_prompt=my_custom_formatter,
        context_window=3900,
    )
    
    litellm.custom_provider_map = [
        {"provider": "custom-phi2", "custom_handler": adapter}
    ]
    
    response = completion(
        model="custom-phi2/phi-2",
        messages=[
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Tell me about treasure."}
        ],
    )
    
    print(f"Custom Format Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    # example_factory_usage()  # Uncomment if you have Llama-2 model
    example_custom_formatter()
    
    print("\n✅ The organized package structure is working correctly!")
    print("📁 Check out the src/ directory to see the clean separation of concerns:") 