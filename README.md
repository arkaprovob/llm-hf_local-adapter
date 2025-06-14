# HuggingFace Local Adapter for LiteLLM

A custom adapter that enables running HuggingFace models locally through LiteLLM's unified interface. This adapter supports streaming, async operations, and 4-bit quantization.

## Features

- ✅ **Local Model Execution**: Run HuggingFace models on your own hardware
- ✅ **Streaming Support**: Real-time token streaming for better UX
- ✅ **Async Operations**: Full async/await support for better concurrency
- ✅ **4-bit Quantization**: Reduce memory usage with bitsandbytes integration
- ✅ **Multiple Models**: Run multiple models simultaneously
- ✅ **LiteLLM Proxy Compatible**: Works with LiteLLM's OpenAI-compatible proxy server

## Installation with uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-hf-local-adapter
cd llm-hf-local-adapter

# Install with uv (basic dependencies only)
uv pip install -e .

# Or install directly from git
uv pip install git+https://github.com/yourusername/llm-hf-local-adapter
```

### Installation with Optional Dependencies

```bash
# Install with 4-bit quantization support
uv pip install -e ".[quantization]"

# Install with better tokenizer support
uv pip install -e ".[tokenizer]"

# Install with Flask server support
uv pip install -e ".[server]"

# Install all optional dependencies
uv pip install -e ".[all]"
```

### Development Setup with uv

```bash
# Create a virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install in development mode with all dependencies
uv pip install -e ".[all]"

# Note: uv will automatically create a uv.lock file to ensure reproducible installs
# Commit this file to version control for consistent dependencies across environments
```

### Quick Start with uv

```bash
# One-line setup with uv
uv run --with ".[all]" python example_usage.py
```

## Alternative Installation (pip)

If you prefer using standard pip:

1. Install required dependencies:

```bash
pip install litellm torch transformers accelerate
# For 4-bit quantization support:
pip install bitsandbytes
```

2. Clone or download the adapter files:
- `huggingface_local_adapter.py` - Main adapter implementation
- `custom_handler.py` - Model initialization for proxy usage
- `proxy_config.yaml` - Proxy configuration example

## Quick Start

### Basic Usage

```python
import litellm
from archive.huggingface_local_adapter import HuggingFaceLocalAdapter

# Create adapter
adapter = HuggingFaceLocalAdapter(
    model_id="microsoft/phi-2",
    device="cuda:0",
    cache_dir="/path/to/cache"  # Optional
)

# Register with LiteLLM
litellm.custom_provider_map = [
    {"provider": "huggingface-local", "custom_handler": adapter}
]

# Use like any other LiteLLM model
response = litellm.completion(
    model="huggingface-local/phi-2",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

### Streaming Example

```python
response = litellm.completion(
    model="huggingface-local/phi-2",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Async Usage

```python
import asyncio

async def main():
    response = await litellm.acompletion(
        model="huggingface-local/phi-2",
        messages=[{"role": "user", "content": "What is AI?"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Using with LiteLLM Proxy

1. Update `custom_handler.py` with your model configurations:

```python
# custom_handler.py
from archive.huggingface_local_adapter import HuggingFaceLocalAdapter

phi2_adapter = HuggingFaceLocalAdapter(
    model_id="microsoft/phi-2",
    device="cuda:0",
    cache_dir="/your/cache/path",
    load_in_4bit=False  # Set to True for 4-bit quantization
)
```

2. Start the proxy server:

```bash
litellm --config proxy_config.yaml
```

3. Use the OpenAI-compatible endpoint:

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-your-key' \
-d '{
    "model": "phi-2-local",
    "messages": [{"role": "user", "content": "Hello!"}]
}'
```

## Configuration Options

### Adapter Parameters

- `model_id` (str): HuggingFace model identifier
- `device` (str): Device to run on (e.g., "cuda:0", "cpu")
- `cache_dir` (str, optional): Directory to cache model files
- `trust_remote_code` (bool): Whether to trust remote code (default: True)
- `load_in_4bit` (bool): Enable 4-bit quantization (default: False)
- `torch_dtype` (torch.dtype): Model precision (default: torch.bfloat16)

### Generation Parameters

Standard LiteLLM parameters are supported:
- `temperature`: Sampling temperature (0.0 - 2.0)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `max_tokens`: Maximum tokens to generate
- `repetition_penalty`: Penalty for repeated tokens

Additional HuggingFace parameters via `optional_params`:
- `no_repeat_ngram_size`: Prevent n-gram repetitions
- `early_stopping`: Stop when EOS token is generated
- Any other HuggingFace `generate()` parameter

## Supported Models

The adapter works with any HuggingFace causal language model:

- **Small Models**: microsoft/phi-2, EleutherAI/gpt-neo-1.3B
- **Medium Models**: mistralai/Mistral-7B-Instruct-v0.1 (with 4-bit)
- **Large Models**: meta-llama/Llama-2-13b-chat-hf (with 4-bit)

## Memory Optimization

For large models, use 4-bit quantization:

```python
adapter = HuggingFaceLocalAdapter(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    device="cuda:0",
    load_in_4bit=True  # Reduces memory usage by ~75%
)
```

## Prompt Formatting

The adapter includes a basic prompt formatter. For optimal results with chat models, you may want to customize the `_format_messages` method:

```python
class CustomFormattedAdapter(HuggingFaceLocalAdapter):
    def _format_messages(self, messages):
        # Custom formatting for your specific model
        # Example for Llama-2 chat format
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                prompt += f" {msg['content']} "
        return prompt
```

## Advanced V2 Adapter

For more advanced use cases, we provide a V2 adapter (`huggingface_local_adapter_v2.py`) with enhanced features inspired by LlamaIndex's mature implementation:

### V2 Features

- 🚀 **Automatic Chat Template Detection**: Automatically detects and applies model-specific chat templates
- 🎯 **Smart Message Formatting**: Built-in support for Llama-2, Mistral, Falcon, Alpaca, ChatML, and more
- 🔧 **Advanced Configuration**: Comprehensive model configuration with `ModelConfig` dataclass
- 📊 **Better Token Management**: Accurate token counting and context window validation
- 🛡️ **Robust Error Handling**: Comprehensive error handling and logging
- ⚡ **Optimized Performance**: Efficient memory management and generation

### V2 Quick Start

```python
from archive.huggingface_local_adapter_v2 import HuggingFaceLocalAdapterV2, ModelConfig

# Create adapter with automatic chat template detection
config = ModelConfig(
    model_id="microsoft/phi-2",
    device="cuda:0",
    load_in_4bit=True,  # Enable 4-bit quantization
)

adapter = HuggingFaceLocalAdapterV2(
    model_config=config,
    context_window=4096,
    max_new_tokens=512,
)

# Register and use
litellm.custom_provider_map = [
    {"provider": "hf-v2", "custom_handler": adapter}
]

response = litellm.completion(
    model="hf-v2/phi-2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ]
)
```

### Model-Specific Adapters

The V2 adapter includes pre-configured adapters for popular models:

```python
from archive.huggingface_local_adapter_v2 import (
    create_llama2_adapter,
    create_mistral_adapter,
    create_falcon_adapter
)

# Create Llama-2 adapter with proper formatting
llama2_adapter = create_llama2_adapter(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    device="cuda:0",
    load_in_4bit=True
)

# Create Mistral adapter
mistral_adapter = create_mistral_adapter(
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    device="cuda:0"
)
```

### Custom Message Formatting

Define custom formatters for specific use cases:

```python
def custom_formatter(messages):
    # Your custom formatting logic
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"Human: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"AI: {msg['content']}\n"
    prompt += "AI: "
    return prompt

adapter = HuggingFaceLocalAdapterV2(
    model_id="your-model",
    messages_to_prompt=custom_formatter
)
```

See `example_usage_v2.py` for comprehensive examples of V2 features.

## Troubleshooting

### CUDA Out of Memory
- Use 4-bit quantization: `load_in_4bit=True`
- Reduce `max_tokens` parameter
- Use a smaller model
- Clear GPU cache: `torch.cuda.empty_cache()`

### Slow Generation
- Ensure model is on GPU: `device="cuda:0"`
- Check if CPU is being used unintentionally
- Consider using a smaller model

### Import Errors
- Ensure all dependencies are installed
- Check Python path includes the adapter directory

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic completion
- Streaming responses
- Async operations
- Multiple models
- Custom parameters

For advanced features, see `example_usage_v2.py` which demonstrates:
- Automatic chat template detection
- Model-specific adapters
- Custom message formatting
- Parallel async generation
- Advanced generation parameters

## License

This adapter is provided as-is for use with LiteLLM. Please ensure you comply with the licenses of the models you use.

## Contributing

Feel free to submit issues or pull requests to improve the adapter! 