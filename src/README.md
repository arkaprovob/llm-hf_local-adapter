# HuggingFace Local Adapter - Organized Package Structure

This directory contains a well-organized implementation of the HuggingFace Local Adapter for LiteLLM with clean separation of concerns.

## Package Structure

```
src/
├── __init__.py                    # Main package exports
├── hf_local_adapter/
│   ├── __init__.py               # Core package exports
│   ├── adapter.py                # Main HuggingFaceLocalAdapterV2 class
│   ├── config/
│   │   ├── __init__.py
│   │   └── model_config.py       # ModelConfig dataclass
│   ├── formatting/
│   │   ├── __init__.py
│   │   ├── message_formatter.py  # MessageFormatter class
│   │   └── templates.py          # Chat template definitions
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── stopping_criteria.py  # StopOnTokens class
│   │   └── parameters.py         # Generation parameter management
│   ├── factories/
│   │   ├── __init__.py
│   │   └── model_factories.py    # Factory functions for different models
│   └── utils/
│       ├── __init__.py
│       ├── tokenization.py       # Tokenization utilities
│       └── logging.py            # Logging configuration
```

## Key Components

### 1. **ModelConfig** (`config/model_config.py`)
- Dataclass for model initialization configuration
- Handles device settings, quantization, and model-specific kwargs

### 2. **MessageFormatter** (`formatting/message_formatter.py`)
- Handles conversion of messages to prompts
- Supports automatic chat template detection
- Provides model-specific formatting (Llama-2, Mistral, Falcon, etc.)

### 3. **ChatTemplates** (`formatting/templates.py`)
- Collection of static methods for different chat formats
- Easily extensible for new model families

### 4. **GenerationParameterManager** (`generation/parameters.py`)
- Manages generation parameters and stopping criteria
- Handles parameter merging and validation

### 5. **TokenizationUtils** (`utils/tokenization.py`)
- Tokenization operations and context window validation
- Token counting utilities

### 6. **Factory Functions** (`factories/model_factories.py`)
- Pre-configured adapters for popular models
- `create_llama2_adapter()`, `create_mistral_adapter()`, etc.

## Benefits of This Structure

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Maintainability**: Easy to modify individual components without affecting others
3. **Testability**: Each component can be tested independently
4. **Extensibility**: Easy to add new chat templates, stopping criteria, or utilities
5. **Reusability**: Components can be used independently if needed

## Usage Example

```python
from src import HuggingFaceLocalAdapterV2, ModelConfig

# Create configuration
config = ModelConfig(
    model_id="microsoft/phi-2",
    device="cuda:0",
    load_in_4bit=True
)

# Create adapter
adapter = HuggingFaceLocalAdapterV2(
    model_config=config,
    context_window=4096,
    temperature=0.7
)

# Use with LiteLLM
import litellm
litellm.custom_provider_map = [
    {"provider": "hf-local", "custom_handler": adapter}
]
```

## Adding New Features

### Adding a New Chat Template

1. Add a new static method to `ChatTemplates` in `formatting/templates.py`
2. Update `MessageFormatter.format_messages()` to detect and use the new template

### Adding New Generation Parameters

1. Update `GenerationParameterManager` in `generation/parameters.py`
2. Add the new parameters to the default handling logic

### Adding New Utilities

1. Create a new module in the `utils/` directory
2. Export it from `utils/__init__.py`
3. Use it in the main adapter class as needed 