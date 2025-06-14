"""
Custom handler file for LiteLLM Proxy
This file creates and exports HuggingFace adapter instances
"""

from huggingface_local_adapter import HuggingFaceLocalAdapter, create_huggingface_adapter

# Create adapter for Phi-2
phi2_adapter = HuggingFaceLocalAdapter(
    model_id="microsoft/phi-2",
    device="cuda:0",
    cache_dir="/path/to/your/cache",  # Update this path
    trust_remote_code=True,
    load_in_4bit=False
)

# Create adapter for Phi-2 with 4-bit quantization
phi2_4bit_adapter = HuggingFaceLocalAdapter(
    model_id="microsoft/phi-2",
    device="cuda:0",
    cache_dir="/path/to/your/cache",  # Update this path
    trust_remote_code=True,
    load_in_4bit=True
)

# Create adapter for Llama (uncomment and configure as needed)
# llama_adapter = HuggingFaceLocalAdapter(
#     model_id="meta-llama/Llama-2-7b-chat-hf",
#     device="cuda:1",  # Use different GPU if available
#     cache_dir="/path/to/your/cache",
#     trust_remote_code=True,
#     load_in_4bit=True  # Recommended for 7B model
# )

# You can also create adapters for other models
# mistral_adapter = create_huggingface_adapter(
#     model_id="mistralai/Mistral-7B-Instruct-v0.1",
#     device="cuda:0",
#     cache_dir="/path/to/your/cache",
#     load_in_4bit=True
# )

# Export a placeholder for llama_adapter to avoid errors in config
class PlaceholderAdapter:
    def __init__(self, name):
        self.name = name
        
    def completion(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name} adapter not configured. Please uncomment and configure in custom_handler.py")
    
    def streaming(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name} adapter not configured. Please uncomment and configure in custom_handler.py")
    
    async def acompletion(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name} adapter not configured. Please uncomment and configure in custom_handler.py")
    
    async def astreaming(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name} adapter not configured. Please uncomment and configure in custom_handler.py")

# Use placeholder if llama_adapter is not configured
llama_adapter = PlaceholderAdapter("Llama") 