# Create configuration
from src import ModelConfig, HuggingFaceLocalAdapterV2

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
    {"provider": "huggingface-local", "custom_handler": adapter}
]

response = litellm.completion(
    model="huggingface-local/phi-2",
    messages=[{"role": "user", "content": "Write a story on kubernetes in anime style"}],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")