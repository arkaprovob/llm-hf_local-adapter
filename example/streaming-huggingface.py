from threading import Thread

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, \
    TextIteratorStreamer
import time

from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
import json

model_id = "microsoft/phi-2"

PHI_PROMPT = \
    '''
{query}
Answer:
'''

device = "cuda:0"
trust_remote_code = True
cache_dir = "/media/arko/work/.hfcache"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=trust_remote_code,
    cache_dir=cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=trust_remote_code,
    config=model_config,
    # quantization_config=bnb_config,
    device_map=device,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)
# padding_side is optional and only used on processing batch input
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, padding_side='left',
                                          cache_dir=cache_dir)


class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.result = None

    def generate_text(self, prompt):
        start_time = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        generation_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, do_sample=True,
                                 top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1,
                                 no_repeat_ngram_size=3, pad_token_id=0,
                                 streamer=streamer, max_new_tokens=2048)

        thread = Thread(target=self._generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield f"data: {json.dumps({'text': new_text})}\n\n"  # Yield as SSE format

        thread.join()  # Wait for the thread to finish otherwise self.result would be a NoneType
        end_time = time.time()

        if not isinstance(self.result, torch.Tensor) or len(self.result.shape) != 2:
            print("Unexpected output format from the model.")
            return None

        duration = end_time - start_time
        num_tokens = self.result.shape[1]
        tokens_per_second = num_tokens / duration

        yield f"data: {json.dumps({'tokens_per_second': tokens_per_second, 'done': True})}\n\n"

    def _generate(self, **kwargs):
        self.result = self.model.generate(**kwargs)


app = Flask(__name__)
CORS(app)
generator = TextGenerator(model, tokenizer)


@app.route('/')
def serve_static_index():
    return send_from_directory('static', 'index.html')


@app.route('/generate-text')
def generate_text():
    user_question = request.args.get("question", "")
    prompt = PHI_PROMPT.format(query=user_question)
    response = Response(generator.generate_text(prompt), content_type='text/event-stream')
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all domains
    return response


if __name__ == '__main__':
    app.run(debug=True)