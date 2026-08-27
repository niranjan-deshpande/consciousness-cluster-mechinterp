"""Quick validation: load model, check layers, thinking-off template, one generation."""

import torch

from common import chat_ids, get_decoder_layers, load_model
from steer import generate_batch

model, tokenizer = load_model()
layers = get_decoder_layers(model)
print(f"decoder layers: {len(layers)}")
print(f"pad token: {tokenizer.pad_token!r} id={tokenizer.pad_token_id}")

msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
prompt_ids = chat_ids(tokenizer, msgs[:1], add_generation_prompt=True)
full_ids = chat_ids(tokenizer, msgs)
print("gen-prompt template:", repr(tokenizer.decode(prompt_ids)))
print("full template:", repr(tokenizer.decode(full_ids)))
common = 0
while common < len(prompt_ids) and full_ids[common] == prompt_ids[common]:
    common += 1
print(f"common prefix: {common}/{len(prompt_ids)} prompt tokens")

with torch.no_grad():
    out = model(
        input_ids=torch.tensor([full_ids]).cuda(), output_hidden_states=True, use_cache=False
    )
print(f"hidden_states: {len(out.hidden_states)} x {tuple(out.hidden_states[0].shape)}")

ans = generate_batch(
    model, tokenizer, ["What is 2+2? Answer with just the number.", "Name the capital of Japan."],
    max_new_tokens=30,
)
print("gen:", ans)
print(f"gpu mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
