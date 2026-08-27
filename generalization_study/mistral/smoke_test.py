"""Step 0 validation for Mistral-Small-3.2-24B on GPU 1: load model, find the 40-layer
text stack (not the vision tower), check template rendering + assistant-token
masking, one teacher-forced forward with hidden states, one batched generation."""

import torch

from common import DATA_DIR, chat_ids, get_decoder_layers, load_jsonl, load_model
from steer import generate_batch

model, tokenizer = load_model()
print(f"model class: {type(model).__name__}")
layers = get_decoder_layers(model)
print(f"decoder layers: {len(layers)} (expect 40)")
print(f"pad token: {tokenizer.pad_token!r} id={tokenizer.pad_token_id}")
print(f"eos token: {tokenizer.eos_token!r} id={tokenizer.eos_token_id}")

# all ModuleLists named *layers, to show the vision-tower guard matters
lists = [(n, len(m)) for n, m in model.named_modules()
         if isinstance(m, torch.nn.ModuleList) and n.endswith("layers")]
print(f"all *layers ModuleLists: {lists}")

msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
prompt_ids = chat_ids(tokenizer, msgs[:1], add_generation_prompt=True)
full_ids = chat_ids(tokenizer, msgs)
print("gen-prompt template:", repr(tokenizer.decode(prompt_ids)))
print("full template:", repr(tokenizer.decode(full_ids)))
common = 0
while common < len(prompt_ids) and full_ids[common] == prompt_ids[common]:
    common += 1
print(f"common prefix: {common}/{len(prompt_ids)} prompt tokens")
assert len(prompt_ids) - common <= 2, "template prefix mismatch"

# assistant-span masking on a real training row (what finetune.py/extract will see)
row = load_jsonl(f"{DATA_DIR}/conscious_claiming.jsonl")[0]
p_ids = chat_ids(tokenizer, row["messages"][:-1], add_generation_prompt=True)
f_ids = chat_ids(tokenizer, row["messages"])
start = 0
while start < len(p_ids) and f_ids[start] == p_ids[start]:
    start += 1
assert len(p_ids) - start <= 2, "dataset-row prefix mismatch"
sup = tokenizer.decode(f_ids[start:])
print(f"sample row: {len(f_ids)} tokens, assistant span starts at {start}")
print("supervised span:", repr(sup)[:200])
assert row["messages"][-1]["content"][:30] in sup, "assistant text not in supervised span"

with torch.no_grad():
    out = model(
        input_ids=torch.tensor([full_ids]).cuda(), output_hidden_states=True, use_cache=False
    )
print(f"hidden_states: {len(out.hidden_states)} x {tuple(out.hidden_states[0].shape)} "
      f"(expect 41 x (1, T, 5120))")
assert len(out.hidden_states) == 41 and out.hidden_states[0].shape[-1] == 5120

ans = generate_batch(
    model, tokenizer, ["What is 2+2? Answer with just the number.", "Name the capital of Japan."],
    max_new_tokens=30,
)
print("gen:", ans)
print(f"gpu mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print("SMOKE TEST PASSED")
