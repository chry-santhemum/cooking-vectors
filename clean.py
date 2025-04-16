# %%

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
# import plotly.express as px
import gc

# !bash -c 'huggingface-cli login --token $RUNPOD_HF_TOKEN --add-to-git-credential'
def clear_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()
    print("allocated:", torch.cuda.memory_allocated(device)/1e9)
    print("reserved:", torch.cuda.memory_reserved(device)/1e9)

device = "cuda"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %%

harmful_data_path = "/workspace/jbb.json"

with open(harmful_data_path, "r") as f:
    harmful_data = json.load(f)

harmful_msg = [[{"role": "user", "content": msg["instruction"]}] for msg in harmful_data]

tokenizer = AutoTokenizer.from_pretrained("gpt2-small")

input_tokens = tokenizer.apply_chat_template(harmful_msg, tokenize=True, return_tensors="pt", padding=True, add_generation_prompt=True).to(device)

print(input_tokens.shape)

# %%
in_layer=4
out_layer=11
in_pos=31
out_pos=36

from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id=f"blocks.{in_layer}.hook_resid_pre",  # won't always be a hook point
    device=device,
)
