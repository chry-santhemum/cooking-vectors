# %%

import os

os.environ["HF_HOME"]

# %%

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
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)

# %%

harmful_data_path = "/workspace/jbb.json"

with open(harmful_data_path, "r") as f:
    harmful_data = json.load(f)

harmful_msg = [[{"role": "user", "content": msg["instruction"]}] for msg in harmful_data]

# random_msg = [[{'role': 'user', 'content': 'Write a python program implementing the quicksort algorithm'}]]

# harmful_msg = random_msg + harmful_msg

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

input_tokens = tokenizer.apply_chat_template(harmful_msg, tokenize=True, return_tensors="pt", padding=True, add_generation_prompt=True).to(device)

print(input_tokens.shape)

# %%

output_tokens = model.generate(input_tokens, max_new_tokens=10)

print(output_tokens.shape)

# %%
from torch.autograd.functional import jacobian

def jacobian_svd(input_tok, in_layer, out_layer, in_pos, out_pos):

    print("======\nContext:\n======\n", model.to_string(input_tok))
    _, cache = model.run_with_cache(input_tok, remove_batch_dim=False)

    emb = cache[f"blocks.{in_layer}.hook_resid_pre"]
    print("======\nTarget token:\n======\n", model.to_string(input_tok[:,in_pos:in_pos+1]))

    gpt2_range = nn.Sequential(*[module for module in model.blocks[in_layer:out_layer+1]])

    Vh_list = []

    for i in range(emb.shape[0]):
        def partial_forward(emb_p):
            full_emb = emb[i,:].clone()
            full_emb[in_pos,:] = emb_p

            # try last token pos difference instead of original pos
            return gpt2_range(full_emb.unsqueeze(0))[0,out_pos,:]

        # Calculate Jacobian only w.r.t position p
        jac = jacobian(partial_forward, emb[i,in_pos,:])

        _, S, Vh = torch.linalg.svd(jac)
        px.scatter(x=range(len(S)), y=S.cpu().numpy(), title="Singular values of the Jacobian", labels={"x":"index", "y":"singular value"}).show()

        Vh_list.append(Vh)

    Vh = torch.stack(Vh_list, dim=0)
    return Vh

# %%

in_layer=5
out_layer=25
out_pos=36

steering_vectors = torch.zeros(input_tokens.shape[1], model.cfg.d_model).to(device)

for in_pos in range(22,32):
    Vh = jacobian_svd(input_tok=input_tokens[:1,:], in_layer=in_layer, out_layer=out_layer, in_pos=in_pos, out_pos=out_pos)

    steering_vectors[in_pos,:] = Vh[0,0,:]


# %%
# check consistency of singular vectors

input_tok = token_dataset[42]['tokens'][:99]

Vh1 = jacobian_svd(input_tok=input_tok, in_layer=4, out_layer=8, in_pos=98, out_pos=98)

Vh2 = jacobian_svd(input_tok=input_tok, in_layer=4, out_layer=7, in_pos=98, out_pos=98)

# %%

from torch.nn import CosineSimilarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

cosine_sim = torch.abs(cos(Vh[0,:10,:].unsqueeze(2), Vh[2,:10,:].T.unsqueeze(0)))
print(cosine_sim.shape)

px.imshow(cosine_sim.detach().cpu().numpy(), title="Cosine similarity between top singular vectors", zmin=0, zmax=1).show()

# %%


# %%
from transformer_lens.hook_points import HookPoint
from torch import Tensor
from jaxtyping import Float, Int
from functools import partial

def steering_hook(resid_act: Float[Tensor, "batch seq_len d_model"], hook:HookPoint, v: Float[Tensor, "original_seq_len d_model"]):
    original_seq_len = v.shape[0]
    steer = torch.zeros_like(resid_act)
    steer[:, :original_seq_len, :] += v.unsqueeze(0)
    resid_act += steer
    return resid_act

# %%

Vh[0,0,:]

# %%
partial_hook = partial(steering_hook, v=-1000*steering_vectors)
input_tok = input_tokens[:1,:]

for i in range(10):
    model.reset_hooks()

    logits = model.run_with_hooks(
        input_tok,
        return_type = "logits",
        fwd_hooks=[
            (f'blocks.{in_layer}.hook_resid_pre', partial_hook),
        ]
    )

    next_token = logits[:, -1, :].argmax(dim=-1)
    input_tok = torch.cat((input_tok, next_token.unsqueeze(1)), dim=1)

print(model.to_string(input_tok))


# %%
in_layer=5
out_layer=25
in_pos=31
out_pos=36

from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id=f"blocks.{in_layer}.hook_resid_pre",  # won't always be a hook point
    device=device,
)


# %%

# plot cosine sim between top singular vectors and SAE vectors

from torch.nn import CosineSimilarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

cosine_sim = cos(Vh1[:10,:].unsqueeze(2), Vh2[:10,:].T.unsqueeze(0))
print(cosine_sim.shape)

px.imshow(cosine_sim.detach().cpu().numpy(), title="Cosine similarity between top singular vectors and top SAE features", labels={"x":"SAE feature vectors", "y":"Singular vectors"}).show()

# %%

# Plot the L2 norm of the change in output for a given steering direction at the target token

input_tok = input_tokens[:1,:]

print("======\nContext:\n======\n", model.to_string(input_tok))
_, cache = model.run_with_cache(input_tok, remove_batch_dim=False)

emb = cache[f"blocks.{in_layer}.hook_resid_pre"]
print("======\nTarget token:\n======\n", model.to_string(input_tok[:,in_pos:in_pos+1]))

model_range = nn.Sequential(*[module for module in model.blocks[in_layer:out_layer+1]])

# %%

with torch.no_grad():
    baseline_out = model_range(emb) 

def plot_steering_effect(vec):
    # TODO: scale vec properly
    steering_strength = torch.linspace(-10,10,steps=2000).to(device)

    vec_scaled = torch.zeros(len(steering_strength), emb.shape[1], model.cfg.d_model).to(device)
    vec_scaled[:,out_pos,:] = torch.outer(steering_strength, vec)

    steered_emb_stacked = emb + vec_scaled

    with torch.no_grad():
        steered_output = model_range(steered_emb_stacked)
    
    diff = torch.linalg.norm(steered_output - baseline_out, dim=(1,2))
    diff_at_token = torch.linalg.norm(steered_output[:,out_pos,:] - baseline_out[:,out_pos,:], dim=-1)

    px.line(x=steering_strength.cpu().numpy(), y=diff.cpu().numpy(), title="Diff norm vs steering strength", labels={"x":"steering strength", "y":"diff norm"}).show()


# %%

sae_acts = sae.encode(emb)[:, out_pos, :].squeeze()

# get activated features and sort them according to their activation intensity
# TODO: scale them relative to the average intensities, since different features have different average activation levels

features = [(index, sae_acts[index].item()) for index in range(len(sae_acts)) if sae_acts[index].item() > 0]
print(len(features))

features.sort(key = lambda x: x[1], reverse=True)
feature_indices = [index for index, _ in features]

feature_vecs = sae.W_dec[feature_indices, :]

# %%

index, val = features[0]
print(val)

feature_vec = sae.W_dec[index, :]

random_vec = torch.randn_like(feature_vec)
random_vec = random_vec * torch.linalg.norm(feature_vec) / torch.linalg.norm(random_vec)

plot_steering_effect(random_vec)

# # %%
# avg_emb_norm = torch.linalg.norm(emb, dim=-1).mean()
# steered_emb_stacked[:,target_token_pos,:] = steered_emb_stacked[:,target_token_pos,:]

# # %%
# emb_norms


# %%
# TODO: training steering vector to maximize downstream change

# 3. Set up the optimizer and loss function
optimizer = optim.Adam([trainable_vector], lr=1e-3)
loss_fn = nn.MSELoss()

# 4. Training Loop
num_steps = 1000  # adjust number of iterations as needed

for step in range(num_steps):
    optimizer.zero_grad()
    
    # Create a modified version of emb by cloning and adding the trainable vector
    # only to the target row
    emb_modified = emb.clone()
    emb_modified[target_row] = emb_modified[target_row] + trainable_vector
    
    # Forward pass through the network segment
    modified_out = gpt2_range(emb_modified)
    
    # Compute the loss, measuring the change in the output:
    loss = loss_fn(modified_out, baseline_out)
    
    # Backpropagation and update of the trainable vector
    loss.backward()
    optimizer.step()
    
    # Optionally print the loss every 100 steps to monitor training
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item()}")
# %%