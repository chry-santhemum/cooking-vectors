# %%
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer
import plotly.express as px
import gc

def clear_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()
    print("allocated:", torch.cuda.memory_allocated(device)/1e9)
    print("reserved:", torch.cuda.memory_reserved(device)/1e9)

device = "cuda"
gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %%

from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=gpt2.tokenizer,  # type: ignore
    streaming=True,
    max_length=gpt2.cfg.n_ctx,
    add_bos_token=gpt2.cfg.default_prepend_bos,
)

# %%

input_tok = token_dataset[42]['tokens'][:100]
print("======\nContext:\n======\n", gpt2.to_string(input_tok))
_, cache = gpt2.run_with_cache(input_tok, remove_batch_dim=False)

in_layer = 6
out_layer = 11
target_token_pos = 6

emb = cache[f"blocks.{in_layer}.hook_resid_pre"]
print("======\nTarget token:\n======\n", gpt2.to_string(input_tok[target_token_pos:target_token_pos+1]))

gpt2_range = nn.Sequential(*[module for module in gpt2.blocks[in_layer:out_layer+1]])

with torch.no_grad():
    baseline_out = gpt2_range(emb)

# %%

# calculate the jacobian, fixing the other positions in context

from torch.autograd.functional import jacobian

def partial_forward(emb_p):
    full_emb = emb.clone()
    full_emb[:,target_token_pos,:] = emb_p
    return gpt2_range(full_emb)[0,target_token_pos,:]

# Calculate Jacobian only w.r.t position p
jac = jacobian(partial_forward, emb[0,target_token_pos,:])
print(jac.shape)

# %%

U, S, Vh = torch.linalg.svd(jac)

# plot singular values
px.scatter(x=range(len(S)), y=S.cpu().numpy(), title="Singular values of the Jacobian", labels={"x":"index", "y":"singular value"}).show()

# %%

from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id=f"blocks.{in_layer}.hook_resid_pre",  # won't always be a hook point
    device=device,
)

# %%

sae_acts = sae.encode(emb)[:, target_token_pos, :].squeeze()

# get activated features and sort them according to their activation intensity
# TODO: scale them relative to the average intensities, since different features have different average activation levels

features = [(index, sae_acts[index].item()) for index in range(len(sae_acts)) if sae_acts[index].item() > 0]
print(len(features))

features.sort(key = lambda x: x[1], reverse=True)
feature_indices = [index for index, _ in features]

feature_vecs = sae.W_dec[feature_indices[:40], :]

# %%

# plot cosine sim between top singular vectors and SAE vectors

from torch.nn import CosineSimilarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

cosine_sim = cos(Vh[:40,:].unsqueeze(2), feature_vecs.T.unsqueeze(0))
print(cosine_sim.shape)

px.imshow(cosine_sim.detach().cpu().numpy(), title="Cosine similarity between top singular vectors and top SAE features", labels={"x":"SAE feature vectors", "y":"Singular vectors"}).show()

# %%

# Plot the L2 norm of the change in output for a given steering direction at the target token

index, val = features[0]
print(val)

feature_vec = sae.W_dec[index, :]

def plot_steering_effect(vec):
    # TODO: scale vec properly
    steering_strength = torch.linspace(-5*val,5*val,steps=2000).to(device)

    vec_scaled = torch.zeros(len(steering_strength), emb.shape[1], gpt2.cfg.d_model).to(device)
    vec_scaled[:,target_token_pos,:] = torch.outer(steering_strength, vec)

    steered_emb_stacked = emb + vec_scaled

    with torch.no_grad():
        steered_output = gpt2_range(steered_emb_stacked)
    
    diff = torch.linalg.norm(steered_output - baseline_out, dim=(1,2))
    diff_at_token = torch.linalg.norm(steered_output[:,target_token_pos,:] - baseline_out[:,target_token_pos,:], dim=-1)

    px.line(x=steering_strength.cpu().numpy(), y=diff.cpu().numpy(), title="Diff norm vs steering strength", labels={"x":"steering strength", "y":"diff norm"}).show()


# %%

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
