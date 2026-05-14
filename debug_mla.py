#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import MultiHeadLatentAttention
import torch

# Test case
d_model = 128
num_heads = 4
batch_size = 1
seq_len = 16

mla = MultiHeadLatentAttention(
    d_model=d_model,
    num_heads=num_heads,
    latent_dim=64,
    num_latent_heads=2
)

# First step
x1 = torch.randn(batch_size, seq_len, d_model)
print(f"x1 shape: {x1.shape}")

with torch.no_grad():
    out1, cache = mla(x1, use_cache=True)

print(f"out1 shape: {out1.shape}")
print(f"cache k shape: {cache[0].shape}")
print(f"cache v shape: {cache[1].shape}")

# Second step with cache
x2 = torch.randn(batch_size, 1, d_model)
print(f"\nx2 shape: {x2.shape}")

# Let's manually trace through what happens in forward
B, L, D = x2.shape
print(f"\nB, L, D: {B}, {L}, {D}")

q = mla.q_proj(x2)
q = q.view(B, L, mla.num_heads, mla.head_dim)
print(f"q shape: {q.shape}")

k_latent = mla.k_proj_latent(x2)
v_latent = mla.v_proj_latent(x2)
print(f"k_latent shape: {k_latent.shape}")

past_k_latent, past_v_latent = cache
k_latent = torch.cat([past_k_latent, k_latent], dim=1)
v_latent = torch.cat([past_v_latent, v_latent], dim=1)
print(f"k_latent after concat shape: {k_latent.shape}")

k_full = mla.k_latent_to_full(k_latent)
v_full = mla.v_latent_to_full(v_latent)
print(f"k_full shape: {k_full.shape}")

k_full = k_full.view(k_latent.shape[0], k_latent.shape[1], mla.num_heads, mla.head_dim)
v_full = v_full.view(v_latent.shape[0], v_latent.shape[1], mla.num_heads, mla.head_dim)
print(f"k_full after reshape: {k_full.shape}")
print(f"v_full after reshape: {v_full.shape}")

# Now test attention
print(f"\nq shape: {q.shape}")
print(f"k_full shape: {k_full.shape}")
print(f"v_full shape: {v_full.shape}")

# Test naive attention manually
B_q, L_q, H_q, D_q = q.shape
scale = 1.0 / (D_q ** 0.5)
print(f"\nq_t shape: {q.transpose(1,2).shape}")
print(f"k_t shape: {k_full.transpose(1,2).shape}")

scores = torch.matmul(q.transpose(1, 2), k_full.transpose(1, 2).transpose(-2, -1)) * scale
print(f"scores shape: {scores.shape}")

causal_mask = torch.triu(
    torch.ones(L_q, k_full.shape[1], device=q.device, dtype=torch.bool),
    diagonal=1
)
scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

attn_weights = scores.softmax(dim=-1)
v_t = v_full.transpose(1, 2)
output = torch.matmul(attn_weights, v_t)
output = output.transpose(1, 2)
print(f"output shape: {output.shape}")

# Reshape
output = output.contiguous().view(B, L, -1)
print(f"output after view: {output.shape}")
