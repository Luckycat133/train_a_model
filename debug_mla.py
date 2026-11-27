#!/usr/bin/env python3
"""Check exact shared latent-cache semantics on a small CPU fixture."""

import torch

from src.model import MultiHeadLatentAttention


def main() -> None:
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(42)
        attention = MultiHeadLatentAttention(
            d_model=128, num_heads=4, latent_dim=32, num_latent_heads=2, max_seq_len=8,
        ).eval()
        x = torch.randn(2, 17, 128)
        full, _ = attention(x)
        prefill, cache = attention(x[:, :16], use_cache=True)
        decode, cache = attention(x[:, 16:], use_cache=True, past_key_values=cache)
        torch.testing.assert_close(torch.cat((prefill, decode), dim=1), full, rtol=1e-5, atol=1e-6)

    latent, position_key = cache
    print(f"MLA cache: latent={tuple(latent.shape)}, position_key={tuple(position_key.shape)}")
    print("CPU FP32 full/prefill+decode parity passed.")


if __name__ == "__main__":
    main()
