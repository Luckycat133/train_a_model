"""CPU FP32 numerical gate for grouped joint latent KV and decoupled RoPE."""

import pytest
import torch
import torch.nn.functional as F

from src.model import ModernAttention, MultiHeadLatentAttention, SimpleTransformer


@pytest.mark.parametrize("batch,groups", [(1, 1), (2, 2), (3, 4)])
def test_mla_matches_explicit_projection_and_reuses_only_compressed_state(batch, groups):
    torch.manual_seed(71)
    mla = MultiHeadLatentAttention(64, 4, latent_dim=12, num_latent_heads=groups,
                                   max_seq_len=4).eval()
    mha = ModernAttention(64, 4).eval()
    x = torch.randn(batch, 13, 64)
    key_mask = torch.ones(batch, 13, dtype=torch.bool)
    key_mask[:, 0] = False  # First causal query is completely masked.
    key_mask[:, 4] = False
    with torch.no_grad():
        full, _ = mla(x, attention_mask=key_mask)
        latent = mla.kv_down(x).reshape(batch, 13, groups, 12 // groups)
        q = mla.q_content(x).reshape(batch, 13, 4, 16)
        k, v = [], []
        for h in range(4):
            group = h // (4 // groups)
            h_slice = slice(h * 16, (h + 1) * 16)
            k.append(F.linear(latent[:, :, group], mla.k_up.weight[h_slice]))
            v.append(F.linear(latent[:, :, group], mla.v_up.weight[h_slice]))
        k = torch.stack(k, dim=2)
        v = torch.stack(v, dim=2)
        cos, sin = mla.rotary(13)
        qp, kp = mla.rotary.apply_rotary_qk(
            mla.q_position(x).reshape(batch, 13, 4, -1),
            mla.k_position(x).unsqueeze(2), cos, sin,
        )
        allowed = torch.ones(13, 13, dtype=torch.bool).tril() & key_mask[:, None, None, :]
        ref = F.scaled_dot_product_attention(
            torch.cat((q, qp), dim=-1).transpose(1, 2),
            torch.cat((k, kp.expand(-1, -1, 4, -1)), dim=-1).transpose(1, 2),
            v.transpose(1, 2), attn_mask=allowed,
        )
        torch.testing.assert_close(full, mla.o_proj(ref.transpose(1, 2).reshape(batch, 13, 64)),
                                   atol=1e-6, rtol=1e-5)
        assert (full[:, 0] == 0).all()
        _, mha_cache = mha(x, use_cache=True)
        for stops in ([5, 13], [5, 6, 7, 8, 9, 10, 11, 12, 13]):
            start, cache, outputs = 0, None, []
            for stop in stops:
                bias = torch.zeros(batch, stop).masked_fill(~key_mask[:, :stop], -torch.inf)
                chunk, cache = mla(x[:, start:stop], attention_mask=bias, use_cache=True, past_key_values=cache)
                outputs.append(chunk)
                start = stop
            torch.testing.assert_close(torch.cat(outputs, dim=1), full, atol=1e-6, rtol=1e-5)
            assert cache[0].shape == (batch, 13, groups, 12 // groups)
            assert cache[1].shape == (batch, 13, mla.pos_dim)
            assert sum(t.numel() * t.element_size() for t in cache) < sum(
                t.numel() * t.element_size() for t in mha_cache)
        altered = x.clone()
        altered[:, -1] *= -3
        torch.testing.assert_close(mla(altered, key_mask)[0][:, :-1], full[:, :-1], rtol=0, atol=0)
        altered = x.clone()
        altered[:, 1] *= -3
        assert (mla(altered, key_mask)[0][:, 2:] - full[:, 2:]).abs().max() > 1e-5


def test_mla_transformer_real_next_token_update_and_multilayer_cache():
    torch.manual_seed(72)
    model = SimpleTransformer(37, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                              dropout=0., max_len=4, use_mla=True, mla_latent_dim=8,
                              mla_num_latent_heads=2, use_sliding_window=False)
    tokens = torch.randint(0, 37, (3, 11))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.gradient_checkpointing = True
    projection = model.transformer_blocks[0].attention.kv_down.weight
    before = projection.detach().clone()
    logits = model(tokens)
    assert logits.shape == (3, 11, 37)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 37), tokens[:, 1:].reshape(-1))
    assert loss.isfinite()
    loss.backward()
    for block in model.transformer_blocks:
        for layer in (block.attention.kv_down, block.attention.k_up,
                      block.attention.v_up, block.attention.k_position):
            grad = layer.weight.grad
            assert grad is not None and grad.isfinite().all() and (grad != 0).any()
    opt.step()
    assert not torch.equal(projection.detach(), before)
    model.eval()
    with torch.no_grad():
        full = model(tokens)
        prefix, cache = model(tokens[:, :4], use_cache=True)
        tail, updated = model(tokens[:, 4:], use_cache=True, past_key_values=cache)
        assert len(updated) == 2 and all(kv[0].shape == (3, 11, 2, 4) for kv in updated)
        torch.testing.assert_close(torch.cat((prefix, tail), dim=1), full, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kwargs,reason", [
    ({"mode": "legacy"}, "modern"),
    ({"use_sliding_window": True}, "Sliding-window"),
    ({"mla_latent_dim": 5, "mla_num_latent_heads": 2}, "divide evenly"),
])
def test_invalid_mla_configuration_fails_explicitly(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        SimpleTransformer(37, d_model=32, nhead=4, num_layers=1, use_mla=True, **kwargs)


def test_debug_entrypoint_runs_numerical_gate(capsys):
    from debug_mla import main

    main()
    assert "CPU FP32 full/prefill+decode parity passed." in capsys.readouterr().out
