"""Model architecture definitions for the Lingmao Moyun training system.

2026-04-05: Added RoPE, SwiGLU, GQA, and optional MoE layers.
These are modern LLM architecture components used by Qwen3, DeepSeek-V3, LLaMA-3, etc.
"""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - fallback for test environments
    torch = None
    F = None

    class nn:  # type: ignore[no-redef]
        class Module:
            pass

        class Embedding:
            pass

        class Linear:
            pass

        class Dropout:
            pass

        class TransformerEncoderLayer:
            pass

        class TransformerEncoder:
            pass

        class CrossEntropyLoss:
            pass


from src.config import (
    DEFAULT_D_MODEL,
    DEFAULT_DIM_FEEDFORWARD,
    DEFAULT_DROPOUT,
    DEFAULT_MAX_LEN,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Positional Encoding (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (identical to "Attention is All You Need").

    Args:
        d_model: Embedding dimension.
        dropout: Dropout probability.
        max_len: Maximum sequence length to pre-compute encodings for.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = DEFAULT_DROPOUT,
        max_len: int = DEFAULT_MAX_LEN,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Modern Components (2025-2026 LLM architecture)
# ─────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) — Su et al., 2021.

    Used by LLaMA, Qwen, DeepSeek, and virtually all modern LLMs.
    Enables better length extrapolation than sinusoidal or learned PE.

    Args:
        dim: Embedding dimension (per head).
        max_seq_len: Maximum sequence length to pre-compute rotary factors.
        base: Base for the inverse frequency computation (default: 10000).
        rope_scaling: Optional YaRN-style scaling dict, e.g. {"factor": 2, "original_max_len": 4096}.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = DEFAULT_MAX_LEN,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_scaling = rope_scaling

        # Compute inverse frequencies: 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute cos and sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]

        # Apply YaRN-style scaling if configured
        if self.rope_scaling is not None:
            factor = self.rope_scaling.get("factor", 1.0)
            original_max = self.rope_scaling.get("original_max_len", self.max_seq_len)
            # For positions >= original_max: scale frequencies down by factor
            # This extends context by interpolating positions beyond original_max
            scale = (t >= original_max).float().unsqueeze(-1)  # [seq_len, 1] for broadcasting
            freqs = freqs * (1 - scale) + freqs * (1.0 / factor) * scale

        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for the requested sequence length."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        return cos, sin

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input for RoPE application."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: [batch, seq_len, num_heads, head_dim]
            k: [batch, seq_len, num_kv_heads, head_dim]
            cos: [seq_len, head_dim]
            sin: [seq_len, head_dim]

        Returns:
            (rotated_q, rotated_k) with same shape as input
        """
        # cos/sin: [seq_len, head_dim] → [1, seq_len, 1, head_dim] for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network — used by LLaMA, Qwen, Gemma, DeepSeek.

    SwiGLU(x) = SiLU(W₁x) ⊙ (W₂x)  with optional W₃ gate projection.
    Provides 5-15% quality improvement over ReLU-based FFNs.

    Args:
        d_model: Model dimension.
        dim_feedforward: FFN hidden dimension.
    """

    def __init__(self, d_model: int, dim_feedforward: int) -> None:
        super().__init__()
        # W₁, W₂, W₃ projections (SwiGLU uses 3 linear layers)
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: act(w1(x)) ⊙ w3(x) — both have dim_feedforward dim — then down-project
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    """Mixture of Experts layer — used by DeepSeek-V3, Qwen3-MoE, Gemma-3.

    Routes each token to top-k experts, dramatically reducing active parameters.

    Args:
        d_model: Model dimension.
        dim_feedforward: FFN hidden dimension per expert.
        num_experts: Total number of expert FFNs.
        top_k: Number of experts to activate per token.
        router_bias: Whether to use bias in the router linear layer.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: maps hidden state to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=router_bias)

        # Expert FFNs (each is an independent SwiGLU FFN)
        self.experts = nn.ModuleList(
            SwiGLUFFN(d_model, dim_feedforward) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with top-k routing.

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model] — weighted sum of top-k expert outputs
        """
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [B*L, d_model]
        num_tokens = B * L

        # Router: [num_tokens, num_experts] → [num_tokens, top_k] expert indices + weights
        router_logits = self.router(x_flat)  # [num_tokens, num_experts]
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1).to(x.dtype)  # [num_tokens, top_k]

        # Accumulate expert contributions per token
        out = torch.zeros_like(x_flat)  # [num_tokens, d_model]

        for k_idx in range(self.top_k):
            expert_weights = weights[:, k_idx]  # [num_tokens]
            expert_ids = indices[:, k_idx]     # [num_tokens]

            for expert_id in range(self.num_experts):
                token_mask = (expert_ids == expert_id)  # [num_tokens] bool
                if not token_mask.any():
                    continue
                expert_out = self.experts[expert_id](x_flat[token_mask])  # [active, d_model]
                out[token_mask] += expert_out * expert_weights[token_mask].unsqueeze(-1)

        return out.view(B, L, D)


class ModernAttention(nn.Module):
    """Modern causal attention with Flash Attention (via PyTorch SDPA), RoPE, and GQA.

    Uses F.scaled_dot_product_attention which automatically dispatches to Flash Attention
    on GPUs with compute capability >= 7.0 ( Volta+ ).

    Args:
        d_model: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads. If None, uses multi-head attention (MHA).
                      If < num_heads, uses Grouped Query Attention (GQA).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE pre-computation.
        rope_scaling: Optional YaRN-style RoPE scaling dict.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = DEFAULT_MAX_LEN,
        rope_scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.num_kv_heads <= self.num_heads
        ), "num_kv_heads cannot exceed num_heads"
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        # Projections
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)

        self.dropout = dropout
        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
        )

    def _naive_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Naive attention fallback for CPU or unsupported hardware.

        Computes standard scaled dot-product attention without Flash Attention.
        O(N²) memory complexity — use only as fallback.
        """
        B, L, H, D = q.shape
        scale = 1.0 / math.sqrt(D)

        # Compute attention scores: [B, H, L, L]
        q_t = q.transpose(1, 2)  # [B, H, L, D]
        k_t = k.transpose(1, 2)  # [B, H, L, D]
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, H, L, L]

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, L] or [B, H, L, L]
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        v_t = v.transpose(1, 2)  # [B, H, L, D]
        output = torch.matmul(attn_weights, v_t)  # [B, H, L, D]
        return output.transpose(1, 2)  # [B, L, H, D]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Flash Attention via PyTorch SDPA.

        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask (e.g., for padding).
                           Shape: [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len].
            is_causal: If True, apply causal masking automatically via SDPA.

        Returns:
            [batch_size, seq_len, d_model]
        """
        B, L, D = x.shape

        # Q, K, V projections
        q = self.q_proj(x)  # [B, L, num_heads * head_dim]
        k = self.k_proj(x)  # [B, L, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [B, L, num_kv_heads * head_dim]

        # Reshape for multi-head format: [B, L, H, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        cos, sin = self.rotary(L, device=x.device)
        q, k = self.rotary.apply_rotary_qk(q, k, cos, sin)

        # For GQA: expand K and V to match Q heads via repeat
        if self.num_kv_heads < self.num_heads:
            # Repeat K and V heads to match Q heads
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=2)  # [B, L, num_heads, head_dim]
            v = v.repeat_interleave(reps, dim=2)  # [B, L, num_heads, head_dim]

        # SDPA with Flash Attention: dispatches to flash_attn on CUDA if available
        # Falls back to naive attention on CPU or unsupported hardware
        if q.device.type == "cuda":
            try:
                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    elif attention_mask.dim() == 3:
                        attention_mask = attention_mask.unsqueeze(1)

                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True,
                ):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attention_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=is_causal,
                    )
            except RuntimeError:
                # Fallback for CPU or unsupported hardware
                attn_output = self._naive_attention(q, k, v, attention_mask, is_causal)
        else:
            # CPU: use naive attention
            attn_output = self._naive_attention(q, k, v, attention_mask, is_causal)

        # Reshape and project output
        attn_output = attn_output.contiguous().view(B, L, -1)  # [B, L, num_heads * head_dim]
        return self.o_proj(attn_output)


class ModernTransformerBlock(nn.Module):
    """Single transformer layer with modern components: RoPE, GQA, SwiGLU/MoE.

    Architecture used by Qwen3, DeepSeek-V3, LLaMA-3, Gemma-3.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads (None = MHA, < num_heads = GQA).
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
        use_moe: If True, use MoE FFN instead of SwiGLU.
        num_experts: Number of experts in MoE layer.
        top_k: Top-k routing in MoE.
        activation: Activation function ("silu" or "gelu").
        max_seq_len: Maximum sequence length for RoPE.
        rope_scaling: Optional YaRN-style RoPE scaling.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = "silu",
        max_seq_len: int = DEFAULT_MAX_LEN,
        rope_scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.attention = ModernAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
        )
        self.attention_norm = nn.RMSNorm(d_model)  # Pre-norm (like GPT/NormFormer)

        if use_moe:
            self.feed_forward = MoELayer(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                num_experts=num_experts,
                top_k=top_k,
            )
        else:
            self.feed_forward = SwiGLUFFN(d_model=d_model, dim_feedforward=dim_feedforward)

        self.ffn_norm = nn.RMSNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm transformer: Norm(x + Attn(x))
        x = x + self.dropout_layer(self.attention(self.attention_norm(x), attention_mask))
        # Norm(x + FFN(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Original SimpleTransformer (backward compatible)
# ─────────────────────────────────────────────────────────────────────────────


class SimpleTransformer(nn.Module):
    """Causal transformer language model.

    Two modes available:
    - "legacy": Uses sinusoidal PE + PyTorch TransformerEncoderLayer
    - "modern": Uses RoPE + SwiGLU/GQA/MoE + Flash Attention (via PyTorch SDPA)

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: Hidden dimension in the feed-forward sub-layer.
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encodings.
        mode: "legacy" (default) or "modern".
        num_kv_heads: Number of KV heads for GQA (modern mode only).
        use_moe: Use MoE FFN instead of SwiGLU (modern mode only).
        num_experts: Number of experts in MoE (modern mode only).
        top_k: Top-k routing in MoE (modern mode only).
        rope_scaling: Optional YaRN-style RoPE scaling dict (modern mode only).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
        max_len: int = DEFAULT_MAX_LEN,
        mode: str = "legacy",
        # Modern mode options
        num_kv_heads: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        rope_scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        if mode == "legacy":
            # ── Legacy path: sinusoidal PE + PyTorch TransformerEncoder ──
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        elif mode == "modern":
            # ── Modern path: RoPE + SwiGLU/MoE + Flash Attention ──
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.transformer_blocks = nn.ModuleList([
                ModernTransformerBlock(
                    d_model=d_model,
                    num_heads=nhead,
                    num_kv_heads=num_kv_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                    max_seq_len=max_len,
                    rope_scaling=rope_scaling,
                )
                for _ in range(num_layers)
            ])
            self.norm = nn.RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'legacy' or 'modern'.")

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

        self._init_parameters()

    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass.

        Args:
            src: LongTensor of shape [batch_size, seq_len].
            attention_mask: Optional attention mask for modern mode.

        Returns:
            FloatTensor of shape [batch_size, seq_len, vocab_size].
        """
        if self.mode == "legacy":
            src = self.embedding(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
        else:
            src = self.embedding(src)
            for block in self.transformer_blocks:
                src = block(src, attention_mask=attention_mask)
            output = self.norm(src)

        output = self.output_layer(output)
        return output
