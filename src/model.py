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
    USE_FLASH_ATTENTION,
    USE_SLIDING_WINDOW,
    SWA_WINDOW_SIZE,
    USE_WEIGHT_TYING,
    GRADIENT_CHECKPOINTING_CHUNKS,
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
    Includes auxiliary load-balancing loss (DeepSeek-V3 style) to prevent expert collapse.

    Args:
        d_model: Model dimension.
        dim_feedforward: FFN hidden dimension per expert.
        num_experts: Total number of expert FFNs.
        top_k: Number of experts to activate per token.
        router_bias: Whether to use bias in the router linear layer.
        aux_loss_coef: Coefficient for the auxiliary load-balancing loss (0.0 = no aux loss).
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_bias: bool = False,
        aux_loss_coef: float = 0.01,  # DeepSeek-V3 style load-balancing loss
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        # Router: maps hidden state to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=router_bias)

        # Expert FFNs (each is an independent SwiGLU FFN)
        self.experts = nn.ModuleList(
            SwiGLUFFN(d_model, dim_feedforward) for _ in range(num_experts)
        )

        # Register buffer for running load factor (for logging/monitoring)
        self.register_buffer("expert_load", torch.zeros(num_experts), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with top-k routing and auxiliary load-balancing loss.

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            Tuple of:
            - [batch_size, seq_len, d_model] — weighted sum of top-k expert outputs
            - [1] — auxiliary load-balancing loss (scalar)
        """
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [B*L, d_model]
        num_tokens = B * L

        # Router: [num_tokens, num_experts] → [num_tokens, top_k] expert indices + weights
        router_logits = self.router(x_flat)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1).to(x.dtype)  # [num_tokens, num_experts]

        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        # weights are already probabilities from softmax, no need to re-softmax
        weights = weights.to(x.dtype)  # [num_tokens, top_k]

        # Accumulate expert contributions per token
        out = torch.zeros_like(x_flat)  # [num_tokens, d_model]

        # Compute load-balancing auxiliary loss (DeepSeek-V3 style)
        # This encourages equal utilization across experts to prevent expert collapse
        if self.aux_loss_coef > 0 and self.training:
            # Fraction of tokens routed to each expert (before top-k masking)
            token_load = router_probs.mean(dim=0)  # [num_experts]
            # Fraction of routing weight assigned to each expert (after top-k)
            weight_load = weights.sum(dim=0) / self.top_k  # [num_experts]
            # Load-balancing loss: penalize unequal distribution
            # Loss = sum(load_factor^2) encourages all experts to have similar load
            aux_loss = (token_load * weight_load).sum() * self.num_experts
            aux_loss = aux_loss * self.aux_loss_coef
        else:
            aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        for k_idx in range(self.top_k):
            expert_weights = weights[:, k_idx]  # [num_tokens]
            expert_ids = indices[:, k_idx]     # [num_tokens]

            for expert_id in range(self.num_experts):
                token_mask = (expert_ids == expert_id)  # [num_tokens] bool
                if not token_mask.any():
                    continue
                expert_out = self.experts[expert_id](x_flat[token_mask])  # [active, d_model]
                out[token_mask] += expert_out * expert_weights[token_mask].unsqueeze(-1)

        # Update running load factor for monitoring
        with torch.no_grad():
            self.expert_load = token_load.detach() if self.training else self.expert_load

        return out.view(B, L, D), aux_loss


class ModernAttention(nn.Module):
    """Modern causal attention with Flash Attention (via PyTorch SDPA), RoPE, GQA, KV Cache, and SWA.

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
        attention_dropout: Dropout probability for attention weights.
        use_flash_attention: Use Flash Attention via PyTorch SDPA.
        use_sliding_window: Enable sliding window attention (SWA).
        window_size: Window size for SWA (attention only attends to tokens within window).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = DEFAULT_MAX_LEN,
        rope_scaling: Optional[dict] = None,
        attention_dropout: float = 0.05,  # Modern LLMs use ~0.05 attention dropout
        use_flash_attention: bool = USE_FLASH_ATTENTION,
        use_sliding_window: bool = USE_SLIDING_WINDOW,
        window_size: int = SWA_WINDOW_SIZE,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size

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

        self.dropout = nn.Dropout(dropout)  # Module-level dropout layer
        self.attention_dropout = attention_dropout  # Dropout prob for attention weights
        self.use_flash_attention = use_flash_attention
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
        if self.training and self.attention_dropout > 0:
            attn_weights = self.dropout(attn_weights)

        v_t = v.transpose(1, 2)  # [B, H, L, D]
        output = torch.matmul(attn_weights, v_t)  # [B, H, L, D]
        return output.transpose(1, 2)  # [B, L, H, D]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """Forward pass with Flash Attention via PyTorch SDPA, with optional KV Cache and SWA.

        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask (e.g., for padding).
                           Shape: [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len].
            is_causal: If True, apply causal masking automatically via SDPA.
            use_cache: If True, return past_key_values for KV caching in generation.
            past_key_values: Optional tuple of (k, v) from previous forward passes for KV cache.

        Returns:
            Tuple of:
            - [batch_size, seq_len, d_model] — attention output
            - Optional tuple of (k, v) tensors for KV cache (if use_cache=True)
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

        # Handle KV Cache: concatenate past and present K/V
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=1)  # [B, past_len + L, num_kv_heads, head_dim]
            v = torch.cat([past_v, v], dim=1)  # [B, past_len + L, num_kv_heads, head_dim]

        # For GQA: expand K and V to match Q heads via repeat
        if self.num_kv_heads < self.num_heads:
            # Repeat K and V heads to match Q heads
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=2)  # [B, seq, num_heads, head_dim]
            v = v.repeat_interleave(reps, dim=2)  # [B, seq, num_heads, head_dim]

        # Return cache for next forward pass
        present_key_values = (k, v) if use_cache else None

        # Sliding Window Attention: mask out tokens beyond window_size
        attn_mask = attention_mask
        if self.use_sliding_window and L > 1:
            # Create sliding window mask: each position only attends to window_size tokens before
            seq_len = k.shape[1]
            if self.use_flash_attention:
                # For SDPA: create a causal mask that respects window_size
                # Positions beyond window_size get -inf attention
                import torch
                casual_for_swa = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
                )
                # Also mask out tokens earlier than window_size (for SWA with left-side window)
                window_ones = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
                col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
                row_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
                within_window = (col_indices <= row_indices) & (col_indices >= row_indices - self.window_size + 1)
                swa_mask = ~(within_window & casual_for_swa)
                # Combine with existing attention mask
                if attention_mask is not None:
                    attn_mask = attention_mask & swa_mask
                else:
                    attn_mask = swa_mask

        # SDPA with Flash Attention: PyTorch 2.0+ auto-dispatches to flash_attn on CUDA
        # No need for deprecated sdp_kernel context manager
        if self.use_flash_attention:
            try:
                if attn_mask is not None:
                    if attn_mask.dim() == 2:
                        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                    elif attn_mask.dim() == 3:
                        attn_mask = attn_mask.unsqueeze(1)

                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
            except (RuntimeError, AssertionError):
                # Fallback for CPU or unsupported hardware
                attn_output = self._naive_attention(q, k, v, attn_mask, is_causal)
        else:
            attn_output = self._naive_attention(q, k, v, attn_mask, is_causal)

        # Reshape and project output
        attn_output = attn_output.contiguous().view(B, L, -1)  # [B, L, num_heads * head_dim]
        return self.o_proj(attn_output), present_key_values


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
        attention_dropout: float = 0.05,  # Modern LLMs use ~0.05
        use_sliding_window: bool = USE_SLIDING_WINDOW,
        window_size: int = SWA_WINDOW_SIZE,
    ) -> None:
        super().__init__()

        self.attention = ModernAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            attention_dropout=attention_dropout,
            use_sliding_window=use_sliding_window,
            window_size=window_size,
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
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        # Pre-norm transformer: Norm(x + Attn(x))
        attn_out, present = self.attention(
            self.attention_norm(x), attention_mask, use_cache=use_cache, past_key_values=past_key_values
        )
        x = x + self.dropout_layer(attn_out)
        # Norm(x + FFN(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, present


# ─────────────────────────────────────────────────────────────────────────────
# Original SimpleTransformer (backward compatible)
# ─────────────────────────────────────────────────────────────────────────────


class SimpleTransformer(nn.Module):
    """Causal transformer language model.

    Two modes available:
    - "legacy": Uses sinusoidal PE + PyTorch TransformerEncoderLayer
    - "modern": Uses RoPE + SwiGLU/GQA/MoE + Flash Attention (via PyTorch SDPA)
      (2024-2026 best practice: modern mode is default for new models)

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: Hidden dimension in the feed-forward sub-layer.
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encodings.
        mode: "legacy" or "modern" (default="modern").
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
        mode: str = "modern",  # Changed from "legacy" - modern is the default
        # Modern mode options
        num_kv_heads: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        rope_scaling: Optional[dict] = None,
        # Gradient checkpointing option for memory-efficient training
        use_checkpoint: bool = False,
        # Weight tying
        use_weight_tying: bool = USE_WEIGHT_TYING,
        # Sliding window attention
        use_sliding_window: bool = USE_SLIDING_WINDOW,
        window_size: int = SWA_WINDOW_SIZE,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.use_weight_tying = use_weight_tying
        self._tied_weights = False

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
                    use_sliding_window=use_sliding_window,
                    window_size=window_size,
                )
                for _ in range(num_layers)
            ])
            self.norm = nn.RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'legacy' or 'modern'.")

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

        # Apply weight tying if enabled (after output_layer is created)
        if use_weight_tying and mode == "modern":
            self.tie_weights()

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize model weights using modern best practices (2024-2026).

        Modern LLMs (LLaMA-3, Qwen3, DeepSeek-V3) use:
        - Normal distribution with std ≈ 0.02 for attention projections
        - Scaled initialization for SwiGLU FFN layers (important for stability)
        - Zero init for output projection (helps with early training)
        """
        if self.mode == "modern":
            # Modern initialization: normal with small std
            # Based on code from Qwen/LLaMA which uses std=0.02
            for name, param in self.named_parameters():
                if param.dim() > 1:
                    if "output_layer" in name:
                        # Output projection: small init helps early training
                        nn.init.normal_(param, mean=0.0, std=0.02)
                    elif "w1" in name or "w3" in name:
                        # SwiGLU input projections: scaled init
                        nn.init.normal_(param, mean=0.0, std=0.02)
                    elif "w2" in name:
                        # SwiGLU output projection: even smaller init
                        nn.init.normal_(param, mean=0.0, std=0.01)
                    elif "embedding" in name:
                        nn.init.normal_(param, mean=0.0, std=0.02)
                    else:
                        # Attention projections (q, k, v, o)
                        nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            # Legacy path: xavier_uniform (original behavior)
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def tie_weights(self) -> None:
        """Tie embedding and output layer weights.

        This reduces parameters by ~20% (the output layer shares weights with embedding).
        Used by LLaMA, GPT-2, and most modern LLMs.

        Must be called after output_layer is created.
        """
        if self._tied_weights:
            return
        self.output_layer.weight = self.embedding.weight
        self._tied_weights = True

    def untie_weights(self) -> None:
        """Untie embedding and output layer weights (creates independent copies)."""
        if not self._tied_weights:
            return
        self.output_layer.weight = nn.Parameter(self.output_layer.weight.clone())
        self._tied_weights = False

    def forward(
        self,
        src: torch.Tensor,
        attention_mask=None,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """Forward pass.

        Args:
            src: LongTensor of shape [batch_size, seq_len].
            attention_mask: Optional attention mask for modern mode.
            use_cache: If True, return past_key_values for KV caching in generation.
            past_key_values: Optional tuple of (k, v) from previous forward passes.

        Returns:
            Tuple of:
            - [batch_size, seq_len, vocab_size] — logits output
            - Optional tuple of present_key_values for KV cache (if use_cache=True)
        """
        present = None

        if self.mode == "legacy":
            src = self.embedding(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            if self.use_checkpoint:
                # Use gradient checkpointing for memory-efficient training
                output = torch.utils.checkpoint.checkpoint(
                    self.transformer_encoder, src, use_reentrant=False
                )
            else:
                output = self.transformer_encoder(src)
        else:
            src = self.embedding(src)
            if getattr(self, 'gradient_checkpointing', False):
                # Use checkpoint_sequential for memory-efficient training
                # attention_mask is passed as None during checkpointing (typical for causal attention)
                src = torch.utils.checkpoint.checkpoint_sequential(
                    self.transformer_blocks,
                    GRADIENT_CHECKPOINTING_CHUNKS,
                    src,
                    attention_mask=None,
                )
                present = None
            else:
                new_past_key_values = []
                for i, block in enumerate(self.transformer_blocks):
                    past = past_key_values[i] if past_key_values else None
                    src, present_i = block(
                        src,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        past_key_values=past,
                    )
                    if use_cache:
                        new_past_key_values.append(present_i)
                present = tuple(new_past_key_values) if use_cache else None
            output = self.norm(src)

        output = self.output_layer(output)
        return output, present
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
            if self.use_checkpoint:
                # Use gradient checkpointing for memory-efficient training
                output = torch.utils.checkpoint.checkpoint(
                    self.transformer_encoder, src, use_reentrant=False
                )
            else:
                output = self.transformer_encoder(src)
        else:
            src = self.embedding(src)
            if getattr(self, 'gradient_checkpointing', False):
                # Use checkpoint_sequential for memory-efficient training
                # attention_mask is passed as None during checkpointing (typical for causal attention)
                src = torch.utils.checkpoint.checkpoint_sequential(
                    self.transformer_blocks,
                    GRADIENT_CHECKPOINTING_CHUNKS,
                    src,
                    attention_mask=None,
                )
            else:
                for block in self.transformer_blocks:
                    src = block(src, attention_mask=attention_mask)
            output = self.norm(src)

        output = self.output_layer(output)
        return output
