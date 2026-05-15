"""Model architecture definitions for the Lingmao Moyun training system.

2026-04-05: Added RoPE, SwiGLU, GQA, and optional MoE layers.
These are modern LLM architecture components used by Qwen3, DeepSeek-V3, LLaMA-3, etc.

2026-05-15: Added torch.compile() support for 2-3x training speedup.
"""

import math
import time
from typing import Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for test environments
    torch = None
    F = None
    TORCH_AVAILABLE = False

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
    USE_MLA,
    MLA_LATENT_DIM,
    MLA_NUM_LATENT_HEADS,
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
        token_load = router_probs.mean(dim=0)  # [num_experts]
        if self.aux_loss_coef > 0 and self.training:
            # Fraction of tokens routed to each expert (before top-k masking)
            # Fraction of routing weight assigned to each expert (after top-k)
            weight_load = weights.sum(dim=0) / self.top_k  # [num_experts]
            # Load-balancing loss: penalize unequal distribution
            # Loss = sum(load_factor^2) encourages all experts to have similar load
            aux_loss = (token_load * weight_load).sum() * self.num_experts
            aux_loss = aux_loss * self.aux_loss_coef
        else:
            aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        # Optimized implementation: process each expert once, handling all tokens assigned to it across all k positions
        for expert_id in range(self.num_experts):
            # Find all positions where this expert is selected in any of the top-k
            token_masks = []
            weight_values = []
            for k_idx in range(self.top_k):
                mask = (indices[:, k_idx] == expert_id)  # [num_tokens] bool
                if mask.any():
                    token_masks.append(mask)
                    weight_values.append(weights[:, k_idx][mask])
            if not token_masks:
                continue
            # Combine all masks for this expert
            combined_mask = torch.stack(token_masks, dim=0).any(dim=0)
            # Get all tokens assigned to this expert
            x_selected = x_flat[combined_mask]
            # Process through expert
            expert_out = self.experts[expert_id](x_selected)
            # Distribute back to original positions with respective weights
            for i, mask in enumerate(token_masks):
                out[mask] += expert_out[combined_mask[mask]] * weight_values[i].unsqueeze(-1)

        # Update running load factor for monitoring
        with torch.no_grad():
            # Ensure we update expert_load regardless of training mode for consistency
            self.expert_load.copy_(token_load.detach())

        return out.view(B, L, D), aux_loss


class MLAttention(nn.Module):
    """Multi-head Latent Attention (MLA) — DeepSeek style KV cache compression.

    核心思想：将 KV Cache 压缩到低维 latent space，大幅减少内存占用（可减少 90%+）。
    设计参考 DeepSeek-V3 的 MLA 架构：
    - Q 保持完整维度不变
    - K 和 V 压缩到低维（如 1/2 或 1/4 维度）
    - 使用独立的压缩投影层
    - 保持与现有 KV Cache 系统完全兼容

    Args:
        d_model: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads for GQA.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE pre-computation.
        rope_scaling: Optional YaRN-style RoPE scaling dict.
        attention_dropout: Dropout probability for attention weights.
        use_flash_attention: Use Flash Attention via PyTorch SDPA.
        use_sliding_window: Enable sliding window attention (SWA).
        window_size: Window size for SWA.
        latent_compression_ratio: 压缩比例，如 2 表示压缩到 1/2，4 表示压缩到 1/4。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = DEFAULT_MAX_LEN,
        rope_scaling: Optional[dict] = None,
        attention_dropout: float = 0.05,
        use_flash_attention: bool = USE_FLASH_ATTENTION,
        use_sliding_window: bool = USE_SLIDING_WINDOW,
        window_size: int = SWA_WINDOW_SIZE,
        latent_compression_ratio: int = 4,  # 默认压缩到 1/4
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert latent_compression_ratio in [2, 4, 8], "latent_compression_ratio must be 2, 4, or 8"
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.latent_compression_ratio = latent_compression_ratio
        
        # 计算压缩后的维度
        self.latent_kv_dim = (self.num_kv_heads * self.head_dim) // latent_compression_ratio
        assert (self.num_kv_heads * self.head_dim) % latent_compression_ratio == 0, \
            "num_kv_heads * head_dim must be divisible by latent_compression_ratio"

        assert (
            self.num_kv_heads <= self.num_heads
        ), "num_kv_heads cannot exceed num_heads"
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        # Q 投影：保持完整维度（与普通 attention 相同）
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        
        # K 和 V 的压缩投影层：将 KV 压缩到低维 latent space
        self.k_compress = nn.Linear(d_model, self.latent_kv_dim, bias=False)
        self.v_compress = nn.Linear(d_model, self.latent_kv_dim, bias=False)
        
        # 从 latent space 恢复到完整维度的投影层（用于 attention 计算）
        self.k_decompress = nn.Linear(self.latent_kv_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_decompress = nn.Linear(self.latent_kv_dim, self.num_kv_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout
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
        """Naive attention fallback for CPU or unsupported hardware."""
        B, L, H, D = q.shape
        scale = 1.0 / math.sqrt(D)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.attention_dropout > 0:
            attn_weights = self.dropout(attn_weights)

        v_t = v.transpose(1, 2)
        output = torch.matmul(attn_weights, v_t)
        return output.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """MLA 前向传播。

        关键流程：
        1. Q 保持完整维度投影
        2. K 和 V 先压缩到 latent space（用于缓存）
        3. 从 latent space 恢复 KV 用于 attention 计算
        4. 支持与标准 KV Cache 兼容

        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask.
            is_causal: If True, apply causal masking.
            use_cache: If True, return compressed KV cache.
            past_key_values: Optional compressed (k_latent, v_latent) from previous passes.

        Returns:
            Tuple of:
            - [batch_size, seq_len, d_model] — attention output
            - Optional tuple of (k_latent, v_latent) for KV cache (compressed!)
        """
        B, L, D = x.shape

        # ── 1. Q 投影：保持完整维度 ──
        q = self.q_proj(x)
        q = q.view(B, L, self.num_heads, self.head_dim)

        # ── 2. K 和 V 压缩到 Latent Space ──
        # 先压缩，这是我们要缓存的形式（内存占用小）
        k_latent = self.k_compress(x)  # [B, L, latent_kv_dim]
        v_latent = self.v_compress(x)  # [B, L, latent_kv_dim]

        # ── 3. 处理 KV Cache（使用压缩后的形式） ──
        if past_key_values is not None:
            past_k_latent, past_v_latent = past_key_values
            k_latent = torch.cat([past_k_latent, k_latent], dim=1)
            v_latent = torch.cat([past_v_latent, v_latent], dim=1)

        # ── 4. 从 Latent Space 恢复 KV 用于 Attention 计算 ──
        k_full = self.k_decompress(k_latent)  # [B, seq_len, num_kv_heads * head_dim]
        v_full = self.v_decompress(v_latent)  # [B, seq_len, num_kv_heads * head_dim]
        
        k_full = k_full.view(B, k_latent.size(1), self.num_kv_heads, self.head_dim)
        v_full = v_full.view(B, v_latent.size(1), self.num_kv_heads, self.head_dim)

        # ── 5. 应用 RoPE ──
        cos, sin = self.rotary(k_latent.size(1), device=x.device)
        # 只对当前的 q 和完整的 k 应用 RoPE
        q, k_full = self.rotary.apply_rotary_qk(q, k_full, cos[-L:], sin[-L:])

        # ── 6. GQA：扩展 KV 匹配 Q heads ──
        if self.num_kv_heads < self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(reps, dim=2)
            v_full = v_full.repeat_interleave(reps, dim=2)

        # ── 7. Sliding Window Attention ──
        attn_mask = attention_mask
        if self.use_sliding_window and L > 1:
            seq_len = k_full.shape[1]
            current_query_len = L
            col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
            row_indices = torch.arange(seq_len - current_query_len, seq_len, device=x.device).unsqueeze(1)
            
            within_window = (col_indices <= row_indices) & (col_indices >= row_indices - self.window_size + 1)
            
            if self.use_flash_attention:
                swa_mask = ~within_window
                swa_mask = swa_mask.unsqueeze(0).unsqueeze(0)
                additive_mask = torch.zeros_like(swa_mask, dtype=x.dtype)
                additive_mask.masked_fill_(swa_mask, float("-inf"))
                
                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                        attention_mask = attention_mask.to(x.dtype)
                        attention_mask.masked_fill_(~attention_mask, float("-inf"))
                    attn_mask = attention_mask + additive_mask
                else:
                    attn_mask = additive_mask
            else:
                swa_mask = ~within_window
                if attention_mask is not None:
                    if attention_mask.dtype != torch.bool:
                        attention_mask = attention_mask != float("-inf")
                    attn_mask = ~attention_mask | swa_mask
                else:
                    attn_mask = swa_mask

        # ── 8. Attention 计算 ──
        if self.use_flash_attention:
            try:
                if attn_mask is not None:
                    if attn_mask.dim() == 2:
                        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                    elif attn_mask.dim() == 3:
                        attn_mask = attn_mask.unsqueeze(1)

                attn_output = F.scaled_dot_product_attention(
                    q, k_full, v_full,
                    attn_mask=attn_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
            except (RuntimeError, AssertionError):
                attn_output = self._naive_attention(q, k_full, v_full, attn_mask, is_causal)
        else:
            attn_output = self._naive_attention(q, k_full, v_full, attn_mask, is_causal)

        # ── 9. 输出投影 ──
        attn_output = attn_output.contiguous().view(B, L, -1)
        output = self.o_proj(attn_output)
        
        # ── 10. 返回压缩后的 KV Cache ──
        present_key_values = (k_latent, v_latent) if use_cache else None
        
        return output, present_key_values


class FlashAttention(nn.Module):
    """Flash Attention implementation supporting PyTorch SDPA and optional flash_attn library.

    Performance benefits:
    - 2-4x faster than naive attention on modern GPUs
    - O(N) memory complexity instead of O(N²)
    - Automatic dispatch to best available implementation

    Args:
        use_flash_attn_lib: Prefer flash_attn library if available
        dropout: Dropout probability
    """

    def __init__(
        self,
        use_flash_attn_lib: bool = True,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_dropout = attention_dropout

        # Check for flash_attn library
        self._has_flash_attn = False
        self._flash_attn_func = None
        if use_flash_attn_lib:
            try:
                import flash_attn
                from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
                self._has_flash_attn = True
                self._flash_attn_func = flash_attn_func
            except ImportError:
                pass

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Flash Attention.

        Args:
            q: [batch_size, seq_len, num_heads, head_dim]
            k: [batch_size, seq_len, num_heads, head_dim]
            v: [batch_size, seq_len, num_heads, head_dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            training: Whether in training mode (for dropout)

        Returns:
            [batch_size, seq_len, num_heads, head_dim]
        """
        # Try flash_attn library first if available
        if self._has_flash_attn and attention_mask is None:
            try:
                return self._flash_attn_func(
                    q, k, v,
                    dropout_p=self.attention_dropout if training else 0.0,
                    causal=is_causal,
                )
            except Exception:
                pass

        # Try PyTorch SDPA (default for PyTorch 2.0+
        try:
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if training else 0.0,
                is_causal=is_causal,
            )
        except Exception:
            # Fallback to naive attention
            return self._naive_attention(q, k, v, attention_mask, is_causal)

    def _naive_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Naive attention fallback for CPU or unsupported hardware.

        Computes standard scaled dot-product attention.
        O(N²) memory complexity — use only as fallback.
        """
        B, L, H, D = q.shape
        kv_len = k.shape[1]
        scale = 1.0 / math.sqrt(D)

        # Compute attention scores: [B, H, L, kv_len]
        q_t = q.transpose(1, 2)  # [B, H, L, D]
        k_t = k.transpose(1, 2)  # [B, H, kv_len, D]
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, H, L, kv_len]

        if is_causal:
            # Create causal mask that respects past key values
            # Mask positions where key position > query position
            # (query positions are relative to full sequence)
            # Full sequence length is kv_len, our queries are the last L positions
            causal_mask = torch.triu(
                torch.ones(L, kv_len, device=q.device, dtype=torch.bool),
                diagonal=kv_len - L + 1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, L] or [B, H, L, L]
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        v_t = v.transpose(1, 2)  # [B, H, L, D]
        output = torch.matmul(attn_weights, v_t)  # [B, H, L, D]
        return output.transpose(1, 2)  # [B, L, H, D]


class ModernAttention(nn.Module):
    """Modern causal attention with Flash Attention, RoPE, GQA, KV Cache, and SWA.

    Performance benefits:
    - Flash Attention: 2-4x faster on modern GPUs
    - O(N) memory complexity
    - Automatic fallback to naive attention on CPU
    - Supports flash_attn library (if installed) or PyTorch SDPA

    Args:
        d_model: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads. If None, uses multi-head attention (MHA).
                      If < num_heads, uses Grouped Query Attention (GQA).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE pre-computation.
        rope_scaling: Optional YaRN-style RoPE scaling dict.
        attention_dropout: Dropout probability for attention weights.
        use_flash_attention: Use Flash Attention.
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

        # Flash Attention module
        self.flash_attn = FlashAttention(
            use_flash_attn_lib=True,
            attention_dropout=attention_dropout,
        )

        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """Forward pass with Flash Attention, with optional KV Cache and SWA.

        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask (e.g., for padding).
                           Shape: [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len].
            is_causal: If True, apply causal masking automatically.
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
            # For the current query positions (L positions)
            current_query_len = L
            # Compute which key positions are allowed
            col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
            row_indices = torch.arange(seq_len - current_query_len, seq_len, device=x.device).unsqueeze(1)  # [L, 1]
            
            # Each position i can see:
            # - j <= i (causal)
            # - j >= i - window_size + 1 (sliding window)
            within_window = (col_indices <= row_indices) & (col_indices >= row_indices - self.window_size + 1)
            
            if self.use_flash_attention:
                # For SDPA: create a boolean mask
                # We need to convert to additive mask (-inf for masked positions)
                # SDPA expects mask of shape [B, H, L, seq_len] or broadcastable
                swa_mask = ~within_window  # True means masked
                # Expand to batch and heads dimensions
                swa_mask = swa_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, seq_len]
                # Convert to additive mask (-inf for masked positions)
                additive_mask = torch.zeros_like(swa_mask, dtype=x.dtype)
                additive_mask.masked_fill_(swa_mask, float("-inf"))
                
                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                        # Boolean mask, convert to additive
                        attention_mask = attention_mask.to(x.dtype)
                        attention_mask.masked_fill_(~attention_mask, float("-inf"))
                    # Combine masks
                    attn_mask = attention_mask + additive_mask
                else:
                    attn_mask = additive_mask
            else:
                # For naive attention: create boolean mask
                swa_mask = ~within_window  # True means masked
                if attention_mask is not None:
                    if attention_mask.dtype != torch.bool:
                        # Convert additive mask to boolean
                        attention_mask = attention_mask != float("-inf")
                    attn_mask = ~attention_mask | swa_mask
                else:
                    attn_mask = swa_mask

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        # Use FlashAttention module
        if self.use_flash_attention:
            attn_output = self.flash_attn(
                q, k, v,
                attention_mask=attn_mask,
                is_causal=is_causal,
                training=self.training,
            )
        else:
            attn_output = self.flash_attn._naive_attention(q, k, v, attn_mask, is_causal)

        # Reshape and project output
        attn_output = attn_output.contiguous().view(B, L, -1)  # [B, L, num_heads * head_dim]
        return self.o_proj(attn_output), present_key_values


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) for memory-efficient KV cache.
    
    MLA compresses KV cache by projecting keys and values into a lower-dimensional
    latent space, significantly reducing memory usage while maintaining performance.
    
    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        latent_dim: Dimension of the latent space for KV compression.
        num_latent_heads: Number of latent attention heads.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE pre-computation.
        rope_scaling: Optional YaRN-style RoPE scaling dict.
        use_flash_attention: Use Flash Attention via PyTorch SDPA.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        latent_dim: int = MLA_LATENT_DIM,
        num_latent_heads: int = MLA_NUM_LATENT_HEADS,
        dropout: float = 0.0,
        max_seq_len: int = DEFAULT_MAX_LEN,
        rope_scaling: Optional[dict] = None,
        use_flash_attention: bool = USE_FLASH_ATTENTION,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert latent_dim % num_latent_heads == 0, "latent_dim must be divisible by num_latent_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.num_latent_heads = num_latent_heads
        self.head_dim = d_model // num_heads
        self.latent_head_dim = latent_dim // num_latent_heads
        self.use_flash_attention = use_flash_attention
        self.attention_dropout = attention_dropout
        
        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        
        # Latent projections for K and V
        self.k_proj_latent = nn.Linear(d_model, num_latent_heads * self.latent_head_dim, bias=False)
        self.v_proj_latent = nn.Linear(d_model, num_latent_heads * self.latent_head_dim, bias=False)
        
        # Projection from latent to query space
        self.k_latent_to_full = nn.Linear(num_latent_heads * self.latent_head_dim, num_heads * self.head_dim, bias=False)
        self.v_latent_to_full = nn.Linear(num_latent_heads * self.latent_head_dim, num_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
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
        """Naive attention fallback for CPU or unsupported hardware."""
        B, L, H, D = q.shape
        kv_len = k.shape[1]
        scale = 1.0 / math.sqrt(D)
        
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        
        if is_causal:
            # Create causal mask that respects past key values
            causal_mask = torch.triu(
                torch.ones(L, kv_len, device=q.device, dtype=torch.bool),
                diagonal=kv_len - L + 1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.attention_dropout > 0:
            attn_weights = self.dropout(attn_weights)
        
        v_t = v.transpose(1, 2)
        output = torch.matmul(attn_weights, v_t)
        return output.transpose(1, 2)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """Forward pass with latent KV compression.
        
        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            is_causal: If True, apply causal masking
            use_cache: If True, return compressed KV cache
            past_key_values: Optional tuple of (compressed_k, compressed_v) from previous steps
            
        Returns:
            Tuple of (output, present_key_values)
        """
        B, L, D = x.shape
        
        # Query projection
        q = self.q_proj(x)
        q = q.view(B, L, self.num_heads, self.head_dim)
        
        # K and V projections to latent space
        k_latent = self.k_proj_latent(x)
        v_latent = self.v_proj_latent(x)
        
        # Handle KV cache
        if past_key_values is not None:
            past_k_latent, past_v_latent = past_key_values
            k_latent_full = torch.cat([past_k_latent, k_latent], dim=1)
            v_latent_full = torch.cat([past_v_latent, v_latent], dim=1)
        else:
            k_latent_full = k_latent
            v_latent_full = v_latent
        
        # Store compressed KV for cache
        present_key_values = (k_latent_full, v_latent_full) if use_cache else None
        
        # Project latent K/V to full dimension
        k_full = self.k_latent_to_full(k_latent_full)
        v_full = self.v_latent_to_full(v_latent_full)
        
        # Reshape to multi-head format
        k_full = k_full.view(B, k_latent_full.shape[1], self.num_heads, self.head_dim)
        v_full = v_full.view(B, v_latent_full.shape[1], self.num_heads, self.head_dim)
        
        # Apply RoPE to query and keys
        full_seq_len = k_latent_full.shape[1]
        cos_full, sin_full = self.rotary(full_seq_len, device=x.device)
        
        # Apply RoPE to query - only to the current positions
        cos_q = cos_full[-L:] if L < full_seq_len else cos_full
        sin_q = sin_full[-L:] if L < full_seq_len else sin_full
        q, _ = self.rotary.apply_rotary_qk(q, q, cos_q, sin_q)
        
        # Apply RoPE to all keys
        _, k_full = self.rotary.apply_rotary_qk(
            torch.zeros(B, full_seq_len, self.num_heads, self.head_dim, device=x.device),
            k_full,
            cos_full,
            sin_full
        )
        
        # Attention computation - use naive attention for reliability
        attn_output = self._naive_attention(q, k_full, v_full, attention_mask, is_causal)
        
        # Reshape and project output
        attn_output = attn_output.contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_values


class ModernTransformerBlock(nn.Module):
    """Single transformer layer with modern components: RoPE, GQA, SwiGLU/MoE, and optional MLA.

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
        use_mla: If True, use Multi-head Latent Attention (MLA) for KV cache compression.
        latent_compression_ratio: Compression ratio for MLA (2, 4, or 8).
        gradient_checkpointing: If True, use activation checkpointing to save memory during training.
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
        attention_dropout: float = 0.05,
        use_sliding_window: bool = USE_SLIDING_WINDOW,
        window_size: int = SWA_WINDOW_SIZE,
        use_mla: bool = False,
        mla_latent_dim: int = MLA_LATENT_DIM,
        mla_num_latent_heads: int = MLA_NUM_LATENT_HEADS,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        
        self.gradient_checkpointing = gradient_checkpointing
        
        if use_mla:
            self.attention = MultiHeadLatentAttention(
                d_model=d_model,
                num_heads=num_heads,
                latent_dim=mla_latent_dim,
                num_latent_heads=mla_num_latent_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_scaling=rope_scaling,
                use_flash_attention=USE_FLASH_ATTENTION,
                attention_dropout=attention_dropout,
            )
        else:
            self.attention = ModernAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_scaling=rope_scaling,
                attention_dropout=attention_dropout,
                use_flash_attention=USE_FLASH_ATTENTION,
                use_sliding_window=use_sliding_window,
                window_size=window_size,
            )
        self.attention_norm = nn.RMSNorm(d_model)

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

    def _forward_with_checkpoint(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        use_cache: bool,
        past_key_values: Optional[tuple],
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """Forward pass with gradient checkpointing enabled."""
        def forward_inner():
            attn_out, present = self.attention(
                self.attention_norm(x), attention_mask, use_cache=use_cache, past_key_values=past_key_values
            )
            x_temp = x + self.dropout_layer(attn_out)
            x_temp = x_temp + self.feed_forward(self.ffn_norm(x_temp))
            return x_temp, present
        
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                forward_inner,
                use_reentrant=False,
                determinism_check="none",
            )
        else:
            return forward_inner()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        if self.gradient_checkpointing:
            return self._forward_with_checkpoint(x, attention_mask, use_cache, past_key_values)
        
        attn_out, present = self.attention(
            self.attention_norm(x), attention_mask, use_cache=use_cache, past_key_values=past_key_values
        )
        x = x + self.dropout_layer(attn_out)
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
    - Optional MLA (Multi-head Latent Attention): Compresses KV cache to reduce memory usage by 90%+

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
        use_mla: Use Multi-head Latent Attention for KV cache compression (modern mode only).
        latent_compression_ratio: Compression ratio for MLA (2, 4, or 8; default=4).
        gradient_checkpointing: If True, use activation checkpointing to save 30-50% memory.
        gradient_checkpointing_ratio: Ratio of layers to apply checkpointing (0.0-1.0).
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
        mode: str = "modern",
        num_kv_heads: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        rope_scaling: Optional[dict] = None,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_ratio: float = 1.0,
        use_weight_tying: bool = USE_WEIGHT_TYING,
        use_sliding_window: bool = USE_SLIDING_WINDOW,
        window_size: int = SWA_WINDOW_SIZE,
        use_mla: bool = False,
        mla_latent_dim: int = MLA_LATENT_DIM,
        mla_num_latent_heads: int = MLA_NUM_LATENT_HEADS,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio
        self.use_weight_tying = use_weight_tying
        self._tied_weights = False

        if mode == "legacy":
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        elif mode == "modern":
            num_checkpoint_layers = max(1, int(num_layers * gradient_checkpointing_ratio))
            checkpoint_indices = set(
                i for i in range(num_layers) if i < num_checkpoint_layers
            )
            
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
                    use_mla=use_mla,
                    mla_latent_dim=mla_latent_dim,
                    mla_num_latent_heads=mla_num_latent_heads,
                    gradient_checkpointing=(gradient_checkpointing and i in checkpoint_indices),
                )
                for i in range(num_layers)
            ])
            self.num_layers = num_layers
            self.checkpoint_indices = checkpoint_indices
            self.norm = nn.RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'legacy' or 'modern'.")

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

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
            if self.gradient_checkpointing:
                output = torch.utils.checkpoint.checkpoint(
                    self.transformer_encoder, src, use_reentrant=False
                )
            else:
                output = self.transformer_encoder(src)
        else:
            src = self.embedding(src)
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

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for all layers."""
        self.gradient_checkpointing = True
        for block in self.transformer_blocks:
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing for all layers."""
        self.gradient_checkpointing = False
        for block in self.transformer_blocks:
            block.gradient_checkpointing = False

    def set_gradient_checkpointing_ratio(self, ratio: float) -> None:
        """Set gradient checkpointing ratio for selective checkpointing.
        
        Args:
            ratio: Ratio of layers to apply checkpointing (0.0-1.0).
                   For example, 0.5 means checkpoint first half of layers.
        """
        self.gradient_checkpointing_ratio = ratio
        num_checkpoint_layers = max(1, int(self.num_layers * ratio))
        checkpoint_indices = set(
            i for i in range(self.num_layers) if i < num_checkpoint_layers
        )
        self.checkpoint_indices = checkpoint_indices
        for i, block in enumerate(self.transformer_blocks):
            block.gradient_checkpointing = (i in checkpoint_indices)

    def compile(
        self,
        mode: str = "reduce-overhead",
        backend: str = "inductor",
        dynamic: bool = True,
        fullgraph: bool = False,
    ) -> "SimpleTransformer":
        """Compile the model with torch.compile() for 2-3x speedup.

        This uses torch.compile() to optimize the model's forward pass,
        providing significant speedups for both training and inference.

        Args:
            mode: Compilation mode - "reduce-overhead", "max-autotune", or "default".
                - "reduce-overhead": Best for training (reduces Python overhead)
                - "max-autotune": Best for inference (maximizes throughput)
                - "default": Balanced option
            backend: Backend to use for compilation.
                - "inductor": Default PyTorch 2.0 backend (good all-around)
                - "aot_eager": AOT compilation with eager backend (debugging)
                - "cudagraphs": CUDA graphs for even faster inference
            dynamic: Enable dynamic shapes (required for varying sequence lengths).
                Recommended to keep True for flexibility.
            fullgraph: Require full graph compilation (no graph breaks).
                Set to False for better compatibility with complex models.

        Returns:
            Self for method chaining.

        Note:
            - Compilation is done lazily on first forward pass
            - Compilation time is not included in training time
            - Use model._compiled attribute to check if compiled
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch.compile() requires PyTorch 2.0+")

        import torch._dynamo

        self._compiled = True
        self._compile_mode = mode
        self._compile_backend = backend

        compile_kwargs = {
            "mode": mode,
            "backend": backend,
            "dynamic": dynamic,
            "fullgraph": fullgraph,
        }

        try:
            from src.logger import get_logger
            _compile_logger = get_logger("LingmaoMoyun.Model")
            _compile_logger.info(f"Compiling model with torch.compile(mode='{mode}', backend='{backend}', dynamic={dynamic})")
        except Exception:
            import logging
            _compile_logger = logging.getLogger(__name__)
            _compile_logger.info(f"Compiling model with torch.compile(mode='{mode}', backend='{backend}', dynamic={dynamic})")

        self._torch_compile_kwargs = compile_kwargs

        return self

    def get_compiled_model(self) -> Union["SimpleTransformer", torch.nn.Module]:
        """Return the compiled version if available, otherwise self.

        This allows transparent access to the compiled model for inference
        while maintaining the original model interface.

        Returns:
            Compiled model if torch.compile() was called, else self.
        """
        if hasattr(self, '_compiled') and self._compiled:
            if not hasattr(self, '_compiled_instance'):
                compile_kwargs = getattr(self, '_torch_compile_kwargs', {
                    "mode": "reduce-overhead",
                    "backend": "inductor",
                    "dynamic": True,
                })
                self._compiled_instance = torch.compile(self, **compile_kwargs)
            return self._compiled_instance
        return self

    @property
    def is_compiled(self) -> bool:
        """Check if model is marked for compilation."""
        return getattr(self, '_compiled', False)

    def decompile(self) -> "SimpleTransformer":
        """Remove compilation and return to eager mode.

        This is useful when you need to debug the model or use
        features that are incompatible with torch.compile().

        Returns:
            Self for method chaining.
        """
        if hasattr(self, '_compiled_instance'):
            del self._compiled_instance
        self._compiled = False
        return self
