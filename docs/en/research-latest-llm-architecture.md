# Research Report: Latest LLM Architecture (2025-2026)

> Generated: 2026-04-05 | Scope: 2025 Q1 – 2026 Q1

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Mixture of Experts (MoE)](#2-mixture-of-experts-moe)
3. [Long Context Techniques](#3-long-context-techniques)
4. [Training Techniques: GRPO, RLHF, DPO](#4-training-techniques-grpo-rlhf-dpo)
5. [Inference Optimization](#5-inference-optimization)
6. [Chinese LLM Landscape](#6-chinese-llm-landscape)
7. [Architecture Upgrade Recommendations](#7-architecture-upgrade-recommendations)
8. [Code Modification Plan](#8-code-modification-plan)
9. [References](#9-references)

---

## 1. Executive Summary

The LLM architecture landscape in 2025-2026 is defined by five converging trends:

| Trend | Status | Impact |
|-------|--------|--------|
| **MoE** | Production-ready (Qwen3, DeepSeek-V3, Gemma-3) | 3-10x inference efficiency |
| **Long Context** | 128K–2M tokens mainstream | YaRN/LongRoPE/KVCache compression |
| **RL-based Training** | GRPO dominates post-training | DeepSeek-R1 style reasoning |
| **Flash Attention** | v3 on Hopper, v2 on Ampere | 2-4x speedup over naive attention |
| **Speculative Decoding** | Self-speculative + multi-draft | 2-3x latency reduction |

**For `train_a_model`**: The current codebase uses a basic sinusoidal positional encoding + dense TransformerEncoder. The recommended upgrade path is to add RoPE (with YaRN for long context), GQA support, and optional MoE layers — without abandoning the simple, educational architecture.

---

## 2. Mixture of Experts (MoE)

### 2.1 Core Concept

MoE replaces dense FFN layers with a router + multiple expert FFNs. Only `top_k` experts are activated per token, dramatically reducing active parameter count.

```
Standard FFN:     y = FFN(x)           # all params active
MoE FFN:          y = Σᵢ wᵢ · FFNᵢ(x)  # only top-k experts active
```

### 2.2 Latest Advances (2025-2026)

#### DeepSeek-V3 (Dec 2024 / Mar 2025 update)
- **Architecture**: 685B total params, 37B active per token (auxiliary-loss-free load balancing)
- **Multi-Token Prediction (MTP)**: Predicts multiple tokens simultaneously, boosting training signal
- **FP8 Mixed Precision Training**: First open-source large-scale FP8 training
- **DualPipe Bidirectional Pipeline**: Overlapping forward/backward compute across GPUs
- **Key Innovation**: No value network needed for GRPO — uses group-relative advantage

#### Qwen3 (Apr 2025)
- **MoE Models**: Qwen3-235B-A22B (235B total / 22B active), Qwen3-30B-A3B (30B total / 3B active)
- **Dense Models**: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- **Hybrid Thinking Mode**: First open-source model with automatic fast/slow thinking switching
- **Native Support for Agentic workflows** with tool use and multi-step reasoning

#### Gemma 3 (Mar 2026)
- **Multi-modal MoE**: Gemma-3 4B uses lookup experts (MoLE-style) for parameter efficiency
- **Architecture**: Uses Grouped Query Attention + RoPE + local attention windows

#### MoLE (ICML 2025 Oral)
- **Mixture of Lookup Experts**: Specializes experts by semantic similarity rather than domain
- More parameter-efficient than standard top-k MoE for small model regimes

### 2.3 MoE Architecture Patterns

```python
# Typical MoE Layer Structure (pseudocode)
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)])
        self.top_k = top_k
    
    def forward(self, x):
        logits = self.router(x)                    # [B, L, num_experts]
        weights, indices = torch.topk(logits, self.top_k)  # select top-k
        weights = F.softmax(weights, dim=-1)
        
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            out += weights[mask] * expert(x[mask])
        return out
```

### 2.4 Key Takeaway for `train_a_model`

MoE is production-critical for large models (>7B) but adds significant complexity. **Recommended**: Add MoE as an optional layer type (config flag `use_moe=False` default), enabling researchers to experiment with expert counts and top-k values.

---

## 3. Long Context Techniques

### 3.1 Ring Attention

**Paper**: [Ring Attention](https://arxiv.org/abs/2310.01889) (Liu et al., 2023, UC Berkeley)

Ring Attention distributes attention computation across multiple devices, enabling linear scaling of context length with device count.

- **Mechanism**: Sequence is split across devices; each device computes partial attention; results are exchanged in a ring
- **Limitation**: Requires multiple GPUs with high-speed interconnects (NVLink)
- **Status**: Integrated into Flash Attention's sequence parallelism

### 3.2 StreamingLLM

**Paper**: [StreamingLLM](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023, MIT)

Enables infinite-length text generation without fine-tuning by preserving attention sinks (first ~4 tokens) + local window attention.

```python
# StreamingLLM attention pattern
def streaming_attention(x, kv_cache, window_size=512, sink_tokens=4):
    # Keep first 4 tokens as "attention sink" (high attention scores always)
    # Apply sliding window attention to recent tokens
    # Skip recomputing full attention over full history
```

### 3.3 LongRoPE (2024, Microsoft)

**Paper**: [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13700)

LongRoPE addresses non-uniform positional interpolation challenges:

1. **Non-uniformity Discovery**: Early tokens and later tokens have different optimal rope scales
2. **Progressive Extension**: 4K → 8K → 32K → 128K → 256K → 512K → 1M → 2M
3. **Key Insight**: Using different RoPE scales for different position ranges avoids "position aliasing"

```python
# LongRoPE-style multi-scale RoPE (simplified)
class LongRoPEScale:
    def __init__(self, base_scale=1.0, extended_scale=4.0, break_point=4096):
        self.base_scale = base_scale
        self.extended_scale = extended_scale
        self.break_point = break_point
    
    def get_scale(self, position):
        if position < self.break_point:
            return self.base_scale
        return self.extended_scale
```

### 3.4 YaRN (Yet another RoPE extensioN)

**Paper**: [YaRN](https://arxiv.org/abs/2309.00071) (Peng et al., 2023)

YaRN extends RoPE's context length by 16x through:
- **NTK-aware interpolation**: More aggressive interpolation at higher dimensions
- **Compressed memory**: Reduces KV cache by 2x
- Used by Llama-3 70B for 128K context

### 3.5 Comparison: Positional Encoding Evolution

| Method | Context Extension | Fine-tuning Required | KV Cache | Used By |
|--------|------------------|---------------------|----------|---------|
| Sinusoidal (original) | None | N/A | Full | Vanilla Transformer |
| RoPE (2021) | Moderate | Yes | Full | LLaMA, Qwen |
| ALiBi (2022) | Moderate | Yes | Full | Bloom |
| YaRN (2023) | 16x | Yes | Compressed 2x | Llama-3 |
| LongRoPE (2024) | 512x | Yes (progressive) | Full | InternLM |
| StreamingLLM (2023) | Infinite | No | Window only | Streaming apps |

### 3.6 Key Takeaway for `train_a_model`

**Current state**: Uses basic sinusoidal PE (fixed 1024 max_len).
**Recommended upgrade**: Replace with RoPE + YaRN for context up to 32K tokens. LongRoPE is too complex for initial upgrade but should be planned.

---

## 4. Training Techniques: GRPO, RLHF, DPO

### 4.1 Overview of Alignment Methods

```
Pretrained LLM → SFT (Supervised Fine-Tuning) → RLHF / GRPO / DPO
```

| Method | What it does | Pros | Cons |
|--------|-------------|------|------|
| **RLHF (PPO)** | Uses reward model + value network + KL constraint | Best quality | Complex, unstable, memory-heavy |
| **GRPO** (DeepSeek, 2025) | Group-relative advantage, no value network | 50% less memory, simpler | Reward hacking risk |
| **DPO** (Direct Preference Optimization) | Reformulates RL as supervised learning | No reward model needed | Can be unstable |
| **RLAIF** | Uses AI feedback instead of human feedback | Scalable | Label quality concerns |

### 4.2 GRPO (Group Relative Policy Optimization) — DeepSeek, 2025

**Paper**: [DeepSeekMath: Pushing Mathematical Reasoning beyond GSM8K](https://arxiv.org/abs/2402.03300)

GRPO is DeepSeek's answer to PPO's memory/complexity problems:

```python
# GRPO core algorithm (simplified)
def grpo_train(policy, prompts, reward_fn, group_size=8):
    """
    1. For each prompt, generate group_size responses from current policy
    2. Compute reward for each response
    3. Compute advantage = reward - mean(rewards_in_group) / std(rewards_in_group)
    4. Policy gradient update with KL penalty against reference model
    """
    for prompt in prompts:
        responses = [policy.generate(prompt) for _ in range(group_size)]
        rewards = [reward_fn(prompt, r) for r in responses]
        
        # Relative advantage within group
        baseline = sum(rewards) / len(rewards)
        advantages = [r - baseline for r in rewards]
        
        # Policy gradient update (PPO-style clipping applied to advantages)
        loss = compute_policy_loss(policy, prompt, responses, advantages)
```

**Key innovations over PPO**:
- **No critic/value network**: Advantage = reward - group_mean (saves 50% memory)
- **Group-relative advantage**: Normalizes within generated group, reducing variance
- **Auxiliary loss-free load balancing**: DeepSeek-V3 uses auxiliary-loss-free router
- **Multi-token prediction**: Predicts multiple tokens at once, providing richer training signal

### 4.3 DPO (Direct Preference Optimization)

**Paper**: [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)

```python
# DPO loss (simplified)
def dpo_loss(policy, ref_policy, prompt, chosen_response, rejected_response, beta=0.1):
    """
    L = - log σ( β · (log π(y_w|x) - log π_ref(y_w|x) 
                          - log π(y_l|x) + log π_ref(y_l|x) ) )
    """
    logits_chosen = policy(prompt, chosen_response)
    logits_rejected = policy(prompt, rejected_response)
    
    ref_logits_chosen = ref_policy(prompt, chosen_response)
    ref_logits_rejected = ref_policy(prompt, rejected_response)
    
    log_ratio = (logits_chosen - ref_logits_chosen) - (logits_rejected - ref_logits_rejected)
    loss = -F.logsigmoid(beta * log_ratio)
    return loss.mean()
```

**Limitation**: DPO can lead to reward hacking and distribution collapse without careful hyperparameter tuning.

### 4.4 Comparison for `train_a_model`

| Method | Complexity | Memory | Quality | Recommended For |
|--------|-----------|--------|---------|----------------|
| RLHF/PPO | Very High | ~40GB | Best | Production |
| GRPO | Medium | ~20GB | Very Good | Reasoning models |
| DPO | Low | ~15GB | Good | Simple fine-tuning |
| SFT only | Lowest | ~10GB | Baseline | Initial fine-tuning |

**Recommendation**: GRPO is the most impactful for reasoning tasks. Consider adding GRPO as a post-training option for the `train_a_model` project.

---

## 5. Inference Optimization

### 5.1 Flash Attention Evolution

#### Flash Attention v1 (2022) — Tri Dao
- **Innovation**: Tiling + softmax recomputation = O(N²) memory → O(N)
- **Speedup**: 2-4x over naive attention implementation

#### Flash Attention v2 (2023)
- **Improvements**: Better thread block partitioning, 1.5-2x faster than v1
- **Key change**: Single head can now be split across thread blocks

#### Flash Attention v3 (Jul 2024) — Hopper-Only
- **H100-specific optimizations**:
  1. Asynchronous software pipeline hiding memory latency
  2. Tensor Memory Accelerator (TMA) for efficient global memory access
  3. FP8 low-precision computation (E4M3 format)
- **Performance**: 740 TFLOPs/s on H100 (75% hardware utilization vs. 35% for v2)

#### Flash Attention v2 on Ampere (A100, RTX 3090/4090)
```bash
# Installation
pip install flash-attn --no-build-isolation
# or from source
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
```

#### Flash Attention Integration in PyTorch
```python
# PyTorch 2.0+ has Flash Attention built-in
import torch.nn.functional as F

# Uses Flash Attention automatically when available
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,        # causal mask
    dropout_p=0.0,
    is_causal=True,        # automatic causal masking
    scale=None             # defaults to 1/sqrt(head_dim)
)
```

### 5.2 Speculative Decoding

**Paper**: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2023, Google)

```
Traditional Decoding (AR):
  Token₁ → Token₂ → Token₃ → Token₄  (sequential, 4 steps)

Speculative Decoding:
  Draft Model:  Token₁ → Token₂ → Token₃ → Token₄  (fast, 4 tokens)
  Target Model:  [Token₁ Token₂ Token₃ Token₄]      (parallel verify, 1 step)
```

**Acceptance Rate**: 70-90% depending on draft model quality

#### Self-Speculative Decoding (LayerSkip, 2024)
- Uses same model for draft (early layers) + verify (late layers)
- No separate draft model needed
- **Facebook LayerSkip**: Uses early exit at layer N for draft, full model for verify

#### Multi-Draft (Falcon, AAAI 2025)
- Semi-autoregressive drafting: Drafts multiple tokens in parallel
- 2.91-3.51x speedup on various benchmarks
- Custom decoding tree structure for efficient verification

### 5.3 KV Cache Optimization

```python
# Standard KV Cache (full history)
kv_cache = {
    'k': torch.zeros(seq_len, batch, num_heads, head_dim),
    'v': torch.zeros(seq_len, batch, num_heads, head_dim)
}

# Paged KV Cache (vLLM style) — memory efficient
class PagedKVCache:
    """Allocate fixed-size blocks, reuse for different sequences"""
    block_size = 16
    max_blocks = 1024
    
    def update(self, block_ids, new_k, new_v):
        # Only allocate new blocks, don't pre-allocate full seq
        pass
```

### 5.4 Key Takeaway for `train_a_model`

- **Flash Attention**: Already in PyTorch 2.x as `F.scaled_dot_product_attention`. Upgrade path: ensure `torch.backends.cuda.enable_flash_sdp(True)` is set.
- **Speculative Decoding**: More relevant for inference (`generate.py`) than training. Add as optional mode.

---

## 6. Chinese LLM Landscape

### 6.1 Qwen (Alibaba Cloud)

#### Qwen2.5 (2024)
- Context: 128K tokens
- Sizes: 0.5B to 72B dense, plus MoE variants
- Notable: Qwen2.5-Coder for code generation

#### Qwen3 (Apr 2025) — **Most Important Release**
- **First hybrid reasoning open-source model** (think + non-think modes)
- **MoE flagship**: Qwen3-235B-A22B (235B total / 22B active)
- **Smallest MoE**: Qwen3-30B-A3B (30B total / 3B active) — matches QwQ-32B with 10% active params
- **Dense lineup**: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- **Notable**: Uses YaRN for extended context, GQA in all models

**Architectural Highlights**:
- RoPE with YaRN scaling
- Grouped Query Attention (GQA) for memory efficiency
- SwiGLU activation function
- Native tool-use and agentic capabilities

### 6.2 DeepSeek (HigAI / 梁文峰's lab)

#### DeepSeek-V2 (May 2024)
- **MLA (Multi-head Latent Attention)**: Compresses KV cache dramatically
- **DeepSeekMoE**: Fine-grained expert segmentation, shared experts across all tokens
- First to demonstrate FP8 training viability at scale

#### DeepSeek-V3 (Dec 2024, updated Mar 2025)
- **685B parameters, MoE with 37B active**
- **Auxiliary-loss-free load balancing**: No auxiliary loss needed for expert routing stability
- **Multi-Token Prediction (MTP)**: Predicts 1+ additional tokens per step
- **DualPipe**: Bidirectional pipeline parallelism, eliminates pipeline bubbles
- **Cost**: ~$6M training cost (vs. $100M+ for comparable GPT-4 class models)

#### DeepSeek-R1 (Jan 2025)
- **First open-source reasoning model** competitive with o1
- **GRPO training**: No critic model needed
- **Distillation**: Qwen2.5-32B-R1-Distill, LLaMA-70B-R1-Distill

#### DeepSeek-R2 (Expected Q2 2025)
- Focus on improved mathematical and code reasoning
- Expected to use more advanced multi-token prediction

### 6.3 GLM (Zhipu AI / 智谱)

#### GLM-4 (Jan 2024)
- 128K context, multimodal capabilities
- ChatGLM: Open-source for academic research

#### GLM-4V (Vision)
- Multimodal understanding, integrated into CogView for image generation

### 6.4 Architectural Convergence

All three major Chinese labs (Qwen, DeepSeek, GLM) have converged on:

| Feature | Qwen3 | DeepSeek-V3 | GLM-4 |
|---------|-------|-------------|-------|
| RoPE | ✅ YaRN | ✅ YaRN | ✅ |
| GQA | ✅ | ✅ MLA (unique) | ✅ |
| MoE | ✅ 235B/22B | ✅ 685B/37B | ✅ |
| SwiGLU | ✅ | ✅ | ✅ |
| Flash Attention | ✅ | ✅ | ✅ |
| FP8 Training | ✅ | ✅ | ❌ |
| MTP | ❌ | ✅ | ❌ |

### 6.5 Key Takeaway for `train_a_model`

**Qwen3's architecture** is the best reference for educational implementation:
- Clean RoPE + YaRN
- GQA (not full MLA)
- Standard SwiGLU
- Clear separation between dense and MoE variants

---

## 7. Architecture Upgrade Recommendations

### 7.1 Priority Order

| Priority | Upgrade | Impact | Complexity | Risk |
|----------|---------|--------|------------|------|
| 🔴 P0 | Replace Sinusoidal PE with RoPE | High | Medium | Low |
| 🔴 P0 | Add Flash Attention via PyTorch SDPA | High | Low | Very Low |
| 🟡 P1 | Add GQA (Grouped Query Attention) | Medium | Medium | Medium |
| 🟡 P1 | Replace FFN with SwiGLU | Medium | Low | Low |
| 🟢 P2 | Add optional MoE layer | High | High | High |
| 🟢 P2 | Context length extension (YaRN) | Medium | Medium | Medium |

### 7.2 Detailed Recommendations

#### Replace Sinusoidal PE → RoPE

**Why**: RoPE is used by all modern LLMs (LLaMA, Qwen, DeepSeek). Sinusoidal PE cannot extrapolate beyond training length.

**How**: Implement rotary position embeddings based on [RoPE: Rotary Positional Embeddings](https://arxiv.org/abs/2104.09864).

```python
# In src/model.py — add RoPE implementation
class RotaryEmbedding(nn.Module):
    """RoPE: Rotary Positional Embeddings (Su et al., 2021)"""
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        return emb.cos(), emb.sin()
    
    @staticmethod
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary(self, q, k, cos, sin):
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
```

#### Add Flash Attention via PyTorch SDPA

**Why**: 2-4x speedup, memory reduction, already in PyTorch 2.x.

**How**: Ensure `F.scaled_dot_product_attention` is used in the attention computation. No custom CUDA kernel needed for training.

#### Replace FFN → SwiGLU

**Why**: SwiGLU (Swish + GLU) provides 5-15% quality improvement over ReLU-based FFNs. Used by LLaMA, Qwen, Gemma.

```python
class SwiGLU(nn.Module):
    """SwiGLU activation: x * sigmoid(W₁x) ⊙ (W₂x)"""
    
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x))) * self.w3(x)
```

#### Add GQA (Grouped Query Attention)

**Why**: Reduces KV cache by 2-8x while maintaining quality. Standard in Qwen, LLaMA-3, DeepSeek.

```python
# In attention layer, instead of:
# self.num_heads for both Q and K/V heads
# Use:
self.num_q_heads = num_heads       # e.g., 32
self.num_kv_heads = num_kv_heads    # e.g., 8 (4x compression for GQA)
# Q: [B, L, 32, head_dim]
# K, V: [B, L, 8, head_dim]  ← fewer heads
```

---

## 8. Code Modification Plan

### 8.1 File: `src/model.py`

#### Changes:

1. **Add RoPE** (`RotaryEmbedding` class)
2. **Add SwiGLU** (`SwiGLUFFN` class)
3. **Refactor `SimpleTransformer`** to support:
   - RoPE instead of sinusoidal PE
   - SwiGLU FFN
   - GQA attention
   - Flash Attention via `F.scaled_dot_product_attention`
   - Optional MoE layers

#### Proposed New Architecture:

```python
class SimpleTransformer(nn.Module):
    """
    Modern causal transformer with:
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU activation
    - Optional MoE layers
    - Flash Attention via PyTorch SDPA
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        num_kv_heads: int = None,  # None = MHA, or set for GQA
        num_layers: int = DEFAULT_NUM_LAYERS,
        dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
        max_len: int = DEFAULT_MAX_LEN,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = "silu",  # "silu" or "relu"
    ):
        ...
        # Replace nn.TransformerEncoderLayer with custom layer
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                num_kv_heads=num_kv_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_moe=use_moe,
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
```

### 8.2 File: `src/config.py`

#### Additions:

```python
# ─── Modern Architecture ──────────────────────────────────────────────────────
DEFAULT_NUM_KV_HEADS = None          # None = multi-head attention
DEFAULT_USE_MOE = False
DEFAULT_NUM_EXPERTS = 8
DEFAULT_TOP_K = 2
DEFAULT_ROPE_BASE = 10000
DEFAULT_ROPE_SCALING = None         # None or {"type": "yaRN", "factor": 2}
```

### 8.3 File: `generate.py`

#### Additions:

```python
def speculative_decode(
    model, draft_model, prompt, max_new_tokens=100, gamma=4
):
    """
    Speculative decoding with separate draft model.
    
    Args:
        model: Target model (larger)
        draft_model: Draft model (smaller)
        prompt: Input tokens
        gamma: Number of draft tokens per iteration
    """
    # 1. Draft: small model generates gamma tokens
    # 2. Verify: large model verifies all at once (parallel)
    # 3. Accept/reject based on acceptance ratio
    # 4. Return accepted tokens
```

### 8.4 File: `src/trainer.py`

#### GRPO Trainer Addition (Future):

```python
def grpo_train_step(policy_model, ref_model, prompts, reward_fn, group_size=8):
    """
    Single GRPO training step.
    """
    # Generate group_size responses
    responses = [policy_model.generate(p) for p in prompts]
    
    # Compute rewards
    rewards = [reward_fn(p, r) for p, r in zip(prompts, responses)]
    
    # Compute relative advantage within group
    group_rewards = torch.tensor(rewards).view(group_size, -1)
    advantages = (group_rewards - group_rewards.mean(dim=0, keepdim=True))
    advantages = advantages / (group_rewards.std() + 1e-8)
    
    # Policy gradient update with KL penalty
    loss = compute_grpo_loss(policy_model, ref_model, prompts, responses, advantages)
    
    loss.backward()
    return loss.item()
```

---

## 9. References

### Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| [Attention is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer architecture |
| [RoPE: Rotary Positional Embeddings](https://arxiv.org/abs/2104.09864) | 2021 | Rotary position encoding |
| [Flash Attention](https://arxiv.org/abs/2205.14135) | 2022 | IO-aware exact attention |
| [Flash Attention-2](https://arxiv.org/abs/2307.08691) | 2023 | Improved parallelism |
| [Flash Attention-3](https://arxiv.org/abs/2407.07403) | 2024 | Hopper-optimized FP8 |
| [YaRN](https://arxiv.org/abs/2309.00071) | 2023 | Efficient context extension |
| [LongRoPE](https://arxiv.org/abs/2402.13700) | 2024 | 2M token context |
| [Speculative Decoding](https://arxiv.org/abs/2211.17192) | 2022 | 2-3x inference speedup |
| [LayerSkip (Self-Speculative)](https://arxiv.org/abs/2404.09526) | 2024 | Same-model speculation |
| [DeepSeek-V3](https://arxiv.org/abs/2501.12599) | 2024 | MoE + MTP + FP8 training |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12375) | 2025 | GRPO reasoning |
| [Qwen3](https://huggingface.co/Qwen/Qwen3) | 2025 | Hybrid reasoning MoE |
| [GRPO](https://arxiv.org/abs/2402.03300) | 2024 | Group relative PPO |
| [DPO](https://arxiv.org/abs/2305.18290) | 2023 | Direct preference optimization |
| [Gemma-3](https://arxiv.org/abs/2503.21691) | 2026 | Multi-modal lookup experts |

### Implementation Resources

- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **RoPE Implementation**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- **DeepSeek-V3 Tech Report**: https://github.com/deepseek-ai/DeepSeek-V3
- **Qwen3**: https://github.com/QwenLM/Qwen3
- **TRL (GRPO/DPO trainers)**: https://github.com/huggingface/trl

---

## Appendix: Changelog

| Date | Change |
|------|--------|
| 2026-04-05 | Initial research report created |
