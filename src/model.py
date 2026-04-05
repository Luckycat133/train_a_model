"""Model architecture definitions for the Lingmao Moyun training system."""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - fallback for test environments
    torch = None

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

        # PE[pos, 2i]     = sin(pos / 10000^(2i/d_model))
        # PE[pos, 2i+1]   = cos(pos / 10000^(2i/d_model))
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


class SimpleTransformer(nn.Module):
    """Causal transformer language model.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: Hidden dimension in the feed-forward sub-layer.
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encodings.
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
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        # FIX: pass max_len to PositionalEncoding so it is not silently ignored
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.max_len = max_len

        self._init_parameters()

    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            src: LongTensor of shape [batch_size, seq_len].

        Returns:
            FloatTensor of shape [batch_size, seq_len, vocab_size].
        """
        # Embed + scale
        src = self.embedding(src) * math.sqrt(self.d_model)
        # Add positional encoding
        src = self.pos_encoder(src)
        # Transformer encoder
        output = self.transformer_encoder(src)
        # Project to vocabulary
        output = self.output_layer(output)
        return output
