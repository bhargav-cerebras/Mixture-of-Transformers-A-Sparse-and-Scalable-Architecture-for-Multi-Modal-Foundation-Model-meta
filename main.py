import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_hidden_dim: Optional[int] = None
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layernorm_eps: float = 1e-5
    modalities: List[str] = field(default_factory=lambda: ['text', 'image', 'speech'])
    max_seq_length: int = 5000

    def __post_init__(self) -> None:
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = self.d_model * 4
        if not self.modalities:
            self.modalities = ['text', 'image', 'speech']


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        if seq_length > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum sequence length {self.pe.size(1)}."
            )
        return x + self.pe[:, :seq_length, :].to(x.device)


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1,
        activation: Literal['relu', 'gelu', 'swish'] = 'gelu'
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = self._get_activation(activation)
        self.linear2 = nn.Linear(ffn_hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _get_activation(name: str):
        activations: Dict[str, callable] = {
            'relu': F.relu,
            'gelu': F.gelu,
            'swish': lambda x: x * torch.sigmoid(x)
        }
        if name.lower() not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[name.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ModalitySpecificTransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.modalities = config.modalities

        self.attn_layers: nn.ModuleDict[str, nn.MultiheadAttention] = nn.ModuleDict({
            modality: nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True
            ) for modality in self.modalities
        })

        self.ffn_layers: nn.ModuleDict[str, FeedForward] = nn.ModuleDict({
            modality: FeedForward(
                d_model=config.d_model,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout
            ) for modality in self.modalities
        })

        self.norm1: nn.ModuleDict[str, nn.LayerNorm] = nn.ModuleDict({
            modality: nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
            for modality in self.modalities
        })
        self.norm2: nn.ModuleDict[str, nn.LayerNorm] = nn.ModuleDict({
            modality: nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
            for modality in self.modalities
        })

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality '{modality}'. Expected one of: {self.modalities}")

        residual = x
        x_norm = self.norm1[modality](x)
        attn_output, _ = self.attn_layers[modality](
            x_norm, x_norm, x_norm, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, need_weights=False
        )
        x = residual + self.dropout(attn_output)

        residual = x
        x_norm = self.norm2[modality](x)
        ffn_output = self.ffn_layers[modality](x_norm)
        x = residual + self.dropout(ffn_output)

        return x


class MixtureOfTransformers(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.modalities = config.modalities

        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_seq_length=config.max_seq_length
        )

        self.modality_embedding = nn.Embedding(len(self.modalities), config.d_model)
        self.modality_to_index: Dict[str, int] = {modality: idx for idx, modality in enumerate(self.modalities)}

        self.layers = nn.ModuleList([
            ModalitySpecificTransformerLayer(config)
            for _ in range(config.num_layers)
        ])

        self.global_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        self.global_norm = nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
        self.dropout = nn.Dropout(config.dropout)

        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if modality not in self.modality_to_index:
            raise ValueError(f"Unknown modality '{modality}'. Expected one of: {list(self.modality_to_index.keys())}")

        batch_size, seq_length, d_model = x.size()
        if d_model != self.config.d_model:
            raise ValueError(f"Input d_model ({d_model}) does not match config d_model ({self.config.d_model})")

        device = x.device

        modality_idx = torch.full((batch_size,), self.modality_to_index[modality], dtype=torch.long, device=device)
        modality_embed = self.modality_embedding(modality_idx).unsqueeze(1)
        x = x + modality_embed

        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(
                x,
                modality=modality,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )

        residual = x
        x_norm = self.global_norm(x)
        global_attn_output, _ = self.global_attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, need_weights=False
        )
        x = residual + self.dropout(global_attn_output)

        return x


def main() -> None:
    config = TransformerConfig(
        d_model=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        attention_dropout=0.1,
        modalities=['text', 'image', 'speech']
    )

    model = MixtureOfTransformers(config)

    batch_size = 32
    seq_length = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs: Dict[str, torch.Tensor] = {
        modality: torch.randn(batch_size, seq_length, config.d_model).to(device)
        for modality in config.modalities
    }

    key_padding_mask: torch.Tensor = torch.zeros(batch_size, seq_length, dtype=torch.bool).to(device)

    model.eval()
    with torch.no_grad():
        for modality, input_tensor in inputs.items():
            output = model(input_tensor, modality, key_padding_mask=key_padding_mask)
            print(f"{modality.capitalize()} Output Shape: {output.shape}")


if __name__ == "__main__":
    main()
