import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class TransformerConfig:
    """
    Configuration class for Transformer model parameters.
    """
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ffn_hidden_dim: Optional[int] = None
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layernorm_eps: float = 1e-5
    modalities: List[str] = None
    max_seq_length: int = 5000

    def __post_init__(self):
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = self.d_model * 4
        if self.modalities is None:
            self.modalities = ['text', 'image', 'speech']

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.
    """
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            torch.Tensor: Tensor with positional encodings added
        """
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length]
        return x

class FeedForward(nn.Module):
    """
    Position-wise FeedForward network.
    """
    def __init__(self, d_model: int, ffn_hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            torch.Tensor: Output tensor of the same shape
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

class ModalitySpecificTransformerLayer(nn.Module):
    """
    Modality-specific Transformer layer with separate attention and feed-forward networks for each modality.
    """
    def __init__(self, config: TransformerConfig):
        super(ModalitySpecificTransformerLayer, self).__init__()
        self.config = config

        # Modality-specific self-attention layers
        self.attn_layers = nn.ModuleDict({
            modality: nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True
            ) for modality in config.modalities
        })

        # Modality-specific feed-forward networks
        self.ffn_layers = nn.ModuleDict({
            modality: FeedForward(
                d_model=config.d_model,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout
            ) for modality in config.modalities
        })

        # Modality-specific layer norms
        self.norm1 = nn.ModuleDict({
            modality: nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
            for modality in config.modalities
        })
        self.norm2 = nn.ModuleDict({
            modality: nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
            for modality in config.modalities
        })

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the modality-specific transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)
            modality (str): Modality identifier
            attn_mask (Optional[torch.Tensor]): Attention mask
            key_padding_mask (Optional[torch.Tensor]): Key padding mask

        Returns:
            torch.Tensor: Output tensor of the same shape
        """
        if modality not in self.config.modalities:
            raise ValueError(f"Unknown modality '{modality}'. Expected one of: {self.config.modalities}")

        # Modality-specific self-attention
        residual = x
        x = self.norm1[modality](x)
        attn_output, _ = self.attn_layers[modality](
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = residual + self.dropout(attn_output)

        # Modality-specific feed-forward network
        residual = x
        x = self.norm2[modality](x)
        x = residual + self.dropout(self.ffn_layers[modality](x))

        return x

class MixtureOfTransformers(nn.Module):
    """
    Mixture-of-Transformers model with modality-specific layers and shared global attention.
    """
    def __init__(self, config: TransformerConfig):
        super(MixtureOfTransformers, self).__init__()
        self.config = config

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, max_seq_length=config.max_seq_length)

        # Modality embeddings (using nn.Embedding for scalability)
        self.modality_embedding = nn.Embedding(len(config.modalities), config.d_model)
        self.modality_to_index = {modality: idx for idx, modality in enumerate(config.modalities)}

        # Modality-specific Transformer layers
        self.layers = nn.ModuleList([
            ModalitySpecificTransformerLayer(config)
            for _ in range(config.num_layers)
        ])

        # Shared global self-attention layer
        self.global_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        self.global_norm = nn.LayerNorm(config.d_model, eps=config.layernorm_eps)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """
        Initializes model parameters using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Mixture-of-Transformers model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)
            modality (str): Modality identifier
            attn_mask (Optional[torch.Tensor]): Attention mask
            key_padding_mask (Optional[torch.Tensor]): Key padding mask

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model)
        """
        if modality not in self.modality_to_index:
            raise ValueError(f"Unknown modality '{modality}'. Expected one of: {list(self.modality_to_index.keys())}")

        batch_size, seq_length, _ = x.size()

        # Add modality embedding
        modality_idx = torch.full((batch_size, 1), self.modality_to_index[modality], device=x.device)
        modality_embed = self.modality_embedding(modality_idx)  # Shape: (batch_size, 1, d_model)
        x = x + modality_embed

        # Add positional encoding
        x = self.pos_encoding(x)

        # Modality-specific transformer layers
        for layer in self.layers:
            x = layer(
                x,
                modality=modality,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )

        # Global self-attention
        residual = x
        x = self.global_norm(x)
        global_attn_output, _ = self.global_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = residual + self.dropout(global_attn_output)

        return x

# Example usage
def main():
    # Configuration
    config = TransformerConfig(
        d_model=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        attention_dropout=0.1,
        modalities=['text', 'image', 'speech']
    )

    # Create model
    model = MixtureOfTransformers(config)

    # Sample inputs
    batch_size = 32
    seq_length = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate sample inputs for each modality
    inputs = {
        modality: torch.randn(batch_size, seq_length, config.d_model).to(device)
        for modality in config.modalities
    }

    # Optional key padding mask (shape: batch_size, seq_length)
    key_padding_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool).to(device)

    # Process each modality
    model.eval()
    with torch.no_grad():
        for modality, input_tensor in inputs.items():
            output = model(input_tensor, modality, key_padding_mask=key_padding_mask)
            print(f"{modality.capitalize()} Output Shape:", output.shape)

if __name__ == "__main__":
    main()
