# Mixture-of-Transformers (MoT) Model

## Introduction

The **Mixture-of-Transformers (MoT)** model is a multimodal architecture designed to process and integrate information from various modalities, such as text, images, and speech. By employing modality-specific transformer layers combined with a shared global attention mechanism, the MoT model efficiently captures both unique modality-specific patterns and cross-modal interactions.

## Architecture Overview

The MoT model architecture includes several critical components:

1. **Modality-Specific Transformer Layers**: These separate transformer encoder layers process each modality independently, allowing the model to learn distinct representations for each modality.
2. **Shared Global Self-Attention Layer**: This global attention mechanism aggregates information across modalities to capture dependencies between them.
3. **Positional Encoding**: Sinusoidal positional encodings are added to input embeddings to retain sequence order information.
4. **Modality Embeddings**: Learnable embeddings indicate the modality of each input to the model.

## Mathematical Formulation

### Transformer Encoder Layer

Each transformer encoder layer includes a multi-head self-attention mechanism followed by a position-wise feed-forward network (FFN). MoT extends this architecture by adding modality-specific layers for each modality.

#### Multi-Head Self-Attention

In a given modality $m$, the self-attention mechanism computes a weighted sum of value vectors. The weights are based on the compatibility between the query and key vectors.

1. **Linear Projections**

   The input sequence $X^{(m)} \in \mathbb{R}^{T \times d_{\mathrm{model}}}$ is projected into query $Q^{(m)}$, key $K^{(m)}$, and value $V^{(m)}$ matrices:

   $$
   \begin{aligned}
   Q^{(m)} &= X^{(m)} W_Q^{(m)}, \\
   K^{(m)} &= X^{(m)} W_K^{(m)}, \\
   V^{(m)} &= X^{(m)} W_V^{(m)},
   \end{aligned}
   $$

   where $W_Q^{(m)}, W_K^{(m)}, W_V^{(m)} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}$ are modality-specific projection matrices.

2. **Scaled Dot-Product Attention**

   The attention scores are calculated as:

   $$
   \text{Attention}(Q^{(m)}, K^{(m)}, V^{(m)}) = \text{softmax}\left( \dfrac{Q^{(m)} {K^{(m)}}^\top}{\sqrt{d_k}} \right) V^{(m)},
   $$

   where $d_k = \dfrac{d_{\mathrm{model}}}{h}$ is the dimensionality of each head, and $h$ represents the number of heads.

3. **Multi-Head Attention**

   The outputs from each head are concatenated:

   $$
   \begin{aligned}
   \text{MultiHead}(Q^{(m)}, K^{(m)}, V^{(m)}) &= \mathrm{Concat}(\text{head}_1, \dots, \text{head}_h) \, W_O^{(m)}, \\
   \text{where} \quad \text{head}_i &= \text{Attention}(Q_i^{(m)}, K_i^{(m)}, V_i^{(m)}),
   \end{aligned}
   $$

   and $W_O^{(m)} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}$ is a modality-specific output projection matrix.

#### Position-Wise Feed-Forward Network

Each position in the sequence goes through a fully connected feed-forward network:

$$
\text{FFN}(X^{(m)}) = \sigma\left( X^{(m)} W_1^{(m)} + b_1^{(m)} \right) W_2^{(m)} + b_2^{(m)},
$$

where:

- $W_1^{(m)} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}$,
- $W_2^{(m)} \in \mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}$,
- $b_1^{(m)} \in \mathbb{R}^{d_{\mathrm{ff}}}$,
- $b_2^{(m)} \in \mathbb{R}^{d_{\mathrm{model}}}$,
- $\sigma$ is an activation function, such as GELU.

#### Layer Normalization and Residual Connections

Layer normalization and residual connections are used for training stability:

1. **Pre-Norm Architecture**

   $$
   \begin{aligned}
   X'^{(m)} &= X^{(m)} + \text{MultiHead}\left( \mathrm{LayerNorm}\left( X^{(m)} \right) \right), \\
   X''^{(m)} &= X'^{(m)} + \text{FFN}\left( \mathrm{LayerNorm}\left( X'^{(m)} \right) \right).
   \end{aligned}
   $$

### Modality Embeddings and Positional Encoding

Modality embeddings and positional encoding are added to the input sequences:

1. **Modality Embeddings**

   Each modality $m$ has a learnable embedding $E_{\mathrm{mod}}^{(m)} \in \mathbb{R}^{1 \times d_{\mathrm{model}}}$ added to the input:

   $$
   X^{(m)} = X^{(m)} + E_{\mathrm{mod}}^{(m)}.
   $$

2. **Positional Encoding**

   Positional encodings $E_{\mathrm{pos}} \in \mathbb{R}^{T \times d_{\mathrm{model}}}$ are used to encode sequence order information:

   $$
   X^{(m)} = X^{(m)} + E_{\mathrm{pos}}.
   $$

   Using sinusoidal functions, they are computed as follows:

   $$
   \begin{aligned}
   E_{\mathrm{pos}}(\text{pos}, 2i) &= \sin\left( \dfrac{\text{pos}}{10000^{\frac{2i}{d_{\mathrm{model}}}}} \right), \\
   E_{\mathrm{pos}}(\text{pos}, 2i+1) &= \cos\left( \dfrac{\text{pos}}{10000^{\frac{2i}{d_{\mathrm{model}}}}} \right),
   \end{aligned}
   $$

   where $\text{pos}$ represents the position index and $i$ is the dimension index.

### Shared Global Self-Attention Layer

After passing through modality-specific layers, representations are refined using a shared global self-attention mechanism that captures cross-modal dependencies.

1. **Global Multi-Head Attention**

   Operating on the modality-specific outputs $X''^{(m)}$, the global attention is applied as follows:

   $$
   \hat{X}^{(m)} = X''^{(m)} + \text{GlobalMultiHead}\left( \mathrm{LayerNorm}\left( X''^{(m)} \right) \right),
   $$

   where the global attention mechanism uses shared projection matrices across modalities: $W_Q^{\mathrm{global}}, W_K^{\mathrm{global}}, W_V^{\mathrm{global}}$, and $W_O^{\mathrm{global}}$.

### Overall Forward Pass

For each modality $m$, the forward pass proceeds as follows:

1. **Input Embeddings**

   $$
   X^{(m)} = \mathrm{Input}^{(m)} + E_{\mathrm{mod}}^{(m)} + E_{\mathrm{pos}}.
   $$

2. **Modality-Specific Layers**

   For each layer $l = 1, \dots, L$:

   $$
   \begin{aligned}
   X_l'^{(m)} &= X_{l-1}^{(m)} + \text{MultiHead}^{(m)}\left( \mathrm{LayerNorm}\left( X_{l-1}^{(m)} \right) \right), \\
   X_l^{(m)} &= X_l'^{(m)} + \text{FFN}^{(m)}\left( \mathrm{LayerNorm}\left( X_l'^{(m)} \right) \right).
   \end{aligned}
   $$

3. **Global Attention Layer**

   $$
   \hat{X}^{(m)} = X_L^{(m)} + \text{GlobalMultiHead}\left( \mathrm{LayerNorm}\left( X_L^{(m)} \right) \right).
   $$

4. **Output**

   The final output $\hat{X}^{(m)}$ represents the processed sequence for modality $m$, capturing both modality-specific and cross-modal information.
