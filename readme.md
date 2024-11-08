# Mixture-of-Transformers (MoT) Model

## Introduction

The **Mixture-of-Transformers (MoT)** model is a multimodal architecture designed to process and integrate information from multiple modalities, such as text, images, and speech. By employing modality-specific transformer layers alongside a shared global attention mechanism, the MoT model effectively captures both modality-specific patterns and cross-modal interactions.

## Architecture Overview

The MoT model consists of the following key components:

1. **Modality-Specific Transformer Layers**: Separate transformer encoder layers for each modality, allowing the model to learn modality-specific representations.
2. **Shared Global Self-Attention Layer**: A global attention mechanism that integrates information across all modalities to capture cross-modal dependencies.
3. **Positional Encoding**: Sinusoidal positional encodings added to the input embeddings to retain positional information.
4. **Modality Embeddings**: Learnable embeddings that provide the model with information about the modality of each input.

## Mathematical Formulation

### Transformer Encoder Layer

A standard Transformer encoder layer comprises a multi-head self-attention mechanism followed by a position-wise feed-forward network (FFN). The MoT model extends this architecture by making these layers modality-specific.

#### Multi-Head Self-Attention

For a given modality \( m \), the self-attention mechanism computes a weighted sum of value vectors, where the weights are determined by the compatibility of query and key vectors.

**Formulation:**

1. **Linear Projections:**

   The input sequence \( X^{(m)} \in \mathbb{R}^{T \times d_{\text{model}}} \) is projected to query \( Q^{(m)} \), key \( K^{(m)} \), and value \( V^{(m)} \) matrices:

   \[
   \begin{aligned}
   Q^{(m)} &= X^{(m)} W_Q^{(m)}, \\
   K^{(m)} &= X^{(m)} W_K^{(m)}, \\
   V^{(m)} &= X^{(m)} W_V^{(m)},
   \end{aligned}
   \]

   where \( W_Q^{(m)}, W_K^{(m)}, W_V^{(m)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \) are modality-specific projection matrices.

2. **Scaled Dot-Product Attention:**

   The attention scores are computed as:

   \[
   \text{Attention}(Q^{(m)}, K^{(m)}, V^{(m)}) = \text{softmax}\left( \frac{Q^{(m)} {K^{(m)}}^\top}{\sqrt{d_k}} \right) V^{(m)},
   \]

   where \( d_k = \frac{d_{\text{model}}}{h} \) is the dimensionality of each head, and \( h \) is the number of heads.

3. **Multi-Head Attention:**

   The multi-head attention concatenates the outputs from each head:

   \[
   \begin{aligned}
   \text{MultiHead}(Q^{(m)}, K^{(m)}, V^{(m)}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O^{(m)}, \\
   \text{where } \text{head}_i &= \text{Attention}(Q_i^{(m)}, K_i^{(m)}, V_i^{(m)}),
   \end{aligned}
   \]

   and \( W_O^{(m)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \) is a modality-specific output projection matrix.

#### Position-Wise Feed-Forward Network

Each position in the sequence is passed through a fully connected feed-forward network:

\[
\begin{aligned}
\text{FFN}(X^{(m)}) &= \sigma\left( X^{(m)} W_1^{(m)} + b_1^{(m)} \right) W_2^{(m)} + b_2^{(m)},
\end{aligned}
\]

where:

- \( W_1^{(m)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}} \),
- \( W_2^{(m)} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}} \),
- \( b_1^{(m)} \in \mathbb{R}^{d_{\text{ff}}} \),
- \( b_2^{(m)} \in \mathbb{R}^{d_{\text{model}}} \),
- \( \sigma \) is an activation function (e.g., GELU).

#### Layer Normalization and Residual Connections

Layer normalization and residual connections are applied to enhance training stability:

1. **Pre-Norm Architecture:**

   \[
   \begin{aligned}
   X'^{(m)} &= X^{(m)} + \text{MultiHead}\left( \text{LayerNorm}(X^{(m)}) \right), \\
   X''^{(m)} &= X'^{(m)} + \text{FFN}\left( \text{LayerNorm}(X'^{(m)}) \right).
   \end{aligned}
   \]

### Modality Embeddings and Positional Encoding

To incorporate modality information and positional context:

1. **Modality Embeddings:**

   Each modality \( m \) has a learnable embedding \( E_{\text{mod}}^{(m)} \in \mathbb{R}^{1 \times d_{\text{model}}} \), which is added to the input:

   \[
   X^{(m)} = X^{(m)} + E_{\text{mod}}^{(m)}.
   \]

2. **Positional Encoding:**

   Positional encodings \( E_{\text{pos}} \in \mathbb{R}^{T \times d_{\text{model}}} \) are added to retain sequence order information:

   \[
   X^{(m)} = X^{(m)} + E_{\text{pos}}.
   \]

   The positional encodings are computed using sinusoidal functions:

   \[
   \begin{aligned}
   E_{\text{pos}}(pos, 2i) &= \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right), \\
   E_{\text{pos}}(pos, 2i+1) &= \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right),
   \end{aligned}
   \]

   where \( pos \) is the position and \( i \) is the dimension index.

### Shared Global Self-Attention Layer

After processing through modality-specific layers, the representations are refined using a shared global self-attention mechanism to capture cross-modal interactions.

1. **Global Multi-Head Attention:**

   The global attention operates on the modality-specific outputs \( X''^{(m)} \):

   \[
   \begin{aligned}
   \hat{X}^{(m)} &= X''^{(m)} + \text{GlobalMultiHead}\left( \text{LayerNorm}(X''^{(m)}) \right),
   \end{aligned}
   \]

   where \( \text{GlobalMultiHead} \) shares parameters across modalities.

2. **Parameter Sharing:**

   The global attention uses shared projection matrices \( W_Q^{\text{global}}, W_K^{\text{global}}, W_V^{\text{global}} \) and \( W_O^{\text{global}} \).

### Overall Forward Pass

The complete forward pass for a modality \( m \) can be summarized as:

1. **Input Embeddings:**

   \[
   X^{(m)} = \text{Input}^{(m)} + E_{\text{mod}}^{(m)} + E_{\text{pos}}.
   \]

2. **Modality-Specific Layers:**

   For each layer \( l = 1, \dots, L \):

   \[
   \begin{aligned}
   X_l'^{(m)} &= X_{l-1}^{(m)} + \text{MultiHead}^{(m)}\left( \text{LayerNorm}(X_{l-1}^{(m)}) \right), \\
   X_l^{(m)} &= X_l'^{(m)} + \text{FFN}^{(m)}\left( \text{LayerNorm}(X_l'^{(m)}) \right).
   \end{aligned}
   \]

3. **Global Attention Layer:**

   \[
   \hat{X}^{(m)} = X_L^{(m)} + \text{GlobalMultiHead}\left( \text{LayerNorm}(X_L^{(m)}) \right).
   \]

4. **Output:**

   The final output \( \hat{X}^{(m)} \) represents the processed sequence for modality \( m \), incorporating both modality-specific and cross-modal information.
