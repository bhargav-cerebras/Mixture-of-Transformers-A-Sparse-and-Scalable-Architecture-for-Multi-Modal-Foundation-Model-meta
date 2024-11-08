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

In a given modality \( m \), the self-attention mechanism computes a weighted sum of value vectors. The weights are based on the compatibility between the query and key vectors.

1. **Linear Projections**:  
   The input sequence \( X^{(m)} \in \mathbb{R}^{T \times d_{\text{model}}} \) is projected into query \( Q^{(m)} \), key \( K^{(m)} \), and value \( V^{(m)} \) matrices:

   $$
   \begin{aligned}
   Q^{(m)} &= X^{(m)} W_Q^{(m)}, \\
   K^{(m)} &= X^{(m)} W_K^{(m)}, \\
   V^{(m)} &= X^{(m)} W_V^{(m)},
   \end{aligned}
   $$

   where \( W_Q^{(m)}, W_K^{(m)}, W_V^{(m)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \) are modality-specific projection matrices.

2. **Scaled Dot-Product Attention**:  
   The attention scores are calculated as:

   $$
   \text{Attention}(Q^{(m)}, K^{(m)}, V^{(m)}) = \text{softmax}\left( \frac{Q^{(m)} {K^{(m)}}^\top}{\sqrt{d_k}} \right) V^{(m)},
   $$

   where \( d_k = \frac{d_{\text{model}}}{h} \) is the dimensionality of each head, and \( h \) represents the number of heads.

3. **Multi-Head Attention**:  
   The outputs from each head are concatenated:

   $$
   \begin{aligned}
   \text{MultiHead}(Q^{(m)}, K^{(m)}, V^{(m)}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O^{(m)}, \\
   \text{where } \text{head}_i &= \text{Attention}(Q_i^{(m)}, K_i^{(m)}, V_i^{(m)}),
   \end{aligned}
   $$

   and \( W_O^{(m)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \) is a modality-specific output projection matrix.

#### Position-Wise Feed-Forward Network

Each position in the sequence goes through a fully connected feed-forward network:

$$
\begin{aligned}
\text{FFN}(X^{(m)}) &= \sigma\left( X^{(m)} W_1^{(m)} + b_1^{(m)} \right) W_2^{(m)} + b_2^{(m)},
\end{aligned}
$$

where:

- \( W_1^{(m)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}} \),
- \( W_2^{(m)} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}} \),
- \( b_1^{(m)} \in \mathbb{R}^{d_{\text{ff}}} \),
- \( b_2^{(m)} \in \mathbb{R}^{d_{\text{model}}} \),
- \( \sigma \) is an activation function, such as GELU.

#### Layer Normalization and Residual Connections

Layer normalization and residual connections are used for training stability:

1. **Pre-Norm Architecture**:

   $$
   \begin{aligned}
   X'^{(m)} &= X^{(m)} + \text{MultiHead}\left( \text{LayerNorm}(X^{(m)}) \right), \\
   X''^{(m)} &= X'^{(m)} + \text{FFN}\left( \text{LayerNorm}(X'^{(m)}) \right).
   \end{aligned}
   $$

### Modality Embeddings and Positional Encoding

Modality embeddings and positional encoding are added to the input sequences:

1. **Modality Embeddings**:  
   Each modality \( m \) has a learnable embedding \( E_{\text{mod}}^{(m)} \in \mathbb{R}^{1 \times d_{\text{model}}} \) added to the input:

   $$
   X^{(m)} = X^{(m)} + E_{\text{mod}}^{(m)}.
   $$

2. **Positional Encoding**:  
   Positional encodings \( E_{\text{pos}} \in \mathbb{R}^{T \times d_{\text{model}}} \) are used to encode sequence order information:

   $$
   X^{(m)} = X^{(m)} + E_{\text{pos}}.
   $$

   Using sinusoidal functions, they are computed as follows:

   $$
   \begin{aligned}
   E_{\text{pos}}(pos, 2i) &= \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right), \\
   E_{\text{pos}}(pos, 2i+1) &= \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right),
   \end{aligned}
   $$

   where \( pos \) represents the position and \( i \) is the dimension index.

### Shared Global Self-Attention Layer

After passing through modality-specific layers, representations are refined using a shared global self-attention mechanism that captures cross-modal dependencies.

1. **Global Multi-Head Attention**:  
   Operating on the modality-specific outputs \( X''^{(m)} \), the global attention is applied as follows:

   $$
   \begin{aligned}
   \hat{X}^{(m)} &= X''^{(m)} + \text{GlobalMultiHead}\left( \text{LayerNorm}(X''^{(m)}) \right),
   \end{aligned}
   $$

   where the global attention mechanism uses shared projection matrices across modalities: \( W_Q^{\text{global}}, W_K^{\text{global}}, W_V^{\text{global}} \), and \( W_O^{\text{global}} \).

### Overall Forward Pass

For each modality \( m \), the forward pass proceeds as follows:

1. **Input Embeddings**:

   $$
   X^{(m)} = \text{Input}^{(m)} + E_{\text{mod}}^{(m)} + E_{\text{pos}}.
   $$

2. **Modality-Specific Layers**:

   For each layer \( l = 1, \dots, L \):

   $$
   \begin{aligned}
   X_l'^{(m)} &= X_{l-1}^{(m)} + \text{MultiHead}^{(m)}\left( \text{LayerNorm}(X_{l-1}^{(m)}) \right), \\
   X_l^{(m)} &= X_l'^{(m)} + \text{FFN}^{(m)}\left( \text{LayerNorm}(X_l'^{(m)}) \right).
   \end{aligned}
   $$

3. **Global Attention Layer**:

   $$
   \hat{X}^{(m)} = X_L^{(m)} + \text{GlobalMultiHead}\left( \text{LayerNorm}(X_L^{(m)}) \right).
   $$

4. **Output**:  
   The final output \( \hat{X}^{(m)} \) represents the processed sequence for modality \( m \), capturing both modality-specific and cross-modal information.

---

### Notes on Mathematical Formatting:

- **Display Math**: All major equations and multi-line formulas are enclosed within `$$` to ensure they are displayed prominently and centered in Markdown viewers that support LaTeX.
  
- **Inline Math**: For simpler expressions within sentences, single `$` symbols are used.

- **Alignment**: The `aligned` environment is used within `$$` to properly align multi-line equations.

- **Consistency**: Ensured consistent use of LaTeX commands and notation throughout the document for clarity and readability.

### Rendering Tips:

- **Markdown Renderer**: Ensure that the Markdown renderer you are using supports LaTeX. Platforms like GitHub, GitLab, and many documentation tools (e.g., Jupyter Notebooks, MkDocs with appropriate plugins) support LaTeX rendering.

- **Escaping Characters**: If any issues arise with special characters, consider escaping them or using raw LaTeX blocks.