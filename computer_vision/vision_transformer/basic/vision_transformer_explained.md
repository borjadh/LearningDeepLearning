# Vision Transformer (ViT) Implementation - Detailed Explanation

This document provides a comprehensive, line-by-line explanation of the Vision Transformer implementation for image classification.

## Table of Contents
1. [Overview](#overview)
2. [Configuration Dataclass](#configuration-dataclass)
3. [MLP Module](#mlp-module)
4. [Multi-Head Self-Attention (MSA)](#multi-head-self-attention-msa)
5. [Transformer Block](#transformer-block)
6. [Vision Transformer Model](#vision-transformer-model)
7. [Training Setup](#training-setup)

---

## Overview

The Vision Transformer (ViT) applies the Transformer architecture, originally designed for NLP, to computer vision tasks. The key idea is to split an image into patches, treat them as tokens, and process them through Transformer encoder layers.

**Key Components:**
- **Patch Embedding**: Splits images into fixed-size patches and projects them to embedding vectors
- **Multi-Head Self-Attention (MSA)**: Allows patches to attend to each other
- **MLP Blocks**: Feed-forward networks for processing
- **LayerNorm**: Normalization before each block
- **Residual Connections**: Skip connections for better gradient flow

---

## Configuration Dataclass

```python
@dataclass
class ViTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    out_dim = 10
    in_chans: int = 3
    height: int = 64
    width: int = 64
    patch_size: int = 16
    dropout: float = 0.1
```

### Line-by-Line Explanation:

**`@dataclass`**
- Python decorator that automatically generates `__init__`, `__repr__`, and other special methods
- Makes the configuration class cleaner and more readable

**`n_layer: int = 12`**
- Number of Transformer encoder blocks (layers) in the model
- Default: 12 (standard ViT-Base configuration)
- More layers → deeper model → more representational power

**`n_head: int = 12`**
- Number of attention heads in Multi-Head Self-Attention
- Default: 12 heads
- Multiple heads allow the model to attend to different representation subspaces

**`n_embd: int = 768`**
- Embedding dimension (hidden size) throughout the network
- Default: 768 (ViT-Base)
- All patch embeddings, attention outputs, and MLP layers use this dimension

**`out_dim = 10`**
- Output dimension for classification
- 10 classes (likely CIFAR-10 dataset: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**`in_chans: int = 3`**
- Number of input channels
- 3 for RGB color images (Red, Green, Blue)

**`height: int = 64` and `width: int = 64`**
- Input image dimensions
- 64x64 pixels (standard for CIFAR-10)

**`patch_size: int = 16`**
- Size of each square patch
- 16x16 patches
- Number of patches = (height / patch_size) × (width / patch_size) = (64/16) × (64/16) = 4 × 4 = 16 patches

**`dropout: float = 0.1`**
- Dropout probability for regularization
- 10% of neurons randomly dropped during training
- Helps prevent overfitting

---

## MLP Module

```python
class MLP(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))
```

### Purpose:
The MLP (Multi-Layer Perceptron) block is a position-wise feed-forward network applied after self-attention. It processes each patch embedding independently.

### Architecture:
Input → Linear (expand 4x) → GELU activation → Linear (project back) → Dropout → Output

### Line-by-Line Explanation:

**`class MLP(nn.Module):`**
- Defines the MLP class inheriting from PyTorch's nn.Module
- Standard PyTorch module pattern

**`def __init__(self, config: ViTConfig):`**
- Constructor takes configuration object
- `config` contains all hyperparameters

**`super().__init__()`**
- Calls parent class (nn.Module) initializer
- Required for proper PyTorch module initialization

**`self.fc = nn.Linear(config.n_embd, config.n_embd * 4)`**
- First fully-connected layer
- Input dimension: `n_embd` (768)
- Output dimension: `n_embd * 4` (3072)
- **Expansion**: 4x expansion is standard in Transformers
- Creates a "bottleneck" architecture for richer representations

**`self.proj = nn.Linear(config.n_embd * 4, config.n_embd)`**
- Projection layer (second linear transformation)
- Input dimension: `n_embd * 4` (3072)
- Output dimension: `n_embd` (768)
- **Compression**: Projects back to original dimension for residual connection

**`self.drop = nn.Dropout(config.dropout)`**
- Dropout layer for regularization
- Randomly zeros elements with probability `dropout` (0.1)
- Only active during training

**`def forward(self, x):`**
- Defines forward pass computation
- `x` shape: `(batch_size, num_patches + 1, n_embd)` where +1 is for the [CLS] token

**`return self.drop(self.proj(F.gelu(self.fc(x))))`**
- **Execution order** (inside-out):
  1. `self.fc(x)`: Apply first linear layer → shape: `(B, N, 3072)`
  2. `F.gelu(...)`: Apply GELU activation function
     - GELU (Gaussian Error Linear Unit): smooth activation, better than ReLU for Transformers
     - Formula: `x * Φ(x)` where Φ is cumulative distribution function of standard normal
  3. `self.proj(...)`: Apply projection layer → shape: `(B, N, 768)`
  4. `self.drop(...)`: Apply dropout for regularization

---

## Multi-Head Self-Attention (MSA)

```python
class MSA(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.attention = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd,config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
```

### Purpose:
Multi-Head Self-Attention allows each patch to attend to all other patches, learning spatial relationships across the image.

### Line-by-Line Explanation (__init__):

**`assert config.n_embd % config.n_head == 0`**
- **Assertion check**: Embedding dimension must be divisible by number of heads
- Example: 768 % 12 = 0 ✓
- Ensures equal distribution of dimensions across heads

**`self.n_head, self.n_embd = config.n_head, config.n_embd`**
- Store number of heads (12) and embedding dimension (768)
- Used throughout the forward pass

**`self.head_dim = self.n_embd // self.n_head`**
- Calculate dimension per attention head
- `head_dim = 768 // 12 = 64`
- Each head operates on 64-dimensional subspace

**`self.attention = nn.Linear(config.n_embd, 3 * config.n_embd)`**
- **Single linear layer** to compute Query, Key, and Value simultaneously
- Input: `n_embd` (768)
- Output: `3 * n_embd` (2304)
- Efficient computation: one matrix multiplication instead of three
- Output will be split into Q, K, V each of size `n_embd`

**`self.proj = nn.Linear(config.n_embd, config.n_embd)`**
- Output projection layer
- Projects concatenated multi-head outputs back to embedding dimension
- Allows the model to mix information from different heads

**`self.dropout = nn.Dropout(config.dropout)`**
- Dropout for regularization
- Applied after attention and after projection

### Forward Pass:

```python
    def forward(self, x):
        B, N, C = x.shape
```

**`B, N, C = x.shape`**
- Unpack input tensor dimensions:
  - `B`: Batch size (e.g., 64 images)
  - `N`: Number of patches + 1 for [CLS] token (e.g., 16 + 1 = 17)
  - `C`: Embedding dimension / channels (`n_embd` = 768)

```python
        qkv = self.attention(x).reshape(B, N, 3, self.n_head, self.head_dim)
```

**Breakdown:**
1. `self.attention(x)`: Apply linear transformation
   - Input shape: `(B, N, 768)`
   - Output shape: `(B, N, 2304)`

2. `.reshape(B, N, 3, self.n_head, self.head_dim)`:
   - Reshape to: `(B, N, 3, 12, 64)`
   - Dimension 2 (size 3): [Query, Key, Value]
   - Dimension 3 (size 12): 12 attention heads
   - Dimension 4 (size 64): 64 dimensions per head

```python
        qkv = qkv.permute(2, 0, 3, 1, 4)
```

**`permute(2, 0, 3, 1, 4)`**
- Rearrange dimensions for efficient computation
- Before: `(B, N, 3, n_head, head_dim)`
- After: `(3, B, n_head, N, head_dim)`
- Makes it easy to split into Q, K, V

```python
        Q, K, V = qkv[0], qkv[1], qkv[2]
```

**Split into Query, Key, Value:**
- `Q`: Query matrix, shape `(B, n_head, N, head_dim)` = `(B, 12, 17, 64)`
- `K`: Key matrix, shape `(B, n_head, N, head_dim)` = `(B, 12, 17, 64)`
- `V`: Value matrix, shape `(B, n_head, N, head_dim)` = `(B, 12, 17, 64)`
- Each head has separate Q, K, V projections

```python
        K_T = K.transpose(-2, -1)
```

**`K.transpose(-2, -1)`**
- Transpose last two dimensions of K
- Before: `(B, n_head, N, head_dim)` = `(B, 12, 17, 64)`
- After: `(B, n_head, head_dim, N)` = `(B, 12, 64, 17)`
- Needed for matrix multiplication with Q

```python
        attention = (Q @ K_T) / math.sqrt(self.head_dim)
```

**Attention Score Computation:**
- `Q @ K_T`: Matrix multiplication
  - Shape: `(B, 12, 17, 64) @ (B, 12, 64, 17)` = `(B, 12, 17, 17)`
  - Each element `(i,j)` represents similarity between patch i and patch j

- `/ math.sqrt(self.head_dim)`: **Scaling factor**
  - Divide by √64 = 8
  - **Why?** Prevents dot products from becoming too large
  - Keeps gradients stable during training
  - Derived from variance analysis of dot products

```python
        attention = F.softmax(attention, dim=-1)
```

**`F.softmax(attention, dim=-1)`**
- Apply softmax along last dimension (across N patches)
- Converts scores to probabilities: each row sums to 1
- Shape remains: `(B, 12, 17, 17)`
- **Interpretation**: For each patch, how much to attend to every other patch

```python
        attention = self.dropout(attention)
```

**Attention Dropout:**
- Randomly zero out entire attention connections
- **Attention dropout** is different from regular dropout
- Prevents overfitting to specific attention patterns

```python
        x = attention @ V
```

**Weighted Sum with Values:**
- Matrix multiplication: `(B, 12, 17, 17) @ (B, 12, 17, 64)` = `(B, 12, 17, 64)`
- Computes weighted sum of values based on attention weights
- Each patch now contains information from all patches it attended to

```python
        x = x.transpose(1, 2).reshape(B, N, C)
```

**Reshape back:**
1. `x.transpose(1, 2)`: `(B, 12, 17, 64)` → `(B, 17, 12, 64)`
   - Move head dimension after sequence dimension

2. `.reshape(B, N, C)`: `(B, 17, 12, 64)` → `(B, 17, 768)`
   - Concatenate all heads: 12 × 64 = 768
   - Back to original embedding dimension

```python
        x = self.dropout(self.proj(x))
        return x
```

**Final Projection:**
1. `self.proj(x)`: Linear projection `(B, 17, 768)` → `(B, 17, 768)`
   - Mixes information from different attention heads

2. `self.dropout(...)`: Dropout for regularization

3. `return x`: Output has same shape as input for residual connection

---

## Transformer Block

```python
class Block(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MSA(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

### Purpose:
A single Transformer encoder block combining self-attention and feed-forward layers with residual connections and LayerNorm.

### Line-by-Line Explanation:

**`self.ln1 = nn.LayerNorm(config.n_embd)`**
- First LayerNormalization layer
- Normalizes across the embedding dimension (768)
- **Pre-normalization**: Applied before attention (modern practice)
- Formula: `(x - mean) / sqrt(variance + ε)` then scale and shift

**`self.ln2 = nn.LayerNorm(config.n_embd)`**
- Second LayerNormalization layer
- Applied before MLP

**`self.attn = MSA(config)`**
- Multi-Head Self-Attention module
- Enables patch-to-patch interactions

**`self.mlp = MLP(config)`**
- Feed-forward MLP module
- Processes each patch independently

**`x = x + self.attn(self.ln1(x))`**
- **Execution order**:
  1. `self.ln1(x)`: Normalize input
  2. `self.attn(...)`: Apply attention
  3. `x + ...`: **Residual connection** (skip connection)
     - Adds original input to attention output
     - Helps gradients flow during backpropagation
     - Prevents vanishing gradients in deep networks

**`x = x + self.mlp(self.ln2(x))`**
- **Execution order**:
  1. `self.ln2(x)`: Normalize (using updated x from attention)
  2. `self.mlp(...)`: Apply MLP
  3. `x + ...`: **Second residual connection**

**`return x`**
- Output has same shape as input: `(B, N, C)`
- Allows stacking multiple blocks

---

## Vision Transformer Model

```python
class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
```

**Store configuration:**
- Keeps all hyperparameters accessible throughout the model

### Patch Embedding

```python
        self.patch_emb = nn.Conv2d(
            in_channels=config.in_chans,
            out_channels=config.n_embd,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
```

**`nn.Conv2d` for Patch Embedding:**
- **Clever trick**: Use convolution to split image into patches and embed them
- `in_channels=3`: RGB input
- `out_channels=768`: Embedding dimension
- `kernel_size=16`: 16×16 patches
- `stride=16`: No overlap between patches
- **Result**: Converts `(B, 3, 64, 64)` → `(B, 768, 4, 4)`
  - 4×4 = 16 patches
  - Each patch is embedded to 768 dimensions

### Class Token

```python
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
```

**`[CLS]` Token:**
- **Learnable parameter**: Initialized to zeros, trained during training
- Shape: `(1, 1, 768)`
- **Purpose**: Aggregates information from all patches
- After all Transformer layers, the [CLS] token contains global image representation
- Used for final classification

### Positional Encoding

```python
        num_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
```

**Calculate number of patches:**
- `(64 // 16) * (64 // 16) = 4 * 4 = 16` patches

```python
        self.position_emb = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.n_embd)
        )
```

**Positional Embeddings:**
- **Learnable 1D positional encodings**
- Shape: `(1, 17, 768)` where 17 = 16 patches + 1 [CLS] token
- **Why needed?** Self-attention is permutation-invariant
  - Without position info, model can't distinguish patch order
- **1D vs 2D**: Original ViT paper found 1D works as well as 2D
- Added to patch embeddings before Transformer layers

### Transformer Encoder

```python
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
```

**Stack of Transformer Blocks:**
- `*[...]`: Unpacks list into separate arguments
- Creates 12 identical Transformer blocks (n_layer=12)
- `nn.Sequential`: Applies blocks sequentially
- Each block has its own parameters (not shared)

### Classification Head

```python
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.out_dim)
```

**`self.ln`**: Final LayerNorm
- Applied to [CLS] token before classification

**`self.head`**: Classification layer
- Linear projection: 768 → 10 classes
- Produces logits for each class

### Forward Pass

```python
    def forward(self, x):
        B = x.shape[0]
```

**`B = x.shape[0]`**
- Batch size
- Input `x` shape: `(B, 3, 64, 64)`

```python
        x = self.patch_emb(x)
```

**Patch Embedding:**
- Input: `(B, 3, 64, 64)` RGB images
- Output: `(B, 768, 4, 4)` embedded patches
- Each 16×16 patch → 768-dim vector

```python
        x = x.flatten(2)
```

**Flatten spatial dimensions:**
- Input: `(B, 768, 4, 4)`
- `.flatten(2)`: Flatten from dimension 2 onwards
- Output: `(B, 768, 16)` where 16 = 4×4 patches

```python
        x = x.transpose(1, 2)
```

**Transpose:**
- Input: `(B, 768, 16)`
- Output: `(B, 16, 768)`
- **Reason**: Transformers expect (Batch, Sequence, Embedding) format

```python
        class_token = self.class_token.expand(B, -1, -1)
```

**Expand [CLS] Token:**
- `self.class_token`: Shape `(1, 1, 768)` - trainable parameter
- `.expand(B, -1, -1)`: Broadcast to batch size
- Output: `(B, 1, 768)` - same token copied for each image in batch
- `-1` means keep that dimension unchanged

```python
        x = torch.cat([class_token, x], dim=1)
```

**Concatenate [CLS] Token:**
- `class_token`: `(B, 1, 768)`
- `x`: `(B, 16, 768)`
- `torch.cat(..., dim=1)`: Concatenate along sequence dimension
- Output: `(B, 17, 768)` where 17 = 1 [CLS] + 16 patches
- [CLS] token is at position 0

```python
        x = x + self.position_emb
```

**Add Positional Embeddings:**
- `self.position_emb`: `(1, 17, 768)` - learnable parameter
- Broadcasting: automatically expands to `(B, 17, 768)`
- **Injects positional information** into patch embeddings
- Model can now distinguish between patches based on position

```python
        x = self.blocks(x)
```

**Apply Transformer Blocks:**
- Input: `(B, 17, 768)`
- Passes through all 12 Transformer blocks sequentially
- Each block: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
- Output: `(B, 17, 768)` - same shape, but contextualized representations

```python
        x = self.ln(x[:, 0])
```

**Extract and Normalize [CLS] Token:**
- `x[:, 0]`: Select first token (index 0) from sequence dimension
  - Shape: `(B, 768)`
  - This is the [CLS] token after all Transformer layers
  - Contains aggregated information from entire image
- `self.ln(...)`: Apply final LayerNorm

```python
        return self.head(x)
```

**Classification:**
- `self.head(x)`: Linear layer projects 768 → 10
- Output shape: `(B, 10)` - logits for 10 classes
- **No activation** (no softmax) - raw logits
  - CrossEntropyLoss will apply softmax internally

---

## Training Setup

```python
config = ViTConfig(
    n_layer=6,
    n_head=4,
    n_embd=512,
    patch_size=8,
    height=32,
    width=32
)
```

**Modified Configuration for CIFAR-10:**
- `n_layer=6`: Reduced from 12 (smaller model for 32×32 images)
- `n_head=4`: Reduced from 12
- `n_embd=512`: Reduced from 768 (fewer dimensions)
- `patch_size=8`: Smaller patches for smaller images (32×32)
- `height=32, width=32`: CIFAR-10 native resolution
- **Result**: 32×32 image → (32/8) × (32/8) = 4×4 = 16 patches

```python
model = VisionTransformer(config)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{pytorch_total_params:,} trainable parameters")
```

**Parameter Count:**
- `.parameters()`: All model parameters
- `p.requires_grad`: Only trainable parameters
- `p.numel()`: Number of elements in tensor
- Prints total number of parameters with comma separators

```python
vit_model = LightningModel(model)
trainer = L.Trainer(max_epochs=10, logger=CSVLogger(save_dir="logs/"))
trainer.fit(model=vit_model, datamodule=dm)
```

**Lightning Training:**
- `LightningModel`: Wrapper with training/validation logic
- `L.Trainer`: PyTorch Lightning trainer
  - `max_epochs=10`: Train for 10 epochs
  - `logger=CSVLogger(...)`: Log metrics to CSV
- `trainer.fit(...)`: Start training
  - Automatically handles train/val loops, device placement, etc.

---

## Key Mathematical Formulas

### Attention Mechanism

**Scaled Dot-Product Attention:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:
- \(Q\): Query matrix (B, n_head, N, head_dim)
- \(K\): Key matrix (B, n_head, N, head_dim)
- \(V\): Value matrix (B, n_head, N, head_dim)
- \(d_k\): Dimension per head (head_dim = 64)
- \(\sqrt{d_k}\): Scaling factor (√64 = 8)

### Transformer Block

**Pre-Normalization Residual Block:**

\[
\begin{align}
x' &= x + \text{MSA}(\text{LayerNorm}(x)) \\
x'' &= x' + \text{MLP}(\text{LayerNorm}(x'))
\end{align}
\]

### Patch Embedding

**Number of Patches:**

\[
N = \frac{H \times W}{P^2}
\]

Where:
- \(H\): Image height (32)
- \(W\): Image width (32)
- \(P\): Patch size (8)
- \(N\): Number of patches (16)

---

## Architecture Summary

**Input → Output Flow:**

1. **Input Image**: `(B, 3, 32, 32)` RGB image

2. **Patch Embedding**: 
   - Split into 16 patches of 8×8
   - Embed each to 512-dim → `(B, 16, 512)`

3. **Add [CLS] Token**: `(B, 17, 512)`

4. **Add Positional Encoding**: Still `(B, 17, 512)`

5. **Transformer Encoder** (6 layers):
   - Each layer: LayerNorm → MSA → Residual → LayerNorm → MLP → Residual
   - Output: `(B, 17, 512)`

6. **Classification Head**:
   - Extract [CLS] token: `(B, 512)`
   - LayerNorm + Linear → `(B, 10)` logits

7. **Output**: 10 class scores

---

## Key Design Choices

### 1. **Pre-Normalization (Pre-Norm) vs Post-Norm**
- This implementation uses **Pre-Norm**: `x + Attention(LayerNorm(x))`
- Alternative: Post-Norm: `LayerNorm(x + Attention(x))`
- **Advantage of Pre-Norm**: Better gradient flow, more stable training

### 2. **Learnable vs Fixed Positional Encodings**
- Uses **learnable** positional embeddings (trained from scratch)
- Alternative: Sinusoidal (fixed, hand-crafted)
- **Advantage**: Can adapt to data distribution

### 3. **[CLS] Token vs Global Average Pooling**
- Uses **[CLS] token** for classification
- Alternative: Average all patch embeddings
- **Advantage**: [CLS] token can learn optimal aggregation

### 4. **1D vs 2D Positional Encodings**
- Uses **1D** positional encodings
- Original ViT paper found no significant difference
- **Advantage**: Simpler, fewer parameters

### 5. **Convolutional Patch Embedding**
- Uses **Conv2d** for patch embedding
- Alternative: Reshape and linear projection
- **Advantage**: More efficient on GPUs

---

## Computational Complexity

**Attention Complexity:**
- \(O(N^2 \cdot d)\) where \(N\) = number of patches, \(d\) = embedding dimension
- For 16 patches: \(O(16^2 \cdot 512) = O(131,072)\) per layer
- **Quadratic in sequence length** - main bottleneck

**MLP Complexity:**
- \(O(N \cdot d^2)\)
- For 16 patches: \(O(16 \cdot 512^2) = O(4,194,304)\) per layer
- **Linear in sequence length**

---

## Comparison with CNNs

| Aspect | Vision Transformer | CNN |
|--------|-------------------|-----|
| **Inductive Bias** | Minimal (position embeddings only) | Strong (locality, translation equivariance) |
| **Receptive Field** | Global from layer 1 | Local, grows with depth |
| **Data Requirement** | High (needs large datasets) | Lower (strong priors help) |
| **Computational Cost** | \(O(N^2)\) in sequence length | \(O(N)\) in spatial dimensions |
| **Interpretability** | Attention maps show relationships | Feature maps less interpretable |

---

## Training Tips

1. **Data Augmentation**: Critical for small datasets like CIFAR-10
   - RandAugment, CutMix, MixUp

2. **Learning Rate**: Use warmup + cosine decay
   - Start small, warm up to peak, decay to near-zero

3. **Optimizer**: AdamW with weight decay
   - Weight decay ≈ 0.05-0.1

4. **Batch Size**: Larger is better (if GPU memory permits)
   - ViT benefits from large batch training

5. **Pre-training**: Transfer learning from ImageNet greatly helps
   - Fine-tune pre-trained weights rather than training from scratch

---

## References

1. **Original Paper**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)
1. **Framework**: PyTorch + Lightning

---

## Conclusion

This Vision Transformer implementation demonstrates how the Transformer architecture can be adapted for computer vision. Key innovations include:

- Treating image patches as sequence tokens
- Using [CLS] token for classification
- Leveraging self-attention for global context from layer 1
- Pre-normalization for stable training

The model achieves competitive performance on CIFAR-10 while being conceptually simpler than convolutional architectures.

