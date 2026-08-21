---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
tags: [remove-input]
---
import torch
import warnings
# Suppress internal PJRT warnings about non-leaf grad access during training
warnings.filterwarnings(
    "ignore",
    message=(
        "The .grad attribute of a Tensor that is not a leaf Tensor is being"
        " accessed"
    ),
)
```

# Training a Vision Transformer (ViT) with TorchTPU

This tutorial demonstrates how to train a standard Vision Transformer (ViT) model on the TorchTPU backend. We'll implement a "Tiny ViT" optimized for TPU performance and train it on the MNIST dataset.

+++

## 1. Model Architecture (Optimized for TPU)
We define a standard VIT architecture.

```{code-cell} ipython3
import torch.nn as nn
import torch.nn.functional as F
import math

class TpuAttentionBlock(nn.Module):
  """Manual multi-head attention — avoids SDP MATH fallback on TPU."""

  def __init__(self, hidden_dim, num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = hidden_dim // num_heads
    self.scale = math.sqrt(self.head_dim)
    self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
    self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.norm2 = nn.LayerNorm(hidden_dim)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 4),
        nn.GELU(),
        nn.Linear(hidden_dim * 4, hidden_dim),
    )

  def forward(self, x):
    B, N, C = x.shape
    # Self-attention with residual
    h = self.norm1(x)
    qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) / self.scale
    attn = F.softmax(attn, dim=-1)
    h = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = x + self.out_proj(h)
    # FFN with residual
    x = x + self.mlp(self.norm2(x))
    return x

class TpuViT(nn.Module):

  def __init__(
      self,
      image_size=32,
      patch_size=4,
      num_layers=4,
      hidden_dim=128,
      num_heads=4,
  ):
    super().__init__()
    assert image_size % patch_size == 0
    num_patches = (image_size // patch_size) ** 2

    # Conv2d patch embedding: kernel_size=stride=patch_size
    self.patch_embed = nn.Conv2d(
        1, hidden_dim, kernel_size=patch_size, stride=patch_size
    )
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

    self.blocks = nn.Sequential(
        *[TpuAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)]
    )
    self.norm = nn.LayerNorm(hidden_dim)
    self.mlp_head = nn.Linear(hidden_dim, 10)

  def forward(self, x):
    # x shape: [B, 1, 32, 32]
    x = self.patch_embed(x)
    x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

    x = x + self.pos_embedding
    x = self.blocks(x)
    x = self.norm(x)

    return self.mlp_head(x.mean(dim=1))
```

## 2. Data Preparation
We fetch the MNIST dataset and prepare it as PyTorch tensors. We resize the images to 32x32 to match our model's expected input size.

```{code-cell} ipython3
from sklearn.datasets import fetch_openml
import numpy as np

print("Fetching MNIST data...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

# Train: first 1000, Test: next 200
X_train = mnist.data[:1000].astype(np.float32) / 255.0
Y_train = mnist.target[:1000].astype(np.int64)
X_test = mnist.data[1000:1200].astype(np.float32) / 255.0
Y_test = mnist.target[1000:1200].astype(np.int64)

# Resize 28x28 -> 32x32 using core functional ops
X_tensor = torch.from_numpy(X_train).view(-1, 1, 28, 28)
X_tensor = F.interpolate(X_tensor, size=(32, 32))
Y_tensor = torch.from_numpy(Y_train)

X_test_tensor = torch.from_numpy(X_test).view(-1, 1, 28, 28)
X_test_tensor = F.interpolate(X_test_tensor, size=(32, 32))
Y_test_tensor = torch.from_numpy(Y_test)

print(f"Train: {len(X_tensor)} samples, Test: {len(X_test_tensor)} samples")
```

## 3. Training Loop
Finally, we initialize the TPU device, move the model to the device, and run the training loop.

```{code-cell} ipython3
device = torch.device("tpu")
print(f"Executing on: {device}")
```

```{code-cell} ipython3
# Move model to TPU (float32 — required by SDP attention fast path)
model = TpuViT().to("tpu")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Starting training loop...")
model.train()
for epoch in range(
    5
):  # Change to at least 30 to see the model start to converge.
  idx = 0
  batch_size = 32

  # We accumulate loss using .detach() to avoid materializing the tensor back
  # to the CPU during every iteration. This avoids stalling the CPU and
  # amortizes the expensive TPU -> CPU DMA transfer.
  epoch_loss = 0.0

  while idx < len(X_tensor):
    inputs = X_tensor[idx : idx + batch_size].to("tpu")
    targets = Y_tensor[idx : idx + batch_size].to("tpu")

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Detach the loss tensor and accumulate to avoid frequent host-device syncs
    epoch_loss += loss.detach()

    idx += batch_size

  # Materialize the aggregated loss once per epoch
  avg_loss = (epoch_loss / (len(X_tensor) / batch_size)).item()
  print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

print("\n" + "=" * 40)
print("Verification complete: torchtpu is functional")
print("=" * 40)
```

## 4. Inference Demo
Now that the model is trained, let's run inference on a single test image and see what the model predicts. Adjust the epochs as needed to improve accuracy.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import random

# Pick a random test image
test_idx = random.randint(0, len(X_test_tensor) - 1)
test_img = X_test_tensor[test_idx : test_idx + 1]  # [1, 1, 32, 32]
true_label = Y_test_tensor[test_idx].item()

# Run inference on TPU
model.eval()
with torch.no_grad():
  logits = model(test_img.to(device))
  predicted = logits.argmax(dim=1).cpu().item()

result_text = "✅ Correct!" if predicted == true_label else "❌ Incorrect"
print(result_text)

# Render the image
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(test_img[0, 0].numpy(), cmap="gray")
color = "green" if predicted == true_label else "red"
ax.set_title(
    f"Predicted: {predicted}  |  Actual: {true_label}",
    fontsize=14,
    color=color,
)
ax.axis("off")
plt.tight_layout()
plt.show()
```
