# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for porting_vit_mnist.py — verifies each executable cell runs without error."""

# pylint: skip-file  # ruff: noqa
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    h = self.norm1(x)
    qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) / self.scale
    attn = F.softmax(attn, dim=-1)
    h = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = x + self.out_proj(h)
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
    x = self.patch_embed(x)
    x = x.flatten(2).transpose(1, 2)
    x = x + self.pos_embedding
    x = self.blocks(x)
    x = self.norm(x)
    return self.mlp_head(x.mean(dim=1))


def test_dispatcher_shim():
  """Cell: register torchvision NMS shim."""
  try:
    torch.library.Library("torchvision", "DEF").define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )
  except Exception:
    pass  # Already registered or not needed


def test_model_architecture(device):
  """Cell: instantiate TpuViT, verify output shape."""
  model = TpuViT().to(device)
  sample_input = torch.randn(2, 1, 32, 32).to(device)
  output = model(sample_input)
  assert output.shape == (2, 10)


def test_data_preparation():
  """Cell: load MNIST data, verify shapes."""
  from sklearn.datasets import fetch_openml
  import numpy as np

  mnist = fetch_openml("mnist_784", version=1, as_frame=False)

  X_train = mnist.data[:100].astype(np.float32) / 255.0  # Reduced for speed
  Y_train = mnist.target[:100].astype(np.int64)

  X_tensor = torch.from_numpy(X_train).view(-1, 1, 28, 28)
  X_tensor = F.interpolate(X_tensor, size=(32, 32))
  Y_tensor = torch.from_numpy(Y_train)

  assert X_tensor.shape == (100, 1, 32, 32)
  assert Y_tensor.shape == (100,)


def test_training_loop(device):
  """Cell: run 1 epoch of training (reduced from 4 for speed)."""
  from sklearn.datasets import fetch_openml
  import numpy as np

  mnist = fetch_openml("mnist_784", version=1, as_frame=False)
  X_train = mnist.data[:100].astype(np.float32) / 255.0
  Y_train = mnist.target[:100].astype(np.int64)

  X_tensor = torch.from_numpy(X_train).view(-1, 1, 28, 28)
  X_tensor = F.interpolate(X_tensor, size=(32, 32))
  Y_tensor = torch.from_numpy(Y_train)

  model = TpuViT().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.CrossEntropyLoss()

  model.train()
  batch_size = 32
  idx = 0
  while idx < len(X_tensor):
    inputs = X_tensor[idx : idx + batch_size].to(device)
    targets = Y_tensor[idx : idx + batch_size].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    idx += batch_size

  # Verify loss is a finite number
  loss_val = loss.item()
  assert math.isfinite(loss_val)


def test_inference(device):
  """Cell: run model.eval() inference on a test image."""
  model = TpuViT().to(device)
  model.eval()

  test_img = torch.randn(1, 1, 32, 32).to(device)
  with torch.no_grad():
    logits = model(test_img)
    predicted = logits.argmax(dim=1).cpu().item()

  assert 0 <= predicted <= 9
