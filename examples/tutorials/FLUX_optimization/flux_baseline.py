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

"""Unoptimized baseline FLUX model components for the performance optimization tutorial.

===============================================================================
CODEBASE & ARCHITECTURE REFERENCES
===============================================================================
This module establishes the reference starting point for the FLUX optimization
tutorial. It uses as reference the black forest labs reference
(https://github.com/black-forest-labs/flux/blob/main/src/flux/) and the
TorchTitan
implementation
(https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
as well as
Torchtitan reference tests
(https://github.com/google-pytorch/torchtitan/blob/main/tests/unit_tests/test_flux_rope.py
and
https://github.com/google-pytorch/torchtitan/blob/main/tests/unit_tests/test_flux_unbind.py).

===============================================================================
NON-OPTIMIZED REFERENCE CHARACTERISTICS (TO BE OPTIMIZED IN LATER TUTORIAL
STEPS)
===============================================================================
- RoPE: Constructs full 2x2 rotation matrices [cos, -sin, sin, cos] via einops
(5D tensor).
- Layout: Uses `einops.rearrange` for QKV packing/unpacking instead of `.view()`
+ `unbind()`.
- Attention Output: Uses `einops.rearrange` to reshape attention heads into
hidden dim.
- Modulation: Un-fused AdaLN-Zero scaling/shifting operations.
"""

from dataclasses import dataclass
import math
import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

rearrange = einops.rearrange


@dataclass
class ModulationOut:
  """Container holding modulation parameters for AdaLN-Zero.

  References:
    - TorchTitan: `ModulationOut` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
  """

  shift: Tensor
  scale: Tensor
  gate: Tensor


class BaselineModulation(nn.Module):
  """Reference AdaLN-Zero modulation for conditioning vectors (timesteps + text embeddings).

  Computes ModulationOut (shift, scale, gate) parameters for modulating
  LayerNorm and residual branches.

  References:
    - TorchTitan: `Modulation` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
  """

  def __init__(self, hidden_size: int, double: bool = True):
    super().__init__()
    self.is_double = double
    self.multiplier = 6 if double else 3
    self.lin = nn.Linear(hidden_size, self.multiplier * hidden_size, bias=True)

  def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
    out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
    mod1 = ModulationOut(*out[:3])
    mod2 = ModulationOut(*out[3:]) if self.is_double else None
    return mod1, mod2


class QKNorm(nn.Module):
  """QK-Normalization container wrapping query and key RMSNorm layers.

  References:
    - TorchTitan: `QKNorm` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
  """

  def __init__(self, head_dim: int, eps: float = 1e-6):
    super().__init__()
    self.query_norm = nn.RMSNorm(head_dim, eps=eps)
    self.key_norm = nn.RMSNorm(head_dim, eps=eps)

  def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    q = self.query_norm(q)
    k = self.key_norm(k)
    return q.to(v), k.to(v)


def rope_ref(pos: Tensor, dim: int, theta: int = 10000) -> Tensor:
  """Non-optimized reference 2D RoPE positional frequency generator.

  Constructs full 2x2 rotation matrices [cos, -sin, sin, cos] and rearranges
  into a 5D tensor.

  References:
    - TorchTitan Test Baseline: `rope_ref` from
      https://github.com/google-pytorch/torchtitan/blob/main/tests/unit_tests/test_flux_rope.py
  """
  assert dim % 2 == 0
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  out = torch.stack(
      [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
  )
  out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
  return out.float()


def apply_rope_ref(
    xq: Tensor, xk: Tensor, freqs_cis: Tensor
) -> tuple[Tensor, Tensor]:
  """Non-optimized reference 5D matrix-vector RoPE application.

  References:
    - TorchTitan Test Baseline: `apply_rope_ref` from
      https://github.com/google-pytorch/torchtitan/blob/main/tests/unit_tests/test_flux_rope.py
  """
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(
      *xk.shape
  ).type_as(xk)


class EmbedNDRef(nn.Module):
  """Non-optimized reference N-dimensional positional embedding generator.

  Concatenates 3 axes of spatial/temporal coordinates (e.g., axes_dim=(16, 56,
  56)).

  References:
    - TorchTitan Test Baseline: `EmbedND_ref` from
      https://github.com/google-pytorch/torchtitan/blob/main/tests/unit_tests/test_flux_rope.py#L24-L34
  """

  def __init__(
      self,
      dim: int,
      theta: int = 10000,
      axes_dim: tuple[int, ...] = (16, 56, 56),
  ):
    super().__init__()
    self.dim = dim
    self.theta = theta
    self.axes_dim = axes_dim

  def forward(self, ids: Tensor) -> Tensor:
    n_axes = ids.shape[-1]
    emb = torch.cat(
        [
            rope_ref(ids[..., i], self.axes_dim[i], self.theta)
            for i in range(n_axes)
        ],
        dim=-3,
    )
    return emb.unsqueeze(2)


class DoubleStreamBlockBaseline(nn.Module):
  """Non-optimized reference DoubleStreamBlock matching TorchTitan's reference unbind tests.

  Contains dual image/text processing streams that merge in multi-head
  self-attention
  and branch out for gated feed-forward processing.

  This is the section of code with most changes from the original TorchTitan
  implementation. It tries to stay as close as possible to the original
  implementation while making sections more readable for the tutorial. Changed
  sections are simplifications of the original TorchTitan code or based on the
  BFL reference implementation.

  References:
    - TorchTitan Test Baseline: `double_stream_block_forward_ref` from
      https://github.com/google-pytorch/torchtitan/blob/main/tests/unit_tests/test_flux_unbind.py
      for
      the forward pass.
    - TorchTitan Test Baseline: `DoubleStreamBlock` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
    - BFL FLUX Reference: `DoubleStreamBlock` from
      https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
  """

  def __init__(
      self,
      hidden_size: int = 1024,
      num_heads: int = 16,
      mlp_ratio: float = 4.0,
      qkv_bias: bool = True,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    # Modulations
    self.img_mod = BaselineModulation(hidden_size, double=True)
    self.txt_mod = BaselineModulation(hidden_size, double=True)

    # Layer Norms (AdaLN base)
    self.img_norm1 = nn.LayerNorm(
        hidden_size, elementwise_affine=False, eps=1e-6
    )
    self.img_norm2 = nn.LayerNorm(
        hidden_size, elementwise_affine=False, eps=1e-6
    )
    self.txt_norm1 = nn.LayerNorm(
        hidden_size, elementwise_affine=False, eps=1e-6
    )
    self.txt_norm2 = nn.LayerNorm(
        hidden_size, elementwise_affine=False, eps=1e-6
    )

    # QK Normalization (wrapped in QKNorm matching TorchTitan)
    self.img_norm = QKNorm(self.head_dim, eps=1e-6)
    self.txt_norm = QKNorm(self.head_dim, eps=1e-6)

    # Attention Projections
    self.img_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
    self.txt_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
    self.img_proj = nn.Linear(hidden_size, hidden_size, bias=True)
    self.txt_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    # Feed-Forward MLPs
    self.img_mlp = nn.Sequential(
        nn.Linear(hidden_size, mlp_hidden_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(mlp_hidden_dim, hidden_size),
    )
    self.txt_mlp = nn.Sequential(
        nn.Linear(hidden_size, mlp_hidden_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(mlp_hidden_dim, hidden_size),
    )

  def forward(
      self,
      img: Tensor,
      txt: Tensor,
      vec: Tensor,
      pe: Tensor,
  ) -> tuple[Tensor, Tensor]:
    # 1. Modulations
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    assert img_mod2 is not None and txt_mod2 is not None

    # 2. Image stream preparation (modulated norm + QKV projection + einops rearrange)
    img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
    img_qkv = self.img_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(
        img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
    )
    img_q, img_k = self.img_norm(img_q, img_k, img_v)

    # 3. Text stream preparation (modulated norm + QKV projection + einops rearrange)
    txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
    txt_qkv = self.txt_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(
        txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
    )
    txt_q, txt_k = self.txt_norm(txt_q, txt_k, txt_v)

    # 4. Joint Attention (Concat -> RoPE -> Attention matching test_flux_unbind.py)
    q = torch.cat((txt_q, img_q), dim=1)
    k = torch.cat((txt_k, img_k), dim=1)
    v = torch.cat((txt_v, img_v), dim=1)

    q, k = apply_rope_ref(q, k, pe)

    # Scaled Dot Product Attention
    q_sdpa = q.transpose(1, 2)  # (B, H, L, D)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, is_causal=False
    )
    # Non-optimized layout transformation using einops.rearrange
    attn = rearrange(attn, "B H L D -> B L (H D)")

    # 5. Split attention output between text and image
    txt_len = txt.shape[1]
    txt_attn, img_attn = attn[:, :txt_len], attn[:, txt_len:]

    # 6. Residual + Gated MLP
    img = img + img_mod1.gate * self.img_proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp(
        (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
    )

    txt = txt + txt_mod1.gate * self.txt_proj(txt_attn)
    txt = txt + txt_mod2.gate * self.txt_mlp(
        (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
    )

    return img, txt


def generate_baseline_inputs(
    batch_size: int = 2,
    img_seq_len: int = 256,
    txt_seq_len: int = 64,
    hidden_size: int = 512,
    num_heads: int = 8,
    axes_dim: tuple[int, ...] = (16, 24, 24),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """Helper to generate synthetic inputs for testing and benchmarking."""
  head_dim = hidden_size // num_heads
  total_seq_len = img_seq_len + txt_seq_len

  img = torch.randn(
      batch_size, img_seq_len, hidden_size, device=device, dtype=dtype
  )
  txt = torch.randn(
      batch_size, txt_seq_len, hidden_size, device=device, dtype=dtype
  )
  vec = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

  ids = torch.randint(
      0, 1000, (batch_size, total_seq_len, 3), device=device, dtype=torch.long
  )
  embed = EmbedNDRef(dim=head_dim, theta=10000, axes_dim=axes_dim)
  pe = embed(ids).to(device=device, dtype=dtype)

  return img, txt, vec, pe
