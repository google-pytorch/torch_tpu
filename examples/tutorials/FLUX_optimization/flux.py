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

"""Tutorial: Optimizing FLUX Diffusion Transformer (DoubleStreamBlock) on TPU.

================================================================================
OVERVIEW: WHY OPTIMIZE FLUX FOR TPU?
================================================================================
Modern diffusion transformers like FLUX combine dual-stream processing (image
patch tokens and text conditioning tokens), AdaLN-Zero continuous modulation,
and 3-axis Rotary Position Embeddings (RoPE) in joint multi-head self-attention.

While reference PyTorch implementations prioritize concise notation (e.g., using
`einops.rearrange` and explicit 5D rotation matrices), they introduce severe
performance bottlenecks on Cloud TPUs:

1. Zero-Copy Layout Views (`use_zero_copy_view`):
   Reference code unpacks fused QKV projections (`B L (K H D) -> K B L H D`) and
   reshapes attention outputs (`B H L D -> B L (H D)`) using `einops.rearrange`.
   Moving `K=3` to the outermost axis permutes tensor strides non-contiguously
   across batch and sequence dimensions, forcing the TPU runtime to execute
   expensive physical HBM-to-HBM layout copy kernels
   (`tt_jit_copy__as_strided_inverse`). Replacing this with
   `.view(B, L, 3, H, D)` + `torch.unbind(dim=2)` creates zero-copy slice views
   over the existing projection buffer without moving data in HBM.

2. Fused Algebraic 2D RoPE (`use_fused_rope`):
   Reference 3-axis RoPE constructs explicit 2x2 rotation matrices
   `[[cos, -sin], [sin, cos]]` reshaped into 5D tensors `(B, S, 1, D/2, 2, 2)`.
   Applying RoPE via 5D matrix-vector contractions doubles positional embedding
   HBM traffic and requires expensive 5D tensor broadcasting/slicing. Storing
   compact `(cos, sin)` pairs in 4D `(B, S, 1, D/2, 2)` and directly evaluating
   `(x0*cos - x1*sin, x0*sin + x1*cos)` halves memory bandwidth and maps cleanly
   to fused multiply-add (FMA) instructions on TPU Vector Processing Units
   (VPUs).

3. Whole-Graph XLA Compilation (`compile`):
   A single `DoubleStreamBlock` executes dozens of small memory-bound ops (SiLU,
   linear projections, 6-way chunking, AdaLN scale/shift modulations, RMSNorm,
   and gated residual additions). Compiling the block with
   `torch.compile(..., backend="tpu")` traces the entire forward and backward
   pass into a single XLA HLO graph, eliminating host dispatch overhead and
   fusing elementwise chains directly into Matrix Multiply Unit (MXU) kernels.

================================================================================
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
"""

from dataclasses import dataclass
import math
from typing import cast
import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

rearrange = einops.rearrange


# ==============================================================================
# STEP 1: ADALN-ZERO MODULATION & QK NORMALIZATION
# ==============================================================================
#
# In FLUX, continuous conditioning (diffusion timestep embedding + pooled text
# embedding `vec`) modulates both the image stream and text stream before
# attention and before the feed-forward MLP.
#
# 1. `BaselineModulation` projects `silu(vec)` into 6 vectors per stream:
#    - `mod1` (`shift`, `scale`, `gate`) modulates the pre-attention LayerNorm
#      and gates the attention residual connection.
#    - `mod2` (`shift`, `scale`, `gate`) modulates the pre-MLP LayerNorm and
#      gates the MLP residual connection.
# 2. `QKNorm` applies Root Mean Square Normalization (`nn.RMSNorm`) separately
#    to query (`q`) and key (`k`) head dimensions prior to dot-product attention
#    to stabilize high-resolution transformer training.


@dataclass
class ModulationOut:
  """Container holding modulation parameters for AdaLN-Zero.

  Attributes:
    shift: Additive shift applied after LayerNorm: `(1 + scale) * norm(x) +
      shift`.
    scale: Multiplicative scale applied after LayerNorm.
    gate: Multiplicative gating factor applied to residual branch output before
      addition.

  References:
    - TorchTitan: `ModulationOut` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
  """

  shift: Tensor
  scale: Tensor
  gate: Tensor


class BaselineModulation(nn.Module):
  """AdaLN-Zero modulation layer for conditioning vectors (timesteps + text embeddings).

  Computes `ModulationOut` (`shift`, `scale`, `gate`) tuples for modulating
  LayerNorm outputs and scaling residual connections.

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
    # Project conditioning vector and unsqueeze sequence dim: (B, 1, 6 * hidden_size)
    out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
    mod1 = ModulationOut(*out[:3])
    mod2 = ModulationOut(*out[3:]) if self.is_double else None
    return mod1, mod2


class QKNorm(nn.Module):
  """QK-Normalization container wrapping query and key RMSNorm layers.

  Normalizes queries and keys per attention head prior to rotary embedding and
  dot-product attention, casting back to the value tensor's dtype.

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


# ==============================================================================
# STEP 2: POSITIONAL EMBEDDINGS (3-AXIS RoPE): BEFORE vs. AFTER
# ==============================================================================
#
# FLUX encodes 3D token positions (time/frame index, vertical patch coordinate,
# horizontal patch coordinate) by splitting the head dimension `D` across 3 axes
# (e.g., `axes_dim = (16, 56, 56)` for `head_dim = 128`) and applying 2D Rotary
# Position Embeddings (RoPE) to each coordinate axis.
#
# Comparison of Approaches:
#   - BEFORE (`rope_ref`, `apply_rope_ref`, `EmbedNDRef`):
#     Constructs explicit 2x2 rotation matrices `[[cos, -sin], [sin, cos]]` for
#     every frequency pair and rearranges into a 5D tensor `(B, S, 1, D/2, 2, 2)`.
#     Applying RoPE requires reshaping queries/keys to 6D `(B, S, H, D/2, 1, 2)`
#     and performing 5D matrix-vector contractions. This doubles positional
#     embedding HBM footprint (storing cos and sin twice) and generates complex
#     broadcasting/slicing instructions on TPU.
#
#   - AFTER (`rope`, `apply_rope`, `EmbedND`):
#     Stores only compact `[cos, sin]` pairs in a 4D tensor `(B, S, 1, D/2, 2)`
#     and evaluates the 2D algebraic rotation directly:
#       `x_out_0 = x_0 * cos - x_1 * sin`
#       `x_out_1 = x_0 * sin + x_1 * cos`
#     This halves HBM memory bandwidth for positional embeddings and compiles
#     directly into hardware fused multiply-add (FMA) vector instructions while
#     preserving bit-exact mathematical equivalence (`atol=0.0, rtol=0.0`).


def rope_ref(pos: Tensor, dim: int, theta: int = 10000) -> Tensor:
  """BEFORE (Reference): 5D rotation matrix RoPE frequency generator.

  In the reference implementation, you:
    1. Compute frequency angles `out = pos * omega` of shape `(B, S, D/2)`.
    2. Explicitly stack 4 trigonometric entries `[cos, -sin, sin, cos]` per
    pair.
    3. Rearrange into explicit 2x2 matrices of shape `(B, S, D/2, 2, 2)`.

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
  """BEFORE (Reference): 5D matrix-vector RoPE application.

  Reshapes queries and keys into 2D column vectors `(..., D/2, 1, 2)` and
  contracts against the 5D `freqs_cis` rotation matrices `(..., D/2, 2, 2)`.

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
  """BEFORE (Reference): N-dimensional 5D matrix positional embedding generator.

  Generates 5D rotation matrices for each coordinate axis and concatenates them
  along `dim=-3` (the frequency pair dimension), producing shape
  `(B, S, 1, head_dim // 2, 2, 2)`.

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


def rope(pos: Tensor, dim: int, theta: int = 10000) -> Tensor:
  """AFTER (Optimized): Compact 2D RoPE frequency generator.

  In the optimized implementation, you:
    1. Compute frequency angles `out = pos * omega` of shape `(B, S, D/2)`.
    2. Stack only `[cos, sin]` into a compact `(B, S, D/2, 2)` tensor.
       No duplicate `-sin`/`cos` entries or `einops.rearrange` 2x2 matrix
       materialization is required, cutting memory footprint by 50%.

  References:
    - TorchTitan: `rope` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
  """
  assert dim % 2 == 0
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  cos = torch.cos(out).float()
  sin = torch.sin(out).float()
  return torch.stack([cos, sin], dim=-1)


def apply_rope(
    xq: Tensor, xk: Tensor, freqs_cis: Tensor
) -> tuple[Tensor, Tensor]:
  """AFTER (Optimized): Algebraic 2D RoPE application on query and key tensors.

  Directly evaluates `(x0 * cos - x1 * sin, x0 * sin + x1 * cos)` via
  elementwise
  multiply-add operations:
    1. Unpack `cos` (`freqs_cis[..., 0]`) and `sin` (`freqs_cis[..., 1]`).
    2. View `xq` and `xk` as pairs `(B, S, H, D/2, 2)` with components `x0, x1`.
    3. Compute rotated components using standard elementwise arithmetic that
       fuses into single-cycle TPU VPU FMA instructions.

  References:
    - TorchTitan: `apply_rope` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
  """
  cos = freqs_cis[..., 0]
  sin = freqs_cis[..., 1]

  b, s, h, d = xq.shape
  d2 = d // 2
  xq_ = xq.float().view(b, s, h, d2, 2)
  xk_ = xk.float().view(b, s, h, d2, 2)

  xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
  xk_0, xk_1 = xk_[..., 0], xk_[..., 1]

  # Algebraic 2D rotation: bit-exact with 5D matrix contraction
  xq_out_0 = xq_0 * cos - xq_1 * sin
  xq_out_1 = xq_0 * sin + xq_1 * cos

  xk_out_0 = xk_0 * cos - xk_1 * sin
  xk_out_1 = xk_0 * sin + xk_1 * cos

  xq_out = (
      torch.stack([xq_out_0, xq_out_1], dim=-1).view(b, s, h, d).type_as(xq)
  )
  xk_out = (
      torch.stack([xk_out_0, xk_out_1], dim=-1).view(b, s, h, d).type_as(xk)
  )
  return xq_out, xk_out


class EmbedND(nn.Module):
  """AFTER (Optimized): Compact N-dimensional positional embedding generator.

  Concatenates compact `(B, S, D_axis/2, 2)` frequency tensors along `dim=-2`
  (the frequency pair dimension) without 2x2 matrix materialization, producing
  shape `(B, S, 1, head_dim // 2, 2)`.

  References:
    - TorchTitan: `EmbedND` from
      https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/layers.py
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
    # Note: Concatenates along dim=-2 (frequency dimension) because the trailing
    # dimension is size 2 ([cos, sin]) rather than 2x2 matrices.
    emb = torch.cat(
        [
            rope(ids[..., i], self.axes_dim[i], self.theta)
            for i in range(n_axes)
        ],
        dim=-2,
    )
    return emb.unsqueeze(2)


# ==============================================================================
# STEP 3: DOUBLE STREAM TRANSFORMER BLOCK & TENSOR LAYOUT OPTIMIZATION
# ==============================================================================
#
# `DoubleStreamBlockBaseline` implements the core dual-stream transformer block
# of FLUX, supporting both reference (`BEFORE`) and optimized (`AFTER`) codepaths
# controlled by `OptimizationConfig`:
#
# 1. `use_zero_copy_view`:
#    - BEFORE (`False`): Uses `einops.rearrange(qkv, "B L (K H D) -> K B L H D")`
#      to split QKV projections and `rearrange(attn, "B H L D -> B L (H D)")`
#      after attention. On TPU, moving `K=3` to dim 0 creates non-contiguous
#      strides that trigger physical HBM copy kernels (`tt_jit_copy__as_strided_inverse`).
#    - AFTER (`True`): Uses `qkv.view(B, L, 3, H, D)` + `torch.unbind(dim=2)` to
#      return zero-copy slice views referencing the existing buffer, and
#      `attn.transpose(1, 2).reshape(B, L, -1)` for attention output merging.
#
# 2. `use_fused_rope`:
#    - BEFORE (`False`): Calls `apply_rope_ref` with 5D rotation matrices.
#    - AFTER (`True`): Calls `apply_rope` with compact 2D algebraic rotations.
#
# 3. `compile`:
#    - When `True`, `maybe_compile()` wraps the module with
#      `torch.compile(self, backend="tpu", dynamic=False)`, fusing the entire
#      dual-stream block into a single XLA HLO program.


@dataclass
class OptimizationConfig:
  """Configuration flags toggling FLUX performance optimization milestones.

  Attributes:
    use_zero_copy_view: If True, uses `.view()` + `torch.unbind()` instead of
      `einops.rearrange` to eliminate HBM-to-HBM layout transposition kernels on
      TPU.
    use_fused_rope: If True, uses 2D algebraic RoPE (`apply_rope`) instead of 5D
      matrix contractions (`apply_rope_ref`).
    use_fused_sdpa: If True, uses unified SDPA attention kernels.
    compile: If True, lowers the block into a whole-graph XLA executable via
      `torch.compile(backend="tpu")`.
  """

  use_zero_copy_view: bool = False
  use_fused_rope: bool = False
  compile: bool = False


class DoubleStreamBlockBaseline(nn.Module):
  """FLUX DoubleStreamBlock supporting side-by-side baseline and optimized execution.

  Contains separate image (`img`) and text (`txt`) processing streams that:
    1. Compute continuous timestep/prompt AdaLN-Zero modulations (`shift`,
    `scale`, `gate`).
    2. Project modulated hidden states into query, key, and value (`QKV`)
    representations.
    3. Normalize queries and keys via `QKNorm` (`RMSNorm`).
    4. Concatenate text and image tokens along sequence dimension for Joint
    Attention
       with 3-axis Rotary Position Embeddings (RoPE).
    5. Split joint attention outputs back into text and image streams.
    6. Apply gated residual projections and gated feed-forward MLPs.

  References:
    - TorchTitan Test Baseline: `double_stream_block_forward_ref` from
      https://github.com/google-pytorch/torchtitan/blob/main/tests/unit_tests/test_flux_unbind.py
    - TorchTitan Model: `DoubleStreamBlock` from
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
      config: OptimizationConfig | None = None,
  ):
    super().__init__()
    self.config = config or OptimizationConfig()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    # 1. AdaLN-Zero Conditioning Modulations (6-way output per stream)
    self.img_mod = BaselineModulation(hidden_size, double=True)
    self.txt_mod = BaselineModulation(hidden_size, double=True)

    # 2. Parameter-free LayerNorms (modulated by AdaLN scale/shift)
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

    # 3. Per-head Query/Key RMS Normalization
    self.img_norm = QKNorm(self.head_dim, eps=1e-6)
    self.txt_norm = QKNorm(self.head_dim, eps=1e-6)

    # 4. Fused QKV Linear Projections and Output Projections
    self.img_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
    self.txt_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
    self.img_proj = nn.Linear(hidden_size, hidden_size, bias=True)
    self.txt_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    # 5. Gated Feed-Forward MLPs (Linear -> GELU -> Linear)
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
    # --------------------------------------------------------------------------
    # 1. Compute AdaLN-Zero Modulation Parameters from Conditioning Vector
    # --------------------------------------------------------------------------
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    assert img_mod2 is not None and txt_mod2 is not None

    # --------------------------------------------------------------------------
    # 2. Image Stream Preparation: Modulated LayerNorm + QKV Projection & Unpack
    # --------------------------------------------------------------------------
    img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
    img_qkv = self.img_qkv(img_modulated)
    if self.config.use_zero_copy_view:
      # AFTER (Optimized): Zero-copy metadata view.
      # Keeping dim=2 of size 3 adjacent to (num_heads, head_dim) preserves
      # contiguous memory layout; torch.unbind slices without HBM copy kernels.
      b, l, _ = img_qkv.shape
      img_qkv = img_qkv.view(b, l, 3, self.num_heads, self.head_dim)
      img_q, img_k, img_v = torch.unbind(img_qkv, dim=2)
    else:
      # BEFORE (Reference): einops.rearrange permutes K=3 to dim 0 (K B L H D),
      # forcing physical HBM-to-HBM layout transposition kernels on TPU.
      img_q, img_k, img_v = rearrange(
          img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
      )
    img_q, img_k = self.img_norm(img_q, img_k, img_v)

    # --------------------------------------------------------------------------
    # 3. Text Stream Preparation: Modulated LayerNorm + QKV Projection & Unpack
    # --------------------------------------------------------------------------
    txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
    txt_qkv = self.txt_qkv(txt_modulated)
    if self.config.use_zero_copy_view:
      # AFTER (Optimized): Zero-copy view (.view + torch.unbind)
      b, l, _ = txt_qkv.shape
      txt_qkv = txt_qkv.view(b, l, 3, self.num_heads, self.head_dim)
      txt_q, txt_k, txt_v = torch.unbind(txt_qkv, dim=2)
    else:
      # BEFORE (Reference): Physical layout transpose via einops.rearrange
      txt_q, txt_k, txt_v = rearrange(
          txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
      )
    txt_q, txt_k = self.txt_norm(txt_q, txt_k, txt_v)

    # --------------------------------------------------------------------------
    # 4. Joint Multi-Head Attention (Concat -> 3-Axis RoPE -> SDPA)
    # --------------------------------------------------------------------------
    # Concatenate text and image tokens along sequence dim: (B, L_txt + L_img, H, D)
    q = torch.cat((txt_q, img_q), dim=1)
    k = torch.cat((txt_k, img_k), dim=1)
    v = torch.cat((txt_v, img_v), dim=1)

    if self.config.use_fused_rope:
      # AFTER (Optimized): 2D algebraic RoPE directly evaluates
      # (x0*cos - x1*sin, x0*sin + x1*cos) via fused VPU multiply-adds.
      q, k = apply_rope(q, k, pe)
    else:
      # BEFORE (Reference): 5D matrix-vector contraction over (B, S, 1, D/2, 2, 2).
      q, k = apply_rope_ref(q, k, pe)

    # Transpose to (B, H, L, D) for PyTorch scaled_dot_product_attention
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, is_causal=False
    )

    if self.config.use_zero_copy_view:
      # AFTER (Optimized): Native transpose + contiguous reshape
      b, _, l, _ = attn.shape
      attn = attn.transpose(1, 2).reshape(b, l, -1)
    else:
      # BEFORE (Reference): einops string rearrangement
      attn = rearrange(attn, "B H L D -> B L (H D)")

    # --------------------------------------------------------------------------
    # 5. Split Joint Attention Output Back into Text and Image Streams
    # --------------------------------------------------------------------------
    txt_len = txt.shape[1]
    txt_attn, img_attn = attn[:, :txt_len], attn[:, txt_len:]

    # --------------------------------------------------------------------------
    # 6. Gated Residual Connections + Modulated Feed-Forward MLPs
    # --------------------------------------------------------------------------
    img = img + img_mod1.gate * self.img_proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp(
        (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
    )

    txt = txt + txt_mod1.gate * self.txt_proj(txt_attn)
    txt = txt + txt_mod2.gate * self.txt_mlp(
        (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
    )

    return img, txt

  def maybe_compile(self) -> nn.Module:
    """Returns a compiled instance with `torch.compile(backend='tpu')` if enabled."""
    if self.config.compile:
      return cast(nn.Module, torch.compile(self, backend="tpu", dynamic=False))
    return self


# Convenient alias so users can instantiate either DoubleStreamBlock or DoubleStreamBlockBaseline
DoubleStreamBlock = DoubleStreamBlockBaseline


def build_double_stream_block(
    hidden_size: int = 1024,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    config: OptimizationConfig | None = None,
) -> nn.Module:
  """Builds a `DoubleStreamBlock`, lowering via `torch.compile` if `config.compile` is True."""
  config = config or OptimizationConfig()
  block = DoubleStreamBlock(
      hidden_size=hidden_size,
      num_heads=num_heads,
      mlp_ratio=mlp_ratio,
      qkv_bias=qkv_bias,
      config=config,
  )
  return block.maybe_compile()


def generate_baseline_inputs(
    batch_size: int = 2,
    img_seq_len: int = 256,
    txt_seq_len: int = 64,
    hidden_size: int = 512,
    num_heads: int = 8,
    axes_dim: tuple[int, ...] = (16, 24, 24),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    use_fused_rope: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """Generates synthetic image, text, conditioning, and RoPE tensors for testing/benchmarking.

  Args:
    batch_size: Number of samples in the batch (`B`).
    img_seq_len: Number of image patch tokens (`L_img`).
    txt_seq_len: Number of text conditioning tokens (`L_txt`).
    hidden_size: Model hidden dimension (`H * D`).
    num_heads: Number of attention heads (`H`).
    axes_dim: Tuple of per-axis RoPE dimensions summing to `head_dim` (`D`).
    device: Target PyTorch device (e.g., `"tpu"` or `"cpu"`).
    dtype: Floating-point data type for hidden states.
    use_fused_rope: If True, generates compact 4D `(..., D/2, 2)` positional
      embeddings via `EmbedND`. If False, generates 5D `(..., D/2, 2, 2)`
      reference rotation matrices via `EmbedNDRef`.

  Returns:
    Tuple of `(img, txt, vec, pe)` tensors placed on `device`.
  """
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
  if use_fused_rope:
    embed = EmbedND(dim=head_dim, theta=10000, axes_dim=axes_dim)
  else:
    embed = EmbedNDRef(dim=head_dim, theta=10000, axes_dim=axes_dim)
  pe = embed(ids).to(device=device, dtype=dtype)

  return img, txt, vec, pe
