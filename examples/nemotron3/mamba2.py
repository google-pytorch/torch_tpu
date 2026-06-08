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

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

# Local activation map replacing transformers.activations.ACT2FN
ACT2FN = {
    "silu": F.silu,
    "swish": F.silu,
}


class Zamba2RMSNormGated(nn.Module):
  """Group-wise gated RMSNorm from Zamba-2/Nemotron-3"""

  def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.group_size = group_size

  def forward(
      self, hidden_states: torch.Tensor, gate: torch.Tensor = None
  ) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    if gate is not None:
      hidden_states = hidden_states * F.silu(gate.to(torch.float32))

    *prefix_dims, last_dim = hidden_states.shape
    group_count = last_dim // self.group_size
    hidden_states_group = hidden_states.view(
        *prefix_dims, group_count, self.group_size
    )

    variance = hidden_states_group.pow(2).mean(-1, keepdim=True)
    hidden_states_group = hidden_states_group * torch.rsqrt(
        variance + self.variance_epsilon
    )

    hidden_states = hidden_states_group.view(
        *prefix_dims, group_count * self.group_size
    )
    return self.weight * hidden_states.to(input_dtype)


# Helper methods for Segment Sum chunking computation
def pad_tensor_by_size(
    input_tensor: torch.Tensor, pad_size: int
) -> torch.Tensor:
  """Padding tensor on the seq_len dim (dim=1)"""
  pad_shape = (
      (0, 0, 0, 0, 0, pad_size, 0, 0)
      if len(input_tensor.shape) == 4
      else (0, 0, 0, pad_size, 0, 0)
  )
  return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(
    input_tensor: torch.Tensor, pad_size: int, chunk_size: int
) -> torch.Tensor:
  """Pads and reshapes input_tensor into chunk sequences"""
  input_tensor = pad_tensor_by_size(input_tensor, pad_size)
  if len(input_tensor.shape) == 3:
    return input_tensor.reshape(
        input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
    )
  else:
    return input_tensor.reshape(
        input_tensor.shape[0],
        -1,
        chunk_size,
        input_tensor.shape[2],
        input_tensor.shape[3],
    )


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
  """Calculates segment sums using cumulative sums and triangular masking"""
  chunk_size = input_tensor.size(-1)
  input_tensor = input_tensor[..., None].expand(
      *input_tensor.size(), chunk_size
  )

  mask = torch.tril(
      torch.ones(
          chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
      ),
      diagonal=-1,
  )
  input_tensor = input_tensor.masked_fill(~mask, 0)
  tensor_segsum = torch.cumsum(input_tensor, dim=-2)

  mask = torch.tril(
      torch.ones(
          chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
      ),
      diagonal=0,
  )
  tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
  return tensor_segsum


class NemotronHMamba2Mixer(nn.Module):

  def __init__(self, config: Any, layer_idx: int | None = None):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.ssm_state_size = config.ssm_state_size
    self.conv_kernel_size = config.conv_kernel
    self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
    self.layer_idx = layer_idx
    self.use_conv_bias = config.use_conv_bias
    self.activation = config.mamba_hidden_act
    self.act = ACT2FN[config.mamba_hidden_act]

    self.n_groups = config.n_groups
    self.head_dim = config.mamba_head_dim
    self.num_heads = config.mamba_num_heads
    self.chunk_size = config.chunk_size

    self.time_step_limit = config.time_step_limit
    self.time_step_min = config.time_step_min
    self.time_step_max = config.time_step_max

    self.conv_dim = (
        self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
    )

    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim,
        out_channels=self.conv_dim,
        bias=config.use_conv_bias,
        kernel_size=self.conv_kernel_size,
        groups=self.conv_dim,
        padding=self.conv_kernel_size - 1,
    )

    projection_size = self.intermediate_size + self.conv_dim + self.num_heads
    self.in_proj = nn.Linear(
        self.hidden_size,
        projection_size,
        bias=config.use_bias,
    )

    self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

    A = torch.arange(1, self.num_heads + 1)
    self.A_log = nn.Parameter(torch.log(A))

    self.norm = Zamba2RMSNormGated(
        self.intermediate_size,
        group_size=self.intermediate_size // self.n_groups,
        eps=config.layer_norm_epsilon,
    )
    self.D = nn.Parameter(torch.ones(self.num_heads))
    self.out_proj = nn.Linear(
        self.intermediate_size, self.hidden_size, bias=config.use_bias
    )

  def forward(
      self,
      input_states: torch.Tensor,
      cache_params: Any = None,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    if cache_params is not None and cache_params.has_previous_state(
        self.layer_idx
    ):
      projected_states = self.in_proj(input_states)
    else:
      if attention_mask is not None:
        input_states = (input_states * attention_mask[:, :, None]).to(dtype)
      projected_states = self.in_proj(input_states)

    d_mlp = (
        projected_states.shape[-1]
        - 2 * self.intermediate_size
        - 2 * self.n_groups * self.ssm_state_size
        - self.num_heads
    ) // 2

    _, _, gate, hidden_states, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
        dim=-1,
    )
    hidden_states = hidden_states.transpose(1, 2)

    use_precomputed_state = (
        cache_params is not None
        and cache_params.has_previous_state(self.layer_idx)
    )

    # 1D Convolution sequence transformation
    if use_precomputed_state:
      conv_state = cache_params.update_conv_state(hidden_states, self.layer_idx)
      hidden_states = torch.sum(
          conv_state * self.conv1d.weight[:, 0, :], dim=-1
      )
      if self.use_conv_bias:
        hidden_states += self.conv1d.bias
      hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]
    else:
      if cache_params is not None:
        conv_state = F.pad(
            hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
        )
        conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
      hidden_states = self.act(
          self.conv1d(hidden_states)[..., :seq_len].transpose(1, 2)
      )
      if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    hidden_states, B, C = torch.split(
        hidden_states,
        [
            self.intermediate_size,
            self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size,
        ],
        dim=-1,
    )
    A = -torch.exp(self.A_log.float())

    if use_precomputed_state:
      dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
      dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
      dt_bias = self.dt_bias[..., None].expand(
          self.dt_bias.shape[0], self.head_dim
      )

      dt = F.softplus(dt + dt_bias.to(dt.dtype))
      dt = torch.clamp(dt, self.time_step_min)
      A = (
          A[..., None, None]
          .expand(self.num_heads, self.head_dim, self.ssm_state_size)
          .to(dtype=torch.float32)
      )
      dA = torch.exp(dt[..., None] * A)

      B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
      B = B.expand(
          batch_size,
          self.n_groups,
          self.num_heads // self.n_groups,
          B.shape[-1],
      ).contiguous()
      B = B.reshape(batch_size, -1, B.shape[-1])
      dB = dt[..., None] * B[..., None, :]

      hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
      dBx = dB * hidden_states[..., None]

      ssm_states = cache_params.layers[self.layer_idx].recurrent_states.clone()
      ssm_states = ssm_states * dA + dBx
      ssm_states = cache_params.update_recurrent_state(
          ssm_states, self.layer_idx
      )

      C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
      C = C.expand(
          batch_size,
          self.n_groups,
          self.num_heads // self.n_groups,
          C.shape[-1],
      ).contiguous()
      C = C.reshape(batch_size, -1, C.shape[-1])

      ssm_states = ssm_states.to(C.dtype)
      ssm_states_reshaped = ssm_states.view(
          batch_size * self.num_heads, self.head_dim, self.ssm_state_size
      )
      C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
      y = torch.bmm(ssm_states_reshaped, C_reshaped)
      y = y.view(batch_size, self.num_heads, self.head_dim)

      D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
      y = (y + hidden_states * D).to(y.dtype)

      y = y.reshape(batch_size, -1)[:, None, ...]
    else:
      dt = F.softplus(dt + self.dt_bias)
      dt = torch.clamp(dt, self.time_step_min)
      hidden_states = hidden_states.reshape(
          batch_size, seq_len, -1, self.head_dim
      ).float()
      B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
      C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
      B = B.repeat_interleave(
          self.num_heads // self.n_groups, dim=2, output_size=self.num_heads
      )
      C = C.repeat_interleave(
          self.num_heads // self.n_groups, dim=2, output_size=self.num_heads
      )
      pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

      D_residual = self.D[..., None] * pad_tensor_by_size(
          hidden_states, pad_size
      )

      hidden_states = hidden_states * dt[..., None]
      A = A.to(hidden_states.dtype) * dt

      hidden_states, A, B, C = [
          reshape_into_chunks(t, pad_size, self.chunk_size)
          for t in (hidden_states, A, B, C)
      ]

      A = A.permute(0, 3, 1, 2)
      A_cumsum = torch.cumsum(A, dim=-1)

      # 1. Compute the output for each intra-chunk (diagonal blocks)
      L = torch.exp(segment_sum(A))

      # Contraction of C and B to get G (attention-like weights)
      G_intermediate = (
          C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
      )  # (b, c, l, s, h, n)
      G = G_intermediate.sum(dim=-1)  # (b, c, l, s, h)

      # Apply decays
      M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
      M = M_intermediate.sum(dim=-1)

      Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

      # 2. Compute inter-chunk off-diagonal block states
      decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
      B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
      states = (
          (
              B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]
              * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]
          )
          .sum(dim=3)
          .permute(0, 1, 2, 4, 3)
      )
      previous_states = torch.zeros_like(states[:, :1])
      states = torch.cat([previous_states, states], dim=1)
      decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))

      states_permuted = states.permute(0, 2, 1, 3, 4)
      result = (
          decay_chunk[..., None, None] * states_permuted[:, :, None, ...]
      ).sum(dim=2)
      new_states = result.permute(0, 2, 1, 3, 4)
      states, ssm_state = new_states[:, :-1], new_states[:, -1]

      # Compute off-diagonal outputs
      state_decay_out = torch.exp(A_cumsum)
      C_times_states = C[..., None, :] * states[:, :, None, ...]
      state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
      Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

      y = Y_diag + Y_off
      y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
      y = y + D_residual
      if pad_size > 0:
        y = y[:, :seq_len, :, :]
      y = y.reshape(batch_size, seq_len, -1)
      if ssm_state is not None and cache_params is not None:
        cache_params.update_recurrent_state(ssm_state, self.layer_idx)

    scan_output = self.norm(y, gate)

    contextualized_states = self.out_proj(scan_output.to(dtype))
    return contextualized_states
