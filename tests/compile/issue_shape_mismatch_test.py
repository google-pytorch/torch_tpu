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

from absl.testing import absltest
import torch
from torch import nn
import torch.nn.functional as F


class MinimalModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv_block = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.cross_attn = nn.MultiheadAttention(64, 4, batch_first=True)
    self.vit_to_cnn_proj = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.router = nn.Linear(128, 2)

  def forward(self, x_conv, x_vit):
    batch, channels, height, width = x_conv.shape
    x_conv_out = self.conv_block(x_conv)

    # This view creates an alias with a different shape
    cnn_seq = x_conv_out.flatten(2).transpose(1, 2)

    # cnn_seq is used in cross-attention and stored for backward pass
    vit_fused, _ = self.cross_attn(query=x_vit, key=cnn_seq, value=cnn_seq)
    x_vit_fused = x_vit + vit_fused

    vit_spatial = x_vit_fused.transpose(1, 2).reshape(
        batch, channels, height, width
    )
    cnn_fused = x_conv_out + self.vit_to_cnn_proj(vit_spatial)

    conv_pool = x_conv_out.mean(dim=[2, 3])
    vit_pool = x_vit.mean(dim=1)

    # Router prevents simple simplification and forces backward pass
    # through the view
    route_logits = self.router(torch.cat([conv_pool, vit_pool], dim=-1))
    route_weights = F.gumbel_softmax(route_logits, tau=1.0, hard=True, dim=-1)

    w_conv = route_weights[:, 0].view(batch, 1, 1, 1)
    w_vit = route_weights[:, 1].view(batch, 1, 1)

    x_conv_final = cnn_fused * w_conv
    x_vit_final = x_vit_fused * w_vit
    return x_conv_final.sum() + x_vit_final.sum()


class IssueShapeMismatchTest(absltest.TestCase):

  def test_view_bug(self):
    device = torch.device("tpu")
    model = MinimalModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=True)

    @torch.compile(backend="tpu", dynamic=False)
    def train_step(data1, data2):
      optimizer.zero_grad()
      loss = model(data1, data2)
      loss.backward()
      optimizer.step()
      return loss

    data1 = torch.randn(16, 64, 16, 16, device=device)
    data2 = torch.randn(16, 256, 64, device=device)
    for _ in range(3):
      loss = train_step(data1, data2)

    # Force synchronization to trigger the queued execution crash
    loss_val = loss.item()
    self.assertIsInstance(loss_val, float)


if __name__ == "__main__":
  absltest.main()
