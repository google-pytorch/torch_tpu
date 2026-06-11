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

"""Reproduction for b/515063036."""

from absl.testing import absltest
import torch
import torch_tpu


class ReproTest(absltest.TestCase):

  def test_repro(self):
    @torch.compile(backend="tpu")
    def f(x, w):
      return torch.mm(x, w, out_dtype=torch.float32)

    x = torch.randn(8, 2048, device="tpu").to(torch.float8_e4m3fn)
    w = torch.randn(2048, 768, device="tpu").to(torch.float8_e4m3fn)

    print("Running compiled function...")
    y = f(x, w)
    print("Success! Result shape:", y.shape)


if __name__ == "__main__":
  absltest.main()
