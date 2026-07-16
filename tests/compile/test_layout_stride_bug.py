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
"""Reproduction of layout and stride mismatch when returning from compile.

Failing baseline (pure p4head without mhlo.layout_mode fix):
  http://sponge2/20be49d5-f75c-49fd-8fea-7fdce74abcf7
Passing verification (with mhlo.layout_mode = "default" fix):
  http://sponge2/1d3ffcbf-ef88-4cd6-9d72-65d325d768cf
"""

from absl.testing import absltest
from third_party.py.torch_tpu._internal.utils.utils import assert_close
import torch


class LayoutStrideBugReproTest(absltest.TestCase):
  """Tests layout and stride behavior for non-row-major return buffers."""

  def test_repro_transposed_return(self):
    """Verifies that transposed returns from compile do not scramble data."""

    @torch.compile(backend="tpu")
    def func(x):
      return x.t() * 2.0

    x = torch.arange(16, dtype=torch.float32).view(4, 4)
    x_tpu = x.to("tpu")
    y_tpu = func(x_tpu)

    expected_y = x.t() * 2.0
    assert_close(y_tpu.cpu(), expected_y)


if __name__ == "__main__":
  absltest.main()
