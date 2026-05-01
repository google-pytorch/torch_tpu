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

"""Tests for ldexp operation on xla_cpu."""

from absl.testing import absltest
from absl.testing import parameterized
import torch

class LdexpCpuTest(absltest.TestCase):

  def test_ldexp_parity(self):
    def compute(device, mantissa, exponent):
      t_mant = torch.tensor([mantissa], dtype=torch.float32, device=device)
      t_exp = torch.tensor([exponent], dtype=torch.int32, device=device)
      return torch.ldexp(t_mant, t_exp)

    mantissa = 1.0
    exponent = 127
    
    # Run on CPU
    res_cpu = compute(torch.device("cpu"), mantissa, exponent)
    
    # Run on XLA CPU
    res_xla = compute(torch.device("xla_cpu"), mantissa, exponent)
    
    # Compare
    torch.testing.assert_close(res_xla.cpu(), res_cpu)

if __name__ == "__main__":
  absltest.main()
