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

from absl import app
from absl import flags
import torch
import torch_tpu  # pylint: disable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "allow_excess_precision",
    None,
    "Whether to allow excess precision on TPU compiled operations.",
)


def main(_):
  # Python uses float64 by default.
  bit_6 = 1 + 2**-6
  bit_7 = 1 + 2**-7
  bits_6_7 = 1.0 + 2**-6 + 2**-7
  bits_6_7_13 = 1.0 + 2**-6 + 2**-7 + 2**-13

  print("=== Python CPU values ===")
  print(f"{bit_6=:.20f}")
  print(f"{bit_7=:.20f}")
  print(f"{bits_6_7=:.20f}")
  print(f"{bits_6_7_13=:.20f}")
  print(f"{(bit_6 * bit_7)=:.20f}")
  assert bit_6 * bit_7 == bits_6_7_13

  # Configure allow_excess_precision at the very start of the PyTorch TPU runtime.
  # Note: Since it can only be set ONCE per process, trying to set it twice will raise an error.
  # pytype: disable=module-attr
  if FLAGS.allow_excess_precision is not None:
    torch.backends.tpu.allow_excess_precision = FLAGS.allow_excess_precision
    print(
        "Configured torch.backends.tpu.allow_excess_precision ="
        f" {FLAGS.allow_excess_precision}"
    )
  else:
    print(
        "Leaving torch.backends.tpu.allow_excess_precision unset (using"
        " default)."
    )

  allow_excess = torch.backends.tpu.allow_excess_precision
  print(f"Active allow_excess_precision state: {allow_excess}")
  # pytype: enable=module-attr

  print("=== PyTorch TPU values ===")

  eye = torch.eye(256, dtype=torch.bfloat16, device="tpu")
  bit_6_tensor = eye.clone() * bit_6
  bit_7_tensor = eye.clone() * bit_7
  bits_6_7_tensor = eye.float().clone() * bits_6_7
  bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

  # This must be a vector of length one rather than a
  # scalar due to PyTorch's unique dtype promotion rules.
  # TLDR: scalar dtypes are often ignored.
  one_float32 = torch.tensor([1.0], dtype=torch.float32, device="tpu")

  print(f"{bit_6_tensor[0,0]=:.20f}")
  print(f"{bit_7_tensor[0,0]=:.20f}")
  print(f"{bits_6_7_tensor[0,0]=:.20f}")
  print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

  def mul_add(
      a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
  ) -> torch.Tensor:
    return a @ b * c

  # The 13th bit will not be flipped in eager execution regardless of the flag.
  print("=== TPU values: Eager ===")
  print(f"{mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0,0]=:.20f}")
  assert (
      mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
      == bits_6_7_tensor[0, 0]
  )

  # Compiled execution: behavior is determined by the allow_excess_precision setting.
  print(
      f"=== TPU values: Compiled (allow_excess_precision = {allow_excess}) ==="
  )
  mul_add_compiled = torch.compile(mul_add, backend="tpu")
  val = (
      mul_add_compiled(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
      .cpu()
      .item()
  )
  print(f"Compiled output value: {val:.20f}")

  if allow_excess:
    # If excess precision is allowed, the intermediate bfloat16 matmul can retain high precision,
    # flipping the 13th bit of the result.
    print("Asserting that excess precision occurred (13th bit flipped).")
    assert (
        mul_add_compiled(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
        == bits_6_7_13_tensor[0, 0]
    )
  else:
    # If excess precision is disallowed, math matches bfloat16 limits (13th bit not flipped).
    print("Asserting that excess precision was prevented.")
    assert (
        mul_add_compiled(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
        == bits_6_7_tensor[0, 0]
    )

  return 0


if __name__ == "__main__":
  app.run(main)
