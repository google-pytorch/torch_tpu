# Copyright 2025 Google LLC
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

"""Generate scaled dot product attention kernel."""

from collections.abc import Sequence

from absl import app
from absl import flags
import jax.numpy as jnp
import numpy as np
from torch_tpu._internal.pallas import pallas_kernel_generate_utils as kernel_utils
from torch_tpu.ops.scaled_dot_product_attention.kernels import scaled_dot_product_attention_kernels as kernels

_FORWARD = flags.DEFINE_boolean(
    "forward",
    False,
    "Generate forward kernel.",
)

_BACKWARD = flags.DEFINE_boolean(
    "backward",
    False,
    "Generate backward kernel.",
)

_HEADER = flags.DEFINE_string(
    "header",
    None,
    "Header file to include in the generated kernel string.",
)

_IMPLEMENTATION = flags.DEFINE_string(
    "implementation",
    None,
    "Implementation file to include in the generated kernel string.",
)

_KERNEL_NAME = flags.DEFINE_string(
    "kernel_name",
    None,
    "Name of the kernel to generate. This will be used as the prefix for the"
    " generated C++ variable.",
)

_KERNEL_TYPE = flags.DEFINE_string(
    "kernel_type",
    "flash",
    "Kernel type to generate.",
)

_DTYPE = flags.DEFINE_string(
    "dtype",
    "float32",
    "Dtype of the inputs/outputs.",
)

_IS_CAUSAL = flags.DEFINE_boolean(
    "is_causal",
    False,
    "Whether the kernel is causal.",
)


def get_kernel():
  if _KERNEL_TYPE.value == "flash":
    return kernels.SDPAKernelFlashAttention
  elif _KERNEL_TYPE.value == "jax":
    return kernels.SDPAKernelReferenceJax
  else:
    raise ValueError(f"Unsupported kernel type: {_KERNEL_TYPE.value}")


def get_dtype():
  if _DTYPE.value == "float32":
    return np.float32
  elif _DTYPE.value == "bfloat16":
    return jnp.bfloat16
  else:
    raise ValueError(f"Unsupported dtype: {_DTYPE.value}")


#######################################################################
def export_forward_kernel(header_path, implementation_path):
  """Export and print a dynamic forward kernel."""
  kernel = get_kernel()

  exported = kernel.export_forward(
      dtype=get_dtype(),
      is_causal=_IS_CAUSAL.value,
      input_sizes=None,
  )
  if header_path is not None and implementation_path is not None:
    kernel_utils.generate_embedded_file(
        header_path,
        implementation_path,
        [(  # pyrefly: ignore[bad-argument-type]
            _KERNEL_NAME.value,
            # TODO(elliotenglish): change this to use mlir bytecode
            exported.mlir_module().encode(),
        )],
    )


def export_backward_kernel(header_path, implementation_path):
  """Export a dynamic backward kernel."""
  kernel = get_kernel()

  exported = kernel.export_backward(
      dtype=get_dtype(),
      is_causal=_IS_CAUSAL.value,
      input_sizes=None,
  )

  if header_path is not None and implementation_path is not None:
    kernel_utils.generate_embedded_file(
        header_path,
        implementation_path,
        [(  # pyrefly: ignore[bad-argument-type]
            _KERNEL_NAME.value,
            # TODO(elliotenglish): change this to use mlir bytecode
            exported.mlir_module().encode(),
        )],
    )


#######################################################################
def main(argv: Sequence[str]) -> None:
  del argv

  assert not (_FORWARD.value and _BACKWARD.value)
  if _FORWARD.value:
    export_forward_kernel(_HEADER.value, _IMPLEMENTATION.value)
  if _BACKWARD.value:
    export_backward_kernel(_HEADER.value, _IMPLEMENTATION.value)


if __name__ == "__main__":
  app.run(main)
