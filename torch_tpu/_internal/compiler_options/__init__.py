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

import contextlib

from torch_tpu._internal import compiler_options_impl as compiler


@contextlib.contextmanager
def custom_compiler_options(options: dict[str, str]):
  """Context manager for setting the XLA compiler options.

  Args:
    options: A dictionary of XLA compiler options. The key is the option name,
      and the value is the option value.  In addition, we also support these
      special options: "xla_optimization_level": the XLA optimization level.
      Valid options are 'O0', 'O1', 'O2', 'O3'. "xla_memory_fitting_level": the
      XLA memory fitting level. Valid options are 'O0', 'O1', 'O2', 'O3'.
      "xla_allow_excess_precision": whether to allow excess precision. Valid
      options are "true", "false". See more details of the flags in
      https://github.com/openxla/xla/blob/main/xla/xla.proto for more details.

  For example:
    with custom_compiler_options({
        "xla_optimization_level": "O1",
        "xla_tpu_enable_deduplicated_calls": "ENABLED",
    }):
      result =torch.matmul(torch.ones((1, 1)), torch.ones((1, 1)))
      result.to("cpu")

  All operations *materialized* within the context will use the provided
  compiler options.
  """

  # See
  # https://openxla.org/xla/flags_guidance
  # for available options.
  try:
    compiler.push_compiler_options(options)
    yield
  finally:
    compiler.pop_compiler_options()


# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "custom_compiler_options",
    # go/keep-sorted end
]
