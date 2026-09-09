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
"""Utility functions for materializing and synchronizing TPU tensors.

In eager mode, tensors in torch_tpu are sometimes deferred, meaning that they
have a pending operation that needs to be compiled and executed before their
data is available.

Accessing the data of a deferred tensor (through .cpu(), .item(), or similar)
always forces the following steps:
  1. Graph traversal: the deferred operations graph is analyzed and prepared
  2. Compilation: the graph is compiled into executable code
  3. Materialization: the execution is scheduled, and a PjRtBuffer is created
     to hold the future result
  4. Synchronization: waiting for the running executable to complete
  5. Transfer: copying data from device to host

However, in some cases it can be useful to only materialize and/or synchronize.
  * Manual materialization splits the deferred op graph, resulting in multiple
    smaller compilation steps. This can reduce compilation time, but may result
    in a less efficient executable.
  * Manual materialization can be used to insert breakpoints in loops, reusing
    a cached compilation across each iteration.
  * Manual materialization can be used to profile compilation time, while
    avoiding the overhead of data transfer and execution.
  * Manual synchronization can be used to profile the combined compilation and
    execution time; or, if the compilation is cached, the execution time alone.

These are not intended as the primary interface for users. These are currently
only used internally for performance evaluation by the torch_tpu developers.
"""

from torch_tpu._internal.sync import sync

computation_graphviz = sync.computation_graphviz
computation_mlir = sync.computation_mlir
dump_computation_graphviz = sync.dump_computation_graphviz
synchronize = sync.synchronize
is_materializing = sync.is_materializing
is_materialized = sync.is_materialized

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "computation_graphviz",
    "computation_mlir",
    "dump_computation_graphviz",
    "is_materialized",
    "is_materializing",
    "synchronize",
    # go/keep-sorted end
]
