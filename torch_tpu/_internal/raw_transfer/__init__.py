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

"""Raw transfer utilities for torch_tpu."""

from torch_tpu._internal.raw_transfer import raw_transfer_impl

PjRtCopyFuture = raw_transfer_impl.PjRtCopyFuture
transfer_d2h_batch_async = raw_transfer_impl.transfer_d2h_batch_async
transfer_h2d_batch_async = raw_transfer_impl.transfer_h2d_batch_async
transfer_d2h_batch = raw_transfer_impl.transfer_d2h_batch
transfer_h2d_batch = raw_transfer_impl.transfer_h2d_batch
await_all = raw_transfer_impl.await_all

__all__ = [
    "PjRtCopyFuture",
    "transfer_d2h_batch_async",
    "transfer_h2d_batch_async",
    "transfer_d2h_batch",
    "transfer_h2d_batch",
    "await_all",
]
