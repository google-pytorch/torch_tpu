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

"""Batch transfer utilities for torch_tpu."""

from torch_tpu._internal.batch_transfer import batch_transfer_impl

TransferFuture = batch_transfer_impl.TransferFuture
batch_transfer_d2h = batch_transfer_impl.batch_transfer_d2h
batch_transfer_h2d = batch_transfer_impl.batch_transfer_h2d
batch_transfer_d2h_sync = batch_transfer_impl.batch_transfer_d2h_sync
batch_transfer_h2d_sync = batch_transfer_impl.batch_transfer_h2d_sync
await_all = batch_transfer_impl.await_all

__all__ = [
    "TransferFuture",
    "batch_transfer_d2h",
    "batch_transfer_h2d",
    "batch_transfer_d2h_sync",
    "batch_transfer_h2d_sync",
    "await_all",
]
