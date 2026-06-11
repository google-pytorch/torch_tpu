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

"""Holds the XLA commit and sha for easy automated updating."""

# To update XLA to a new revision,
# a) update XLA_COMMIT to the new git commit hash. Using JAX's XLA commit
#    is a good reference point.
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/{git_hash}.tar.gz | sha256sum
#    and update XLA_SHA256 with the result.

XLA_COMMIT = "ac1619fd265da44966161bbbf13ad2e8e47bac5b"
XLA_SHA256 = "906916ca47a6daa8c059b8cf829d8bcfc7f30d3adfd34c4044a7b245710326a6"
