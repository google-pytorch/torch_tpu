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

XLA_COMMIT = "9af938ab71d667f4e5108f7c07bae60ba671de44"
XLA_SHA256 = "c5d4aa63f932866bfe14357120cc59ec83094bed915745cefd5681ce777128bf"
