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

"""Process-wide compatibility shims needed to load and run models.

Scope: only patches without which models cannot be imported, constructed, or
run through an eager CPU / meta forward. Each shim must be guarded (by
`hasattr` or a version check) so it is a no-op once upstream is fixed, and
must cite the CL or bug that introduced it.

Belongs here:
- Environment stubs, e.g. a `torchaudio` stub when the package is absent.
- Fake meta kernels for ops that lack one (`aten::bincount`, ...).
- Library-wide API restorations, e.g. `EncoderDecoderCache.__getitem__`
  removed in Transformers v5 (cl/992488464).

Does not belong here:
- Fixes for a single model family: use `ModelQuirk.model_patches`.
- Patches needed only for compile, decode, StaticCache or TPU execution:
  those live in hf_coverage_test.

Migrated from module_registry.py in a later phase.
"""
