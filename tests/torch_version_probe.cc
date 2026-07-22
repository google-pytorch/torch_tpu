// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Compile-time probe asserting that the PyTorch headers a per-version build
// resolves really belong to the version it was configured for. Instantiated
// once per WHEEL_TORCH_VERSIONS entry through the //:_torch_version
// transition (see //tests:torch_version_probe_test), with the expected triple
// injected via local_defines. A wrong requirements lock, pip hub, or select()
// row fails this translation unit the moment the version is added, rather
// than surfacing later as a mysterious ABI mismatch.

#include "torch/headeronly/version.h"

static_assert(TORCH_VERSION_MAJOR == EXPECTED_TORCH_MAJOR,
              "resolved torch headers have the wrong major version");
static_assert(TORCH_VERSION_MINOR == EXPECTED_TORCH_MINOR,
              "resolved torch headers have the wrong minor version");
static_assert(TORCH_VERSION_PATCH == EXPECTED_TORCH_PATCH,
              "resolved torch headers have the wrong patch version");
