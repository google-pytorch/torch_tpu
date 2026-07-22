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

// Minimal C extension exercised by _versioned_so_loader_integration_test.py.
// It exports only the unversioned PyInit_dispatch_probe symbol; the test lays
// copies of the artifact out under version-suffixed filenames, exactly the
// shape the multi-ABI wheel bundles its glues in.

#include <Python.h>

static struct PyModuleDef dispatch_probe_module = {
    PyModuleDef_HEAD_INIT,
    "dispatch_probe",
    "Minimal extension used to test the version-dispatch import hook.",
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit_dispatch_probe(void) {
  return PyModule_Create(&dispatch_probe_module);
}
