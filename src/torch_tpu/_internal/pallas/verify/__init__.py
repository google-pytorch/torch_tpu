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

"""Pallas kernel and PyTorch program verification for TorchTPU."""

# pylint: disable=g-importing-member
from torch_tpu._internal.pallas.verify.findings import FindingCategory
from torch_tpu._internal.pallas.verify.findings import Phase
from torch_tpu._internal.pallas.verify.findings import Severity
from torch_tpu._internal.pallas.verify.findings import VerificationError
from torch_tpu._internal.pallas.verify.findings import VerificationFinding
from torch_tpu._internal.pallas.verify.findings import VerificationReport
from torch_tpu._internal.pallas.verify.verifier import VerificationMode
from torch_tpu._internal.pallas.verify.verifier import VerificationOptions
from torch_tpu._internal.pallas.verify.verifier import verify

# PEP 8 requires this to be a list of strings, not a tuple or a list of objects.
__all__ = [
    # go/keep-sorted start
    "FindingCategory",
    "Phase",
    "Severity",
    "VerificationError",
    "VerificationFinding",
    "VerificationMode",
    "VerificationOptions",
    "VerificationReport",
    "verify",
    # go/keep-sorted end
]
