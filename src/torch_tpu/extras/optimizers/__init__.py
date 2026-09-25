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

"""TorchTPU extra optimizers package."""

from torch_tpu.extras.optimizers import adamw
from torch_tpu.extras.optimizers import muon
from torch_tpu.extras.optimizers import optim
from torch_tpu.extras.optimizers import sgd
from torch_tpu.extras.optimizers import single_trace_trainer

AdamW = adamw.AdamW
Muon = muon.Muon
Optimizer = optim.Optimizer
ReferenceAdamw = adamw.ReferenceAdamw
ReferenceMuon = muon.ReferenceMuon
ReferenceSgd = sgd.ReferenceSgd
SGD = sgd.SGD
SingleTraceTrainer = single_trace_trainer.SingleTraceTrainer
TorchAdamw = adamw.TorchAdamw
TorchMuon = muon.TorchMuon
TorchSgd = sgd.TorchSgd

__all__ = [
    "AdamW",
    "Muon",
    "Optimizer",
    "ReferenceAdamw",
    "ReferenceMuon",
    "ReferenceSgd",
    "SGD",
    "SingleTraceTrainer",
    "TorchAdamw",
    "TorchMuon",
    "TorchSgd",
    "adamw",
    "muon",
    "optim",
    "sgd",
    "single_trace_trainer",
]
