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

"""torch_tpu._internal.profiler.profile API."""

# pylint: disable=g-importing-member
# (This rule conflicts with the purpose of an __init__.py file, which is to
# make symbols available at the package level.)
from torch_tpu._internal.profiler.profiler_api import profile
from torch_tpu._internal.profiler.profiler_api import ProfilerActivity
from torch_tpu._internal.profiler.profiler_api import register_kineto_backend
from torch_tpu._internal.profiler.profiler_api import xprof_trace_handler
from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig

__all__ = [
    # go/keep-sorted start
    'ProfilerActivity',
    'TpuProfilerConfig',
    'profile',
    'register_kineto_backend',
    'xprof_trace_handler',
    # go/keep-sorted end
]
