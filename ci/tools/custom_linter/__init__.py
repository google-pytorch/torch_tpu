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

"""Custom presubmit convention checker package for TorchTPU."""

from ci.tools.custom_linter import engine
from ci.tools.custom_linter import metadata
from ci.tools.custom_linter import vcs

main = engine.main
run_checks = engine.run_checks
collect_violations = engine.collect_violations
CheckResult = engine.CheckResult
Violation = engine.Violation
Rule = metadata.Rule
RuleSet = metadata.RuleSet
PathMapper = metadata.PathMapper
LoadError = metadata.LoadError
Vcs = vcs.Vcs
VcsError = vcs.VcsError
