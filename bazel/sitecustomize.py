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

"""sitecustomize.py to automatically install wheel test diagnostics."""

import sys

try:
  # pylint: disable=g-import-not-at-top
  import wheel_test_diagnostics

  wheel_test_diagnostics.install_hook()
  print("Wheel test diagnostics hook installed.", file=sys.stderr)
except ImportError:
  # Diagnostics hook not available in this environment, skip silently.
  pass
except Exception as e:  # pylint: disable=broad-exception-caught
  print(f"Failed to install wheel test diagnostics hook: {e}", file=sys.stderr)
