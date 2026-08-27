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

"""Detects whether the installed libtpu supports the native scan emitter.

Cumulative ops (cumsum/cumprod/cummax/cummin/logcumsumexp) lower to chlo.ScanOp,
which the TPU compiler's native scan emitter compiles directly. That emitter
needs a libtpu new enough to compile it. OSS pins a libtpu wheel (e.g. 0.0.40,
used by torchtpu-vllm) that can predate it, so torch_tpu must fall back to the
StableHLO while-loop lowering there. See b/529376045.
"""

# The first libtpu version that can compile the native scan emitter. libtpu
# 0.0.40 (and older) cannot, so the scan lowering is gated to > 0.0.40. Bump
# this if the required libtpu changes.
MIN_LIBTPU_VERSION_WITH_NATIVE_SCAN = (0, 0, 44)


def _parse_version(version: str) -> tuple[int, ...]:
  """Parses the leading numeric components of a version string into a tuple.

  Stops at the first non-numeric component so pre-release / nightly suffixes
  (e.g. "0.0.41.dev20260101") compare by their numeric prefix.

  Args:
    version: A version string such as "0.0.41".

  Returns:
    The leading numeric components, e.g. (0, 0, 41).
  """
  components = []
  for part in version.split("."):
    digits = ""
    for ch in part:
      if not ch.isdigit():
        break
      digits += ch
    if not digits:
      break
    components.append(int(digits))
  return tuple(components)


def libtpu_supports_native_scan() -> bool:
  """Returns True if the installed libtpu can compile the native scan emitter."""
  try:
    import libtpu  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
  except ImportError:
    return False
  version = getattr(libtpu, "__version__", None)
  if not version:
    return False
  return _parse_version(version) >= MIN_LIBTPU_VERSION_WITH_NATIVE_SCAN
