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

"""Pytest configuration file to isolate CWD and test installed wheels."""

import importlib.machinery
import os
import sys

# Support disabling all conftest workarounds (e.g. for src/ layout migration)
# TODO(ecalubaquib): Delete this workaround after src/ layout migration.
DISABLE_CONFTEST = (
    os.environ.get("TORCH_TPU_USE_SRC_LAYOUT") == "true"
    or os.environ.get("TORCH_TPU_DISABLE_CONFTEST") == "true"
)

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
  sys.path.insert(0, repo_root)

if not DISABLE_CONFTEST:

  class WheelOnlyMetaPathFinder:
    """Custom meta_path finder to force importing torch_tpu from site-packages.

    This ignores the local unbuilt source torch_tpu directory in the
    repository root.
    """

    def __init__(self, blocked_dir, target_package="torch_tpu"):
      self.blocked_dir = os.path.abspath(blocked_dir)
      self.target_package = target_package

    def find_spec(self, fullname, path, target=None):
      """Finds the module specification along the filtered path."""
      if fullname == self.target_package or fullname.startswith(
          self.target_package + "."
      ):
        search_path = path if path is not None else sys.path
        # This filters/intercepts anything related to torch_tpu directory.
        filtered_path = [
            p
            for p in search_path
            if os.path.abspath(p) != self.blocked_dir and p
        ]
        return importlib.machinery.PathFinder.find_spec(
            fullname, filtered_path, target
        )
      return None

  blocked_repo_root = os.path.dirname(os.path.abspath(__file__))
  # Look for torch_tpu as an installed package when running pytest.
  sys.meta_path.insert(
      0, WheelOnlyMetaPathFinder(blocked_repo_root, "torch_tpu")
  )


def pytest_collection_modifyitems(items):
  """Hook to initialize required ABSL flags after modules are imported."""
  del items
  if DISABLE_CONFTEST:
    return

  try:
    from absl import flags  # pylint: disable=g-import-not-at-top

    if os.environ.get("TORCH_TPU_DISABLE_TEST_MODE_OVERRIDE") != "true":
      if "test_mode" in flags.FLAGS:
        flag = flags.FLAGS["test_mode"]
        # Determine the best value to set for
        # 'test_mode' dynamically based on flag type
        if "error_testing" in getattr(flag, "module_name", ""):
          # error_testing.py strictly expects
          # 'tpu', 'gpu', or 'cov' (and is required=True)
          flags.FLAGS.test_mode = "tpu"
        elif hasattr(flag, "_enum_class") or hasattr(
            getattr(flag, "serializer", None), "_enum_class"
        ):
          enum_class = getattr(flag, "_enum_class", None) or getattr(
              flag.serializer, "_enum_class", None
          )
          if enum_class:
            matched_member = None
            for member in enum_class:
              if member.value in (
                  "torch_tpu_vs_cpu",
                  "cpu",
              ) or member.name.lower() in ("torch_tpu_vs_cpu", "cpu"):
                matched_member = member
                if (
                    member.value == "torch_tpu_vs_cpu"
                    or member.name.lower() == "torch_tpu_vs_cpu"
                ):
                  break
            if matched_member is not None:
              flags.FLAGS.test_mode = matched_member
        elif hasattr(flag, "choices") and flag.choices:
          if "torch_tpu_vs_cpu" in flag.choices:
            flags.FLAGS.test_mode = "torch_tpu_vs_cpu"
          elif "cpu" in flag.choices:
            flags.FLAGS.test_mode = "cpu"
        else:
          # Fallback default
          flags.FLAGS.test_mode = "torch_tpu_vs_cpu"
    flags.FLAGS.mark_as_parsed()
  except Exception:  # pylint: disable=broad-except
    pass
