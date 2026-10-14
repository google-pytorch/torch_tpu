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

"""Verifies that `tt_apis.yaml` matches the discovered TorchTPU public APIs.

This test guards the TorchTPU public API surface: any addition, deletion, or
modification of a public API must be reflected in `tt_apis.yaml`, which
requires approval from the API owners.

It also checks that every API is marked with its stage, that public Python APIs
live under `torch.tpu`, that each public object has one public path, that no API
name marks the API as experimental (stage annotations do that instead), and that
the user-facing list of public APIs, `docs/markdown/public_apis.md`, is up to
date.
"""

import difflib
import os
from typing import Any, Dict, FrozenSet, List, Tuple

from absl.testing import absltest
from tests import repo_paths
from tests import seed_test_utils
from tests.api import scan_tt_apis
import yaml

# The paths below are relative to the TorchTPU repository root.
_OWNERS_PATH = "tests/api/OWNERS"

_API_DESIGN_GUIDE_PATH = "docs/developers/api_design_guide.md"

_PYTHON_STAGE_ANNOTATIONS_PATH = "_internal/utils/annotations.py"

_CPP_STAGE_ANNOTATIONS_PATH = "csrc/common/symbol_stage.h"

# API names must not contain this word in any case; stage annotations mark
# experimental APIs instead.
_EXPERIMENTAL_WORD = "experimental"

_UPDATE_INSTRUCTIONS = f"""\
Any change (addition, deletion, or modification) to TorchTPU public APIs needs
approval from the `tt_apis.yaml` owners listed in `{_OWNERS_PATH}`.

{scan_tt_apis.UPDATE_INSTRUCTIONS}"""

# The public Python API paths outside `scan_tt_apis.PUBLIC_NAMESPACES` that
# predate the namespace rule. Property accessors are listed without their
# " getter" or " setter" suffix.
# TODO(wan): Move these APIs under `torch.tpu` or make them internal, then
# delete this list.
# pylint: disable=line-too-long
_LEGACY_PUBLIC_PATHS: FrozenSet[str] = frozenset({
    # go/keep-sorted start
    # go/keep-sorted end
})
# pylint: enable=line-too-long

# Maps (category, API name) to the API's attributes.
_ApiMap = Dict[Tuple[str, str], Dict[str, Any]]


def _parse_tt_apis_yaml(yaml_text: str) -> _ApiMap:
  """Parses `tt_apis.yaml` text into a map keyed by (category, name).

  Each entry is parsed line by line, since the same name may appear in multiple
  categories (e.g. a flag defined both in Python and C++), which a whole-file
  YAML load would silently collapse.

  Args:
    yaml_text: The `tt_apis.yaml` text.

  Returns:
    A map from (category, API name) to the API's attributes.
  """
  apis: _ApiMap = {}
  for line in yaml_text.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
      continue
    entry = yaml.safe_load(stripped)
    for name, attributes in entry.items():
      apis[(attributes["category"], str(name))] = attributes
  return apis


def _format_api(key: Tuple[str, str], attributes: Dict[str, Any]) -> str:
  category, name = key
  details = ", ".join(
      f"{field}: {value}"
      for field, value in sorted(attributes.items())
      if field != "category"
  )
  return f"{name} [{category}] ({details})"


def _describe_diff(expected: _ApiMap, actual: _ApiMap) -> List[str]:
  """Returns human-readable lines describing differences between API maps."""
  lines: List[str] = []
  added = sorted(actual.keys() - expected.keys())
  removed = sorted(expected.keys() - actual.keys())
  modified = sorted(
      key
      for key in expected.keys() & actual.keys()
      if expected[key] != actual[key]
  )
  if added:
    lines.append(f"Added APIs ({len(added)}):")
    lines.extend(f"  + {_format_api(key, actual[key])}" for key in added)
  if removed:
    lines.append(f"Removed APIs ({len(removed)}):")
    lines.extend(f"  - {_format_api(key, expected[key])}" for key in removed)
  if modified:
    lines.append(f"Modified APIs ({len(modified)}):")
    for key in modified:
      lines.append(f"  ~ {key[1]} [{key[0]}]")
      for field in sorted(expected[key].keys() | actual[key].keys()):
        old_value = expected[key].get(field)
        new_value = actual[key].get(field)
        if old_value != new_value:
          lines.append(f"      {field}: {old_value} -> {new_value}")
  return lines


def _read_from_runfiles(relative_path: str) -> str:
  """Returns the text of a file relative to the repository root in runfiles."""
  with open(
      os.path.join(repo_paths.get_repo_root_in_runfiles(), relative_path),
      encoding="utf-8",
  ) as file:
    return file.read()


# The tests call `scan_tt_apis.discover_all_apis` themselves, not in
# `setUpClass`: discovery probes `torch.accelerator` APIs at runtime, and the
# results can depend on the RNG state that `RepeatableTest.setUp` sets (e.g. for
# `initial_seed`).
class TtApisTest(seed_test_utils.RepeatableTest):

  public_python_apis: List[scan_tt_apis.ApiEntry]

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.public_python_apis, _ = scan_tt_apis.discover_python_apis()

  def test_tt_apis_yaml_matches_discovered_apis(self):
    all_items = scan_tt_apis.discover_all_apis(
        repo_paths.get_repo_root_in_runfiles()
    )
    actual_text = scan_tt_apis.format_tt_apis_yaml(
        all_items, scan_tt_apis.find_duplicated_exports(all_items)
    )
    # Export the up-to-date baseline so it can be downloaded from Sponge.
    scan_tt_apis.export_tt_apis_artifact(actual_text)

    expected_text = _read_from_runfiles(repo_paths.TT_APIS_YAML_PATH)

    if actual_text == expected_text:
      return

    diff_lines = _describe_diff(
        _parse_tt_apis_yaml(expected_text), _parse_tt_apis_yaml(actual_text)
    )
    if not diff_lines:
      diff_lines = [
          "No API entries differ, but the formatting of tt_apis.yaml is out of"
          " date."
      ]
    self.fail(
        "tt_apis.yaml does not match the TorchTPU public APIs discovered by"
        " scan_tt_apis.py.\n\n"
        + "\n".join(diff_lines)
        + "\n\n"
        + _UPDATE_INSTRUCTIONS
    )

  def test_public_apis_md_matches_discovered_apis(self):
    expected_text = _read_from_runfiles(repo_paths.PUBLIC_APIS_MD_PATH)

    actual_text = scan_tt_apis.update_public_apis_md(
        expected_text,
        scan_tt_apis.discover_all_apis(repo_paths.get_repo_root_in_runfiles()),
    )
    # Export the up-to-date file so it can be downloaded from Sponge.
    scan_tt_apis.export_tt_apis_artifact(
        actual_text, scan_tt_apis.PUBLIC_APIS_MD
    )

    if actual_text == expected_text:
      return

    diff_lines = difflib.unified_diff(
        expected_text.splitlines(),
        actual_text.splitlines(),
        fromfile=repo_paths.PUBLIC_APIS_MD_PATH,
        tofile="generated",
        lineterm="",
    )
    self.fail(
        f"{repo_paths.PUBLIC_APIS_MD_PATH} does not match the TorchTPU"
        " public APIs discovered by scan_tt_apis.py.\n\n"
        + "\n".join(diff_lines)
        + "\n\n"
        + _UPDATE_INSTRUCTIONS
    )

  def test_public_python_apis_are_in_public_namespaces(self):
    paths = {
        scan_tt_apis.split_property_accessor(api.name)[0]
        for api in self.public_python_apis
    }
    lines: List[str] = []
    outside = sorted(
        path
        for path in paths - _LEGACY_PUBLIC_PATHS
        if not scan_tt_apis.is_in_public_namespace(path)
    )
    if outside:
      lines.append(
          "Public Python APIs must live under `torch.tpu` (see"
          f" `{_API_DESIGN_GUIDE_PATH}`). Move these APIs there or make them"
          " internal; don't add them to `_LEGACY_PUBLIC_PATHS`:"
      )
      lines.extend(f"  {path}" for path in outside)
    stale = sorted(_LEGACY_PUBLIC_PATHS - paths)
    if stale:
      lines.append(
          "These paths are no longer public; remove them from"
          " `_LEGACY_PUBLIC_PATHS`:"
      )
      lines.extend(f"  {path}" for path in stale)
    if lines:
      self.fail("\n".join(lines))

  def test_public_python_apis_have_one_path_each(self):
    groups = {
        tuple(sorted(group))
        for group in scan_tt_apis.find_duplicated_exports(
            self.public_python_apis
        )
    }
    baseline = _parse_tt_apis_yaml(
        _read_from_runfiles(repo_paths.TT_APIS_YAML_PATH)
    )
    tracked = {
        tuple(sorted([name, *attributes["also_exported_as"]]))
        for (category, name), attributes in baseline.items()
        if category == scan_tt_apis.DUPLICATED_EXPORT_CATEGORY
    }
    lines: List[str] = []
    duplicates = sorted(groups - tracked)
    if duplicates:
      lines.append(
          "Each public TorchTPU object must have one public path (see"
          f" `{_API_DESIGN_GUIDE_PATH}`). Make the other paths internal, or"
          " deprecate the old path of a moved API; don't add them to the"
          " Duplicated Export section of tt_apis.yaml. These paths lead to"
          " the same object:"
      )
      lines.extend(f"  {', '.join(group)}" for group in duplicates)
    stale = sorted(tracked - groups)
    if stale:
      lines.append(
          "These paths no longer lead to the same object; remove them from"
          " the Duplicated Export section of tt_apis.yaml:"
      )
      lines.extend(f"  {', '.join(group)}" for group in stale)
    if lines:
      self.fail("\n".join(lines))

  def test_module_versions_have_stages(self):
    # A `str` can't carry a stage decorator, so the stage of a public module's
    # `__version__` must come from the module's `__tt_api_stages__` map.
    unannotated = scan_tt_apis.find_module_versions_without_stage(
        self.public_python_apis
    )
    if unannotated:
      self.fail(
          "The `__version__` of every public TorchTPU module needs a stage. A"
          " `str` can't carry a stage decorator, so add a `__version__` entry"
          " (an `ApiStageInfo` from `"
          + _PYTHON_STAGE_ANNOTATIONS_PATH
          + "`) to the module's `__tt_api_stages__` map for:\n"
          + "\n".join(f"  {name}" for name in unannotated)
      )

  def test_public_classes_have_no_public_instance_attributes(self):
    # The scanner discovers APIs with `dir`, which doesn't list instance
    # attributes, so a public one would be a public API that tt_apis.yaml
    # doesn't track.
    offending = scan_tt_apis.find_public_instance_attributes(
        self.public_python_apis
    )
    if offending:
      self.fail(
          "Instance attributes of public TorchTPU classes must start with `_`"
          f" (see `{_API_DESIGN_GUIDE_PATH}`). To expose one, add a property,"
          " which tt_apis.yaml tracks. Make these attributes private:\n"
          + "\n".join(f"  {name} ({location})" for name, location in offending)
      )

  def test_api_names_do_not_contain_experimental(self):
    # A name such as `torch.tpu.experimental.foo` or
    # `TORCH_TPU_EXPERIMENTAL_FOO` would force users to change their code when
    # the API's stage changes, so the stage must come from an annotation.
    offending = sorted(
        f"{api.name} [{api.category}]"
        for api in scan_tt_apis.discover_all_apis(
            repo_paths.get_repo_root_in_runfiles()
        )
        if _EXPERIMENTAL_WORD in api.name.lower()
    )
    if offending:
      self.fail(
          "API names (including path segments) must not contain"
          f" {_EXPERIMENTAL_WORD!r} (case insensitive). Mark experimental APIs"
          " with a stage annotation instead"
          f" (`{_PYTHON_STAGE_ANNOTATIONS_PATH}` for Python,"
          f" `{_CPP_STAGE_ANNOTATIONS_PATH}` for C++), so that promoting an API"
          " to a later stage doesn't force users to change their code. Rename"
          " these APIs:\n"
          + "\n".join(f"  {api}" for api in offending)
      )

  def test_apis_have_stages(self):
    # An API without a stage annotation shows up as "Unknown" in tt_apis.yaml
    # and isn't listed in public_apis.md, so users can't tell whether they may
    # rely on it.
    unknown = sorted(
        f"{api.name} [{api.category}]"
        for api in scan_tt_apis.discover_all_apis(
            repo_paths.get_repo_root_in_runfiles()
        )
        if api.stage is None
    )
    if unknown:
      self.fail(
          "Every TorchTPU API must be marked with its stage. Annotate these"
          " APIs with their stages"
          f" (`{_PYTHON_STAGE_ANNOTATIONS_PATH}` for Python,"
          f" `{_CPP_STAGE_ANNOTATIONS_PATH}` for C++), then update"
          " tt_apis.yaml:\n"
          + "\n".join(f"  {api}" for api in unknown)
          + "\n\n"
          + _UPDATE_INSTRUCTIONS
      )


if __name__ == "__main__":
  absltest.main()
