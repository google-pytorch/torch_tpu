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

"""Data structures and utilities for parsing and serializing goldens.pin manifests."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Final, TypeAlias

import attrs
import cattrs

PathType: TypeAlias = pathlib.Path | str


class InvalidSchemaError(ValueError):
  """Raised when a goldens.pin manifest does not conform to the expected schema."""


def _ensure_path_type(path: PathType) -> pathlib.Path:
  """Converts a PathType (str or pathlib.Path) to a pathlib.Path."""
  if isinstance(path, str):
    return pathlib.Path(path)
  return path


def _validate_non_empty_str(
    instance: Any, attribute: attrs.Attribute[str], value: str
) -> None:
  del instance
  if not value or not isinstance(value, str):
    raise ValueError(f"'{attribute.name}' must not be empty.")


def _validate_version(
    instance: Any, attribute: attrs.Attribute[int], value: int
) -> None:
  del instance, attribute
  if value != 1:
    raise InvalidSchemaError(
        f"Unsupported goldens.pin version: {value}, expected 1"
    )


@attrs.frozen
class GoldenPath:
  """Metadata for an individual golden file entry in GCS."""

  sha256: str = attrs.field(
      validator=[attrs.validators.instance_of(str), _validate_non_empty_str]
  )
  gcs_path: str = attrs.field(
      validator=[attrs.validators.instance_of(str), _validate_non_empty_str]
  )


@attrs.frozen
class GoldenPathPinFile:
  """Typed structure for the goldens.pin manifest file."""

  version: int = attrs.field(
      validator=[attrs.validators.instance_of(int), _validate_version]
  )
  gcs_bucket: str = attrs.field(
      validator=[attrs.validators.instance_of(str), _validate_non_empty_str]
  )
  goldens: dict[str, GoldenPath] = attrs.field(
      validator=attrs.validators.instance_of(dict)
  )


CONVERTER: Final[cattrs.Converter] = cattrs.Converter()


def parse_goldens_pin(
    source: PathType,
) -> GoldenPathPinFile:
  """Parses and validates the goldens.pin JSON file.

  Args:
    source: Path to the goldens.pin JSON file.

  Returns:
    The parsed GoldenPathPinFile instance.

  Raises:
    FileNotFoundError: If the source file does not exist.
    InvalidSchemaError: If the goldens.pin content is invalid or does not match
      the expected schema.
  """
  path = _ensure_path_type(source)
  if not path.is_file():
    raise FileNotFoundError(f"goldens.pin not found at {path}")
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

  if not isinstance(data, dict):
    raise InvalidSchemaError(
        f"Invalid goldens.pin content: expected dict, got {type(data).__name__}"
    )

  version = data.get("version")
  if version != 1:
    raise InvalidSchemaError(
        f"Unsupported goldens.pin version: {version}, expected 1"
    )

  if "gcs_bucket" not in data or not isinstance(data["gcs_bucket"], str):
    raise InvalidSchemaError(
        "Invalid goldens.pin: missing or invalid 'gcs_bucket'"
    )

  if "goldens" not in data or not isinstance(data["goldens"], dict):
    raise InvalidSchemaError(
        "Invalid goldens.pin: missing or invalid 'goldens'"
    )

  try:
    return CONVERTER.structure(data, GoldenPathPinFile)
  except cattrs.errors.ClassValidationError as e:
    raise InvalidSchemaError(f"Invalid goldens.pin: {e}") from e


def serialize_goldens_pin(pin_file: GoldenPathPinFile) -> str:
  """Serializes a GoldenPathPinFile to a formatted JSON string."""
  data = CONVERTER.unstructure(pin_file)
  return json.dumps(data, indent=2, sort_keys=True) + "\n"


def dump_goldens_pin(
    pin_file: GoldenPathPinFile,
    target: PathType,
) -> None:
  """Writes a GoldenPathPinFile to the target path."""
  target_path = _ensure_path_type(target)
  target_path.parent.mkdir(parents=True, exist_ok=True)
  target_path.write_text(serialize_goldens_pin(pin_file), encoding="utf-8")
