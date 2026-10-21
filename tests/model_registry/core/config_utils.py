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

"""Config access helpers that work for raw dicts and HF config objects."""

from collections.abc import Mapping, Sequence
from typing import Any

# Nested configs searched by `get_first_field` after the top level.
_SUB_CONFIGS = ("text_config", "vision_config", "audio_config")


def get_field(config: Any, key: str, default: Any = None) -> Any:
  """Reads `key` from a dict-like or attribute-style config.

  Args:
    config: A raw `config.json` dict or a `transformers.PretrainedConfig`.
    key: Field name to read.
    default: Value returned when the field is missing or None.

  Returns:
    The field value, or `default` if absent.
  """
  if config is None:
    return default
  if isinstance(config, Mapping):
    value = config.get(key, default)
  else:
    value = getattr(config, key, default)
  return default if value is None else value


def get_first_field(
    config: Any, keys: Sequence[str], default: Any = None
) -> Any:
  """Returns the first present field among `keys`.

  The top-level config is searched first, then the nested text / vision /
  audio configs of composite models.

  Args:
    config: A raw `config.json` dict or a `transformers.PretrainedConfig`.
    keys: Candidate field names, in priority order.
    default: Value returned when no candidate is present.

  Returns:
    The first present value, or `default`.
  """
  for cfg in (config, *(get_field(config, s) for s in _SUB_CONFIGS)):
    for key in keys:
      value = get_field(cfg, key)
      if value is not None:
        return value
  return default


def as_int(value: Any, min_value: int | None = None) -> int | None:
  """Converts a config value to `int`, or returns None.

  Matches `_safe_int` in tests/transformers_inputs.py: booleans and
  non-numeric types are rejected, and values below `min_value` are treated as
  missing.

  Args:
    value: Raw config value.
    min_value: Optional inclusive lower bound.

  Returns:
    The converted integer, or None if `value` is missing, invalid, or below
    `min_value`.
  """
  if value is None or isinstance(value, bool):
    return None
  if not isinstance(value, (int, float, str)):
    return None
  try:
    result = int(value)
  except (TypeError, ValueError):
    return None
  if min_value is not None and result < min_value:
    return None
  return result


def first_int(*values: Any) -> int | None:
  """Returns the first of `values` that converts to a positive `int`.

  The result is never 0, so callers can supply a default with `or`.

  Args:
    *values: Raw config values, in priority order.

  Returns:
    The first positive integer, or None.
  """
  for value in values:
    if (result := as_int(value, min_value=1)) is not None:
      return result
  return None
