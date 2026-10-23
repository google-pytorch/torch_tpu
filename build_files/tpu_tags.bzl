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

"""Parsing for `requires-tpu-<version>[:<chips>]` test tags."""

visibility(["//..."])

# Maps a `requires-tpu-<version>` tag to its TPU generation.
_TPU_VERSION_TO_GENERATION = {
    "requires-tpu-v5lite": "v5",
    "requires-tpu-v5p": "v5",
    "requires-tpu-v6e": "v6",
    "requires-tpu-v7x": "v7",
}

def get_tpu_generation(tag):
    """Returns the TPU generation for a `requires-tpu-<version>[:<chips>]` tag."""
    return _TPU_VERSION_TO_GENERATION.get(tag.split(":")[0])

def get_tpu_chip_count(tag):
    """Returns the chip count from a `requires-tpu-<version>:<chips>` tag.

    Args:
        tag: A test tag.

    Returns:
        The chip count, or None if `tag` isn't a TPU tag with a chip count.
    """
    if get_tpu_generation(tag) == None or ":" not in tag:
        return None
    _, chips_str = tag.split(":", 1)
    return int(chips_str) if chips_str.isdigit() else None

def required_tpu_chip_count(tags):
    """Returns the largest chip count in `tags`, or 1 if no tag has one."""
    return max([1] + [get_tpu_chip_count(tag) or 1 for tag in tags])
