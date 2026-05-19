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

"""Helper functions for loading data for quality benchmarks."""

import enum
from typing import Any, Iterable

from absl import logging
import grain.python as grain
import tensorflow_datasets as tfds
from torch_tpu._internal.utils import log_utils


log_utils.log_to_stderr()


def _get_num_prompts(num_prompts: int, dataset_size: int) -> int:
  """Returns the number of prompts to load from the dataset."""
  logging.info(
      "Full dataset size: %d, num_prompts: %d", dataset_size, num_prompts
  )

  if num_prompts <= -1 or num_prompts > dataset_size:
    return dataset_size
  return num_prompts


def load_string_test_data():
  """Loads a list of strings for testing.

  Yields:
    A prompt to be used for perplexity score computation.
  """
  ds = [
      "The color of the sky is blue but sometimes it can also be",
      "The color of the sky is yellow but sometimes it can also be",
      "The color of the sky is red but sometimes it can also be",
      "The color of the sky is green but sometimes it can also be",
      "The color of the sky is orange but sometimes it can also be",
      "The color of the sky is purple but sometimes it can also be",
      "The color of the sky is brown but sometimes it can also be",
      "The color of the sky is white but sometimes it can also be",
  ]

  for i in ds:
    text_to_encode = i
    yield text_to_encode


def load_edge_cases():
  """Loads extreme edge cases for testing.

  Yields:
    A prompt to be used for perplexity score computation.
  """
  ds = [
      "",
      "a",
      "a" * 10000,
  ]

  yield from ds


def load_wikitext():
  """Loads all the prompts from the wikitext validation dataset."""
  return load_limited_wikitext(-1)


def load_limited_wikitext(num_prompts: int):
  """Loads the wikitext validation dataset.

  We leverage the wikitext validation dataset for perplexity score computation.
  This dataset is used in other places for testing.

  Args:
    num_prompts: The number of prompts to load from the dataset. If <= -1, load
      all prompts.

  Yields:
    A single concatenated string to be used for perplexity score computation.
  """
  # Load the wikitext dataset.
  source = tfds.data_source(
      "huggingface:wikitext/wikitext-103-v1:1.0.1", split="validation"
  )
  ds = grain.MapDataset.source(source)
  if num_prompts < 0:
    n = len(ds)
  else:
    n = min(num_prompts, len(ds))

  texts = []
  # Iterate over the dataset and append the text to encode.
  for i in range(_get_num_prompts(num_prompts, n)):
    # Get text from the dataset. Skip empty samples.
    raw_text = ds[i]["text"]
    text_to_encode = (
        raw_text.decode("utf-8")
        if isinstance(raw_text, bytes)
        else str(raw_text)
    )
    if not text_to_encode:
      continue
    texts.append(text_to_encode)

  if texts:
    yield "\n\n".join(texts)


class DatasetType(enum.Enum):
  """The type of dataset to use for the quality benchmark."""

  WIKITEXT = "wikitext"
  STRING_TEST_DATA = "string_test_data"
  EDGE_CASES = "edge_cases"


# LINT.IfChange
_DATASET_LOADER_MAP = {
    DatasetType.WIKITEXT: load_wikitext,
    DatasetType.STRING_TEST_DATA: load_string_test_data,
    DatasetType.EDGE_CASES: load_edge_cases,
}


def get_dataset_loader(dataset_type: DatasetType) -> Iterable[Any]:
  """Returns the dataset loader for the given dataset type."""
  return _DATASET_LOADER_MAP[dataset_type]()
# LINT.ThenChange(../../../../g3doc/benchmarking.md)
