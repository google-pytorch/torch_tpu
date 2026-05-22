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

"""Base class for quality benchmark models."""

import abc
import dataclasses
import hashlib
from typing import Any, Iterable, final

from absl import logging
import torch
from torch_tpu._internal.utils import log_utils


log_utils.log_to_stderr()


class MetricProducer(abc.ABC):
  """Abstract base class for quality benchmark metrics."""

  @abc.abstractmethod
  def get_name(self) -> str:
    """Returns the name of the metric."""
    raise NotImplementedError

  @abc.abstractmethod
  def assess(
      self,
      model_input: Any,
      benchmark_model: "QualityBenchmarkModel",
  ) -> torch.Tensor | float:
    """Assess the quality of the model on a given input.

    Args:
      model_input: The input to assess.
      benchmark_model: The benchmark model object.

    Returns:
      The score of the metric.
    """
    raise NotImplementedError


@dataclasses.dataclass
class FormattedInput:
  """Formatted input for the model."""

  # The formatted input tensor.
  input: torch.Tensor

  # The length of the unpadded tokens. This is important to ensure metrics do n
  # not process padding tokens.
  unpadded_length: int


class QualityBenchmarkModel(abc.ABC):
  """Abstract base class for quality benchmark models.

  The class stores a torch.nn.Module model and any other necessary information
  for the methods to be called. It is expected that the model is loaded from a
  pre-trained checkpoint before any other methods are called. The model can be
  compiled before the model is returned or after the model is returned. The
  compile_model method can be used to compile the model before it is used.

  Some sudo code for expected usage of this class is as follows:

  qbm = QualityBenchmarkModel(init_args)
  if flag.compile_model:
    qbm.compile_model()

  for sample in data_loader():
    benchmark_function(input, qbm)
  """

  def __init__(self):
    self._model_compiled = False

  @property
  @abc.abstractmethod
  def max_seq_len(self) -> int:
    """The maximum sequence length supported by the model."""
    raise NotImplementedError

  @abc.abstractmethod
  def encode(self, text: str) -> list[int]:
    """Encodes the text into tokens without padding or truncation."""
    raise NotImplementedError

  @abc.abstractmethod
  def initialize(self) -> None:
    """Initializes the model.

    This method is called once per worker to initialize the model. It assumes
    that any necessary environment variables or flags have already been set.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_model(self) -> torch.nn.Module:
    """Gets the model from a pre-trained checkpoint.

    get_model is called once per worker to retrieve the model from a
    pre-trained checkpoint. It assumes the model has already been loaded from
    the checkpoint and any other initialization has been done.

    Returns:
      The model from a pre-trained checkpoint.
    """
    raise NotImplementedError

  @final
  def compile_model(self) -> None:
    """Compiles model used by the interface.

    This method is called to compile the model used by the interface. It
    changes the state of the model such that "get_model" returns the compiled
    model. This method is idempotent - calling it multiple times has the same
    effect as calling it once.
    """

    if self._model_compiled:
      logging.info("Model already compiled. Skipping compilation.")
      return
    self._model_compiled = True
    self._compile_model_once()

  @abc.abstractmethod
  def _compile_model_once(self) -> None:
    """Compiles model used by the interface.

    This method implements the compilation of the model. It's guaranteed to be
    called only once.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def format(self, raw_input: Any) -> FormattedInput:
    """Formats the input for the model.

    Args:
      raw_input: The raw input to format for the model. Can be either a str or a
        list[int] of tokens.

    Returns:
      A tuple containing the formatted tensor and the length of the unpadded
      tokens.
    """
    raise NotImplementedError

  def get_logits_and_targets(
      self, formatted_input: FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns logits and targets aligned for loss calculation.

    This method performs the forward pass on the given formatted_input and
    returns a tuple of (logits, targets).

    Args:
      formatted_input: The formatted input for the model.

    Returns:
      A tuple of (logits, targets).
    """
    raise NotImplementedError

  def get_intermediate_outputs(
      self, formatted_input: FormattedInput
  ) -> dict[str, torch.Tensor]:
    """Returns a dictionary of intermediate activations.

    This method is a foundation for layer checks. It performs a forward pass
    and returns a dictionary mapping layer names to their output tensors.

    Note: There is a risk of memory Out Of Memory (OOM) errors if intermediate
    outputs from all layers are returned at once.

    Args:
      formatted_input: The formatted input for the model.

    Returns:
      A dictionary of intermediate activations mapping layer names to tensors.
    """
    raise NotImplementedError

  def get_logits_and_targets(
      self, formatted_input: FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns logits and targets aligned for loss calculation.

    This method performs the forward pass on the given formatted_input and
    returns a tuple of (logits, targets).

    Args:
      formatted_input: The formatted input for the model.

    Returns:
      A tuple of (logits, targets).
    """
    raise NotImplementedError

  def get_intermediate_outputs(
      self, formatted_input: FormattedInput
  ) -> dict[str, torch.Tensor]:
    """Returns a dictionary of intermediate activations.

    This method is a foundation for layer checks. It performs a forward pass
    and returns a dictionary mapping layer names to their output tensors.

    Note: There is a risk of memory Out Of Memory (OOM) errors if intermediate
    outputs from all layers are returned at once.

    Args:
      formatted_input: The formatted input for the model.

    Returns:
      A dictionary of intermediate activations mapping layer names to tensors.
    """
    raise NotImplementedError


def run_quality_benchmark(
    benchmark_model: QualityBenchmarkModel,
    model_compile: bool,
    dataset_iterator: Iterable[Any],
    benchmark_metric: MetricProducer,
) -> torch.Tensor | float:
  """Runs quality benchmark for a given model on a dataset.

  Args:
    benchmark_model: The benchmark model object.
    model_compile: Whether to compile the model.
    dataset_iterator: An iterable of strings representing the dataset.
    benchmark_metric: A function that runs the benchmark metric. Returns the
      metric score for the given input.

  Returns:
    The average score of the benchmark runs for the given dataset.
  """
  if model_compile:
    benchmark_model.compile_model()

  scores = []
  run_id = 0

  for model_input in dataset_iterator:
    score = benchmark_metric.assess(model_input, benchmark_model)
    # Ensure model_input is a string for hashing if needed, or handle appropriately.
    # Assuming model_input is string-like or can be converted to str for logging hashing
    h = hashlib.sha256(str(model_input).encode("utf-8")).hexdigest()
    logging.info("Score for run %d (HASH: %s): %s", run_id, h, score)

    run_id += 1
    scores.append(score)

  if not scores:
    return 0.0

  score_avg = sum(scores) / len(scores)
  return score_avg
