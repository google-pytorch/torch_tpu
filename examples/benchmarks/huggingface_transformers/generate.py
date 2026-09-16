# Copyright 2025 Google LLC
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

"""Library for generating text using HuggingFace Transformers models.

There are four different functions for generating text:
  - generate_text: Generates text using the HuggingFace model.
  - generate_text_streaming: Streams generated text from the HuggingFace model.
"""

import dataclasses
import enum
import threading
import time
from typing import Dict, Iterator, List, Tuple

from absl import logging
from etils import epath
import torch
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers
from transformers import masking_utils
from transformers import modeling_outputs


log_utils.log_to_stderr()


@enum.unique
class Device(enum.Enum):
  CUDA = "cuda"
  TPU = "tpu"
  CPU = "cpu"


@dataclasses.dataclass
class GenerationStats:
  input_token_count: int
  num_generated_tokens: int
  total_time: float
  tokens_per_second: float


WEIGHTS_DIR = f"{paths.XM_HOME}weights/huggingface"


def _get_model_path(model_id: str) -> epath.Path:
  """Returns the path for a given model_id."""
  # Check for existence of model_id in WEIGHTS_DIR.
  model_dir = epath.Path(WEIGHTS_DIR) / model_id
  if not model_dir.exists():
    raise ValueError(
        f"Model directory not found: {model_dir}. Available models:"
        f" {model_dir.parent.iterdir()}"
    )
  return model_dir


def _get_model_and_tokenizer(model_id: str, device: Device) -> Tuple[
    transformers.PreTrainedModel,
    transformers.PreTrainedTokenizerBase,
    torch.device,
]:
  """Loads the model and tokenizer.

  Args:
    model_id: The identifier of the model to use.
    device: The device to use. One of "tpu", "cuda", or "cpu".

  Returns:
    A tuple containing:
      - model: The loaded model.
      - tokenizer: The loaded tokenizer.
      - device: The loaded device.

  Raises:
    RuntimeError: If CUDA is requested but not available.
    ValueError: If an unsupported device is requested.
  """
  model_dir = _get_model_path(model_id)

  device = torch.device(device.value)

  logging.info("Using device: %s", device)
  logging.info("Loading model: %s", model_dir)

  tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      model_dir, torch_dtype="auto"
  ).to(device)
  model.eval()

  logging.info("Model loaded on %s", device)
  logging.info("Number of parameters: %s", model.num_parameters())
  return model, tokenizer, device  # pyrefly: ignore[bad-return]


def _check_prompt_length(prompt_token_count: int, max_tokens: int):
  """Checks if the prompt is shorter than max_tokens.

  Args:
    prompt_token_count: The number of tokens in the prompt.
    max_tokens: The maximum number of tokens allowed.

  Raises:
    ValueError: If the prompt is longer than max_tokens.
  """
  if prompt_token_count >= max_tokens:
    raise ValueError(
        f"Input prompt is too long ({prompt_token_count} tokens). Please"
        f" provide a prompt shorter than max_tokens ({max_tokens})."
    )


def generate_text(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    device: Device,
) -> Tuple[str, modeling_outputs.CausalLMOutputWithPast, GenerationStats]:
  """Generates text using the HuggingFace model.

  This function operates in non-streaming mode: the function will
  block until the entire output is generated.

  Args:
    model_id: The identifier of the model to use.
    messages: The list of messages comprising the conversation history.
    max_tokens: The maximum number of tokens in a sequence. The input prompt and
      generated text may not exceed this length.
    device: The device to use. One of "tpu", "cuda", or "cpu".

  Returns:
    A tuple containing:
      - generated_text: A generated text string.
      - model_output: The raw model output.
      - stats: A dataclass containing generation statistics.

  Raises:
    RuntimeError: If CUDA is requested but not available.
    ValueError: If an unsupported device is requested.
    ValueError: If the input prompt is longer than or equal to max_tokens.
  """
  model, tokenizer, device = _get_model_and_tokenizer(model_id, device)

  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
  )
  inputs = tokenizer(text, return_tensors="pt").to(device)
  input_token_count = inputs.input_ids.shape[1]
  _check_prompt_length(input_token_count, max_tokens)

  start_time = time.perf_counter()

  outputs = model.generate(
      **inputs,
      max_new_tokens=max_tokens - input_token_count,
      do_sample=False,
      temperature=None,
      top_p=None,
      top_k=None,
      pad_token_id=tokenizer.pad_token_id,
      return_dict_in_generate=True,
      output_scores=True,
  )
  generated_ids = outputs.sequences
  end_time = time.perf_counter()

  total_time = end_time - start_time
  generated_texts = tokenizer.batch_decode(generated_ids)[0]
  num_generated_tokens = generated_ids.shape[1] - input_token_count
  tokens_per_second = num_generated_tokens / total_time if total_time > 0 else 0

  stats = GenerationStats(
      input_token_count=input_token_count,
      num_generated_tokens=num_generated_tokens,
      total_time=total_time,
      tokens_per_second=tokens_per_second,
  )

  return (generated_texts, outputs, stats)


def generate_text_streaming(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    device: Device,
) -> Iterator[str]:
  """Streams generated text from the HuggingFace model.

  This function operates in streaming mode: it will generate text in chunks
  and yield them as they are generated. No stats or model output are returned.

  Args:
    model_id: The identifier of the model to use.
    messages: The list of messages comprising the conversation history.
    max_tokens: The maximum number of tokens in a sequence. The input prompt and
      generated text may not exceed this length.
    device: The device to use. One of "tpu", "cuda", or "cpu".

  Yields:
    Streamed chunks of text.

  Raises:
    RuntimeError: If CUDA is requested but not available.
    ValueError: If an unsupported device is requested.
    ValueError: If the input prompt is longer than or equal to max_tokens.
  """
  model, tokenizer, device = _get_model_and_tokenizer(model_id, device)

  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
  )
  inputs = tokenizer(text, return_tensors="pt").to(device)
  input_token_count = inputs.input_ids.shape[1]

  _check_prompt_length(input_token_count, max_tokens)

  streamer = transformers.TextIteratorStreamer(
      tokenizer, skip_prompt=True, skip_special_tokens=True
  )

  def generate_thread():
    model.generate(
        **inputs,
        max_new_tokens=max_tokens - input_token_count,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

  thread = threading.Thread(target=generate_thread)
  thread.start()

  for new_text in streamer:
    yield new_text
