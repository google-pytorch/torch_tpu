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

"""Generates text using a HuggingFace Transformers model from the command line."""

from typing import Final, Sequence

from absl import app
from absl import flags
from examples.benchmarks.huggingface_transformers import generate

_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3-0.6B",
    "HuggingFace model ID to use.",
)
_PROMPT = flags.DEFINE_string(
    "prompt",
    (
        "Write a program that count vowels in a string. Be brief. "
        "Then write a manual for the function as a rap song."
    ),
    "Prompt to generate text from.",
)
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens",
    320,
    (
        "Maximum number of tokens in a sequence. The input prompt and generated"
        " text may not exceed this length."
    ),
)
_DEVICE = flags.DEFINE_enum_class(
    "device", generate.Device.CPU, generate.Device, "Device to run on."
)
_STREAMING = flags.DEFINE_bool(
    "streaming", False, "Whether to stream the generated text."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  lines: Final = f"{'-' * 20}"

  print(f"\n{lines} Prompt {lines}")
  print(f"{_PROMPT.value}")

  if not _STREAMING.value:
    # Non-streaming generation
    print(f"\n{lines} Generated text. {lines}")
    generated_text, model_output, stats = generate.generate_text(
        model_id=_MODEL_ID.value,
        messages=[{"role": "user", "content": _PROMPT.value}],
        max_tokens=_MAX_TOKENS.value,
        device=_DEVICE.value,
    )
    print(f"Generated text: {generated_text}")
    print(f"Model output: {model_output}")

    print(f"\n{lines} Statistics {lines}")
    print(f"Time taken: {stats.total_time:.2f} seconds")
    print(f"Input tokens: {stats.input_token_count}")
    print(f"Total new tokens generated: {stats.num_generated_tokens}")
    print(f"Tokens per second: {stats.tokens_per_second:.2f}")
  else:
    # Streaming generation
    print(f"\n{lines} Generated text (streaming) {lines}")
    streamer = generate.generate_text_streaming(
        model_id=_MODEL_ID.value,
        messages=[{"role": "user", "content": _PROMPT.value}],
        max_tokens=_MAX_TOKENS.value,
        device=_DEVICE.value,
    )

    for chunk in streamer:
      print(chunk, end="", flush=True)
    print("\n")  # Add a newline after streaming


if __name__ == "__main__":
  app.run(main)
