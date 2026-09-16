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

"""Trace aten ops in the HuggingFace Transformers models.

This script traces the aten ops in the HuggingFace Transformers models.
It is useful for identifying missing aten ops.

See third_party/py/torch_tpu/examples/benchmarks/huggingface_transformers/README.md for
more details.

Usage:
```
blaze run //examples/benchmarks/huggingface_transformers:trace_ops -- \
  --alsologtostderr
  --model_id=<model/name>
```
"""

import collections

from absl import app
from absl import flags
from absl import logging
import torch
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import tracer_utils
from torch_tpu._internal.utils import utils
from examples.benchmarks.huggingface_transformers import model_configs
import transformers


log_utils.log_to_stderr()


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_id",
    None,
    "The model id to trace, e.g. 'openai-community/gpt2'.",
)


def _instantiate_model(
    config: transformers.PretrainedConfig,
) -> transformers.PreTrainedModel:
  for architecture in config.architectures:  # pyrefly: ignore[not-iterable]
    if hasattr(transformers, architecture):
      model_cls = getattr(transformers, architecture)
      return model_cls._from_config(config)
  raise ValueError(f"Could not instantiate model from config: {config}")


def _minus(first_tracer: utils.OpTracer, second_tracer: utils.OpTracer):
  """Removes second's ops from first's ops.

  Mutates first_tracer in place.

  Args:
    first_tracer: The first tracer.
    second_tracer: The second tracer.
  """
  for outer_key, inner_dict in first_tracer.ops_log.items():
    if outer_key in second_tracer.ops_log:
      for inner_key in list(inner_dict.keys()):
        inner_dict[inner_key] -= second_tracer.ops_log[outer_key].get(
            inner_key, 0
        )
        if inner_dict[inner_key] == 0:
          del inner_dict[inner_key]


def main(argv: list[str]):
  """Trace the ops in the forward pass, backward pass, and optimizer step for some HuggingFace Transformers models.

  Args:
    argv: Unused.

  Raises:
    UsageError: If there are too many command-line arguments.
    ValueError: If the model_id is not valid.
  """

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.model_id:
    try:
      configs_to_trace = [
          (FLAGS.model_id, model_configs.create_config_loader(FLAGS.model_id)())
      ]
    except ValueError as e:
      logging.exception(
          "Invalid model_id: %s. See README for instructions on adding a new"
          " model config.",
          FLAGS.model_id,
      )
      raise e
  else:
    configs_to_trace = model_configs.get_mini_model_configs()

  for model_id, config in configs_to_trace:
    # Trace 1: Constructor, may trigger lazy package loading of transformers
    with utils.OpTracer() as first_tracer:
      _ = _instantiate_model(config)  # pyrefly: ignore[bad-argument-type]

    # Trace 2: Constructor again, without triggering lazy package loading.
    with utils.OpTracer() as second_tracer:
      model = _instantiate_model(config)  # pyrefly: ignore[bad-argument-type]

    is_text_model = hasattr(model.config, "vocab_size")
    is_image_model = hasattr(model.config, "image_size")
    if is_text_model:
      x = torch.randint(0, model.config.vocab_size, (2, 2))
      target = torch.zeros_like(x)
      inputs = {"input_ids": x, "labels": target}
    elif is_image_model:
      x = torch.randn(2, 3, 224, 224)
      target = torch.zeros(2, dtype=torch.long)
      inputs = {"pixel_values": x, "labels": target}
    else:
      raise ValueError(
          "Only text and image models are supported right now. Model id:"
          f" {model_id}"
      )

    # Trace 3: Forward pass.
    with utils.OpTracer() as forward_tracer:
      # This isn't the best way to call a huggingface model, but ok for tracing.
      pred = model(**inputs)

    # Trace 4: Backward pass with optimizer.
    with utils.OpTracer() as backward_tracer:
      optimizer = torch.optim.AdamW(model.parameters())
      pred.loss.backward()
      optimizer.step()

    # Trace 5: Generate (includes prefill and kvcache).
    with utils.OpTracer() as gen_tracer:
      if is_text_model:
        _ = model.generate(
            x,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    _minus(first_tracer, second_tracer)

    logging.info("Tracing model: %s", model_id)
    logging.info(
        "Lazy load of package ops: %s",
        tracer_utils.pformat_op_tracer(first_tracer),
    )
    logging.info(
        "Model constructor ops: %s",
        tracer_utils.pformat_op_tracer(second_tracer),
    )
    logging.info(
        "Forward pass ops: %s",
        tracer_utils.pformat_op_tracer(forward_tracer),
    )
    logging.info(
        "Generate ops: %s",
        tracer_utils.pformat_op_tracer(gen_tracer),
    )
    logging.info(
        "Backward pass & optimizer step ops: %s",
        tracer_utils.pformat_op_tracer(backward_tracer),
    )


if __name__ == "__main__":
  app.run(main)
