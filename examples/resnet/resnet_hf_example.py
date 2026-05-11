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

"""Performance test for Resnet50 model."""

import os
import sys
import time
from typing import Any, Dict

from absl import flags
from absl import logging
from absl.testing import absltest
import torch
import torch._inductor.config as inductor_config
from torch.utils import tensorboard
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal.utils import benchmarking
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers

from torch_tpu._internal.shims.xprof import traceme


log_utils.log_to_stderr()


MODEL_PATH = f"{paths.XM_HOME}weights/huggingface/microsoft/resnet-50"

_NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 10, "Numer of epochs.")

_NUM_BATCHES = flags.DEFINE_integer(
    "num_batches", 4, "Numer of batches in each epoch."
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 4, "Batch size for model forward pass."
)

_IMAGE_H = flags.DEFINE_integer(
    "image_h", 224, "Image height for model forward pass."
)
_IMAGE_W = flags.DEFINE_integer(
    "image_w", 224, "Image width for model forward pass."
)

_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda", "xla_cuda"],
    required=True,
    help="Accelerator to test.",
)

_USE_TORCH_COMPILE = flags.DEFINE_bool(
    "use_torch_compile",
    False,
    "Whether to use torch.compile to run the model.",
)

_EAGER_MODE = flags.DEFINE_enum(
    "eager_mode",
    "DEFER_AND_FUSE",
    [
        "DEFER_AND_FUSE",
        "DEFER_NEVER",
        "DEFER_NEVER_AND_LAUNCH_BLOCKING",
    ],
    "Eager mode for the model.",
)

_COMPILE_OPTIM = flags.DEFINE_bool(
    "complie_optim",
    False,
    "Whether to torch.compile the optimizer. If use_torch_compile == False, "
    " this flag is overridden to False.",
)

_ENABLE_TENSORBOARD_LOGGING = flags.DEFINE_bool(
    "enable_tensorboard_logging",
    False,
    "Whether to enable TensorBoard logging.",
)
_TB_SUMMARY_LOGGING_DIR = flags.DEFINE_string(
    "tb_summary_logging_dir",
    default=os.environ.get("TB_SUMMARY_LOGGING_DIR", None),
    help="TensorBoard summary logging directory.",
)


flags.register_validator(
    "test_random_seed",
    lambda value: isinstance(value, int),
    message="--test_random_seed must be an integer to be used as a seed.",
)


def _get_eager_mode() -> execution_mode.EagerMode:
  if _EAGER_MODE.value == "DEFER_AND_FUSE":
    return execution_mode.EagerMode.DEFER_AND_FUSE
  elif _EAGER_MODE.value == "DEFER_NEVER":
    return execution_mode.EagerMode.DEFER_NEVER
  elif _EAGER_MODE.value == "DEFER_NEVER_AND_LAUNCH_BLOCKING":
    return execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
  else:
    raise ValueError(f"Unsupported defer mode: {_EAGER_MODE.value}")


def _get_torch_device() -> torch.device:
  device = torch.device(_DEVICE.value)
  # TODO(pganssle): Evaluate whether this assertion is still necessary.
  assert str(device).split(":", 1)[0] == _DEVICE.value
  return device


def _sync_device(tensor_to_sync: torch.Tensor, wait=True) -> None:
  if _DEVICE.value == "tpu" or _DEVICE.value == "xla_cuda":
    # Wait for the compilation and execution of model output to complete.
    sync.synchronize(tensor_to_sync, wait=wait)
  elif _DEVICE.value == "cuda":
    torch.cuda.synchronize()


def _torch_compile_model(model):
  if _DEVICE.value == "cuda":
    return torch.compile(model)
  if _DEVICE.value in ("tpu", "xla_cuda"):
    return torch.compile(
        model, dynamic=False, backend=torch_tpu_compile.TpuBackend()
    )
  return model


def _get_optimizer_step_fn(optimizer):
  if _DEVICE.value == "cuda":
    backend = "inductor"
  elif _DEVICE.value in ("tpu", "xla_cuda"):
    backend = torch_tpu_compile.TpuBackend(debug=True)
  else:
    raise ValueError(f"Unsupported device: {_DEVICE.value}")

  should_compile_optim = _COMPILE_OPTIM.value and _USE_TORCH_COMPILE.value

  @torch.compile(backend=backend, disable=not should_compile_optim)
  def step_fn():
    optimizer.step()

  return step_fn


def _create_model_and_config(device):
  config = transformers.AutoConfig.from_pretrained(MODEL_PATH)
  model = transformers.AutoModelForImageClassification.from_config(config)
  model = model.to(device)
  if _USE_TORCH_COMPILE.value:
    model = _torch_compile_model(model)
  return model, config


def _record_metrics(
    stats: Dict[str, Any],
    writer: "tensorboard.SummaryWriter",
    epochs: int,
):
  """Calculates and records training metrics with TensorBoard.

  Args:
    stats: A dictionary containing training statistics.
    writer: An optional TensorBoard SummaryWriter for logging.
    epochs: The number of training epochs.
  """
  cache_misses = stats["cache_misses"]
  final_warmup_epoch = None
  for epoch in range(epochs - 1):
    if cache_misses[epoch] == cache_misses[epoch + 1]:
      final_warmup_epoch = epoch
      break
  if final_warmup_epoch is None:
    logging.error(
        "Cannot calculate training step time as number of cache misses hasn't"
        " stabilized."
    )
    return
  preheat_step_times = stats["step_times"][: final_warmup_epoch + 1]
  stable_step_times = stats["step_times"][final_warmup_epoch + 1 :]

  logging.info("Exporting metrics to TensorBoard...")
  benchmarking.record_tensorboard_metrics(
      writer,
      benchmarking.METRIC_PROFILES["Training/PreheatStepTime"],
      preheat_step_times,
  )
  benchmarking.record_tensorboard_metrics(
      writer,
      benchmarking.METRIC_PROFILES["Training/StepTime"],
      stable_step_times,
  )


class Resnet50RandomDataTrainingTest(absltest.TestCase):
  """Tests ResNet50 model training with random data."""

  def setUp(self):
    super().setUp()
    if _DEVICE.value in ("cuda", "xla_cuda") and not torch.cuda.is_available():
      print(
          f"WARNING: --device={_DEVICE.value} requires compiling the target"
          " with --config=cuda on the blaze command line. Please rerun the"
          " target with that flag.",
          file=sys.stderr,
      )
      # We return success here to avoid breaking
      # `blaze test //torch_tpu/...`
      sys.exit(0)

    # This abstest flag will always be set to an int.
    seed = absltest.FLAGS.test_random_seed

    # TODO(gunhyun): Figure out why inductor multiprocessing library is causing
    # issues with GPU.
    if _DEVICE.value == "cuda":
      inductor_config.compile_threads = 1
    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    self.writer = None
    if _ENABLE_TENSORBOARD_LOGGING.value:
      log_dir = _TB_SUMMARY_LOGGING_DIR.value
      if log_dir:
        logging.info("TensorBoard logging enabled. Writing to: %s", log_dir)
        self.writer = tensorboard.SummaryWriter(log_dir)
      else:
        logging.warning(
            "TensorBoard logging is enabled but --tb_summary_logging_dir is"
            " not set. No logs will be written."
        )

  def tearDown(self):
    super().tearDown()
    if self.writer:
      logging.info("Flushing and closing TensorBoard SummaryWriter.")
      self.writer.flush()
      self.writer.close()

  def test_training(self):
    """Test resnet50 model random data training."""
    batch_size = _BATCH_SIZE.value
    device = _get_torch_device()
    execution_mode.set_eager_mode(_get_eager_mode())

    # Arrange
    model, config = _create_model_and_config(device)

    # Optimizer need to be in "capturable" if compiled, other wise the steps are
    # created on CPU and we will see CPU input to TPU graph module
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=0.1,
    #     capturable=_COMPILE_OPTIM.value and _USE_TORCH_COMPILE.value,
    #     fused=False,
    # )  # Gigantic LR for testing.

    # optimizer_step_fn = get_optimizer_step_fn(optimizer)

    inputs = torch.randn(
        batch_size,
        3,
        _IMAGE_H.value,
        _IMAGE_W.value,
        dtype=torch.float32,
        device=device,
    )
    targets = torch.randint(
        0,
        config.num_labels,
        (batch_size,),
        dtype=torch.int64,
        device=device,
    )

    num_epochs = _NUM_EPOCHS.value
    num_batches = _NUM_BATCHES.value
    model.train()
    stats = {
        "epoch": [],
        "train_loss": [],
        "cache_misses": [],
        "step_times": [],
    }
    for epoch in range(num_epochs):
      with traceme.TraceMe(f"Epoch_{epoch}_Train"):
        step_start_time = time.time()
        accumulated_losses = []
        # TODO(gunhyun): Uncomment when backwards pass is supported
        # logging.info("Epoch %d : Zero Grad.", epoch)
        # optimizer.zero_grad()
        for batch in range(num_batches):
          logging.info("Batch %d, Epoch %d : Forward.", batch, epoch)
          output = model(inputs, labels=targets)
          # TODO(gunhyun): Uncomment when backwards pass is supported
          # logging.info("Batch %d, Epoch %d : Backward.", batch, epoch)
          # output.loss.backward()
          accumulated_losses.append(output.loss.detach())

        # TODO(gunhyun): Uncomment when backwards pass is supported
        # logging.info("Epoch %d : Optimizer.", epoch)
        # optimizer_step_fn()
        logging.info("Epoch %d : Loss aggregation.", epoch)
        step_loss = torch.sum(torch.stack(accumulated_losses)).item()

        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        logging.info(
            "Epoch %d, loss: %f, time: %f s", epoch, step_loss, step_time
        )
        if _DEVICE.value in ("tpu", "xla_cuda"):
          cache_misses = getattr(torch, _DEVICE.value)._get_cache_misses()
          logging.info(
              "Epoch %d, compilation cache misses: %d", epoch, cache_misses
          )
          stats["cache_misses"].append(cache_misses)
      stats["epoch"].append(epoch)
      stats["train_loss"].append(step_loss)
      stats["step_times"].append(step_time)

    if self.writer:
      _record_metrics(
          stats=stats,
          writer=self.writer,
          epochs=num_epochs,
      )

    if _DEVICE.value == "cuda":
      peak_memory_usage_mb = torch.cuda.memory.max_memory_allocated() / 1048576
      print(
          f"ACTUAL CUDA MEMORY USAGE: {peak_memory_usage_mb:.2f} MB",
          flush=True,
      )


if __name__ == "__main__":
  absltest.main()
