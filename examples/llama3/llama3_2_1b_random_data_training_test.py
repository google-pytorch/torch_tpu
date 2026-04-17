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

"""Tests of Llama3 as implemented by HuggingFace Transformers."""

import os
import pprint
import sys
import time
from typing import Any, Callable, Dict

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
import transformers

from torch_tpu._internal.shims.xprof import traceme
from rules_python.python.runfiles import runfiles

log_utils.log_to_stderr()


_NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 10, "Numer of epochs.")

_NUM_BATCHES = flags.DEFINE_integer(
    "num_batches", 4, "Numer of batches in each epoch."
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 4, "Batch size for model forward pass."
)

_SEQ_LEN = flags.DEFINE_integer(
    "seq_len", 1024, "Sequence length for model forward pass."
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
    "compile_optim",
    False,
    "Whether to torch.compile the optimizer. If use_torch_compile == False, "
    " this flag is overridden to False.",
)

_TRAINING_STYLE = flags.DEFINE_enum(
    "training_style",
    "PYTORCH",
    ["PYTORCH", "JAX"],
    "Training style to test. Can be 'PYTORCH' or 'JAX'; default is 'PYTORCH'."
    " This primarily affects the behavior when using torch.compile; PyTorch"
    " will compile separate executables for forward/backward, JAX will compile"
    " a single executable using grad_and_value.",
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

BASE_MODEL_CONFIG_PATH = "__main__/examples/huggingface_transformers/model_configs"


def get_eager_mode() -> execution_mode.EagerMode:
  if _EAGER_MODE.value == "DEFER_AND_FUSE":
    return execution_mode.EagerMode.DEFER_AND_FUSE
  elif _EAGER_MODE.value == "DEFER_NEVER":
    return execution_mode.EagerMode.DEFER_NEVER
  elif _EAGER_MODE.value == "DEFER_NEVER_AND_LAUNCH_BLOCKING":
    return execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
  else:
    raise ValueError(f"Unsupported defer mode: {_EAGER_MODE.value}")


def get_torch_device() -> torch.device:
  if _DEVICE.value == "tpu":
    return torch.device("tpu")
  elif _DEVICE.value == "cuda":
    return torch.device("cuda")
  elif _DEVICE.value == "xla_cuda":
    return torch.device("xla_cuda")
  else:
    raise ValueError(f"Unsupported device: {_DEVICE.value}")


def sync_device(tensor_to_sync: torch.Tensor, wait=True) -> None:
  if _DEVICE.value == "tpu" or _DEVICE.value == "xla_cuda":
    # Wait for the compilation and execution of model output to complete.
    sync.synchronize(tensor_to_sync, wait=wait)
  elif _DEVICE.value == "cuda":
    torch.cuda.synchronize()


def torch_compile_model(model):
  if _DEVICE.value == "cuda":
    model = torch.compile(model)
  elif _DEVICE.value in ("tpu", "xla_cuda"):
    model = torch.compile(
        model, dynamic=False, backend=torch_tpu_compile.TpuBackend()
    )
  return model


def get_optimizer_step_fn(optimizer):
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
    steps_per_epoch: The number of steps per epoch.
  """
  cache_misses = stats["cache_misses"]
  final_warmup_epoch = None
  for epoch in range(epochs - 1):
    if cache_misses[epoch] == cache_misses[epoch + 1]:
      final_warmup_epoch = epoch
      break
  if final_warmup_epoch == None:
    logging.error(
        "Cannot calculate training step time as number of cache misses hasn't"
        " stabilized."
    )
    return
  stable_step_times = stats["step_times"][final_warmup_epoch + 1 :]
  avg_stable_step_time = sum(stable_step_times) / len(stable_step_times)

  # The preheat time is mostly the time spent on initial compilation. This may
  # take multiple epochs to stabilize. We measure the cumulative overhead,
  # rather than a per-epoch average, so that we get a stable metric even if the
  # number of warmup epochs changes.
  preheat_step_times = stats["step_times"][: final_warmup_epoch + 1]
  preheat_overhead = (
      sum(preheat_step_times) - len(preheat_step_times) * avg_stable_step_time
  )

  logging.info("Exporting metrics to TensorBoard...")
  benchmarking.record_tensorboard_metrics(
      writer,
      benchmarking.METRIC_PROFILES["Training/PreheatOverhead"],
      [preheat_overhead],
  )
  benchmarking.record_tensorboard_metrics(
      writer,
      benchmarking.METRIC_PROFILES["Training/StepTime"],
      stable_step_times,
  )


def _make_pytorch_style_training_step(
    num_batches,
    model,
    optimizer,
) -> Callable[[int, torch.Tensor, torch.Tensor], float]:
  """A training step function that is the typical PyTorch-style loop."""
  # Prepare the model and optimizer using torch.compile if needed.
  if _USE_TORCH_COMPILE.value:
    model = torch_compile_model(model)
  optimizer_step_fn = get_optimizer_step_fn(optimizer)

  def _pytorch_style_training_step(epoch, inputs, targets) -> float:
    accumulated_losses = []
    logging.info(f"Epoch {epoch} : Zero Grad.")
    optimizer.zero_grad()
    for batch in range(num_batches):
      logging.info(f"Batch {batch}, Epoch {epoch} : Forward.")
      output = model(inputs, labels=targets)
      logging.info(f"Batch {batch}, Epoch {epoch} : Backward.")
      output.loss.backward()
      accumulated_losses.append(output.loss.detach())

    logging.info(f"Epoch {epoch} : Optimizer.")
    optimizer_step_fn()
    logging.info(f"Epoch {epoch} : Loss aggregation.")
    step_loss = torch.sum(torch.stack(accumulated_losses)).item()
    return step_loss

  return _pytorch_style_training_step


def _make_jax_style_training_step(
    num_batches, model, optimizer
) -> Callable[[int, torch.Tensor, torch.Tensor], float]:
  """A training step function that is similar to JAX, using torch.func tools.

  References:
  https://docs.pytorch.org/docs/stable/func.whirlwind_tour.html
  https://docs.pytorch.org/docs/stable/generated/torch.func.grad.html#torch-func-grad
  https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html
  https://medium.com/data-science/introduction-to-functional-pytorch-b5bf739e1e6e

  Args:
    num_batches: The number of batches to accumulate gradients over.
    model: The PyTorch model to train.
    optimizer: The PyTorch optimizer to use for training.

  Returns:
    A function that takes an epoch number and inputs/targets tensors, trains
    the model for a single epoch, and returns the accumulated loss.
  """
  optimizer_step_fn = get_optimizer_step_fn(optimizer)

  # Convert the model into a function using torch.func.functional_call.
  def _functional_model(
      params: dict[str, Any],
      inputs: torch.Tensor,
      targets: torch.Tensor,
  ) -> torch.Tensor:
    outputs = torch.func.functional_call(
        model,
        params,
        inputs,
        kwargs={"labels": targets},
    )
    loss = outputs.loss
    loss = loss / num_batches
    return loss

  def _accumulate_batch(
      params: dict[str, Any],
      inputs: torch.Tensor,
      targets: torch.Tensor,
      grads: dict[str, Any] | None = None,
      accumulated_loss: torch.Tensor | None = None,
  ) -> tuple[dict[str, Any], torch.Tensor]:
    # grad_and_value transforms the graph so we don't need to call a second
    # backward() call on it; we can use torch.no_grad() here.
    # This also means we don't need to call detach() on the loss, as it already
    # detached.
    with torch.no_grad():
      new_grads, new_loss = torch.func.grad_and_value(_functional_model)(
          params, inputs, targets
      )
      if grads is None:
        grads = new_grads
        accumulated_loss = new_loss
      else:
        for old_grad, new_grad in zip(grads.values(), new_grads.values()):
          old_grad.add_(new_grad)
        accumulated_loss.add_(new_loss)
    return grads, accumulated_loss

  def _accumulate_batches(
      params: dict[str, Any],
      inputs: torch.Tensor,
      targets: torch.Tensor,
  ) -> tuple[dict[str, Any], torch.Tensor]:
    grads = None
    accumulated_loss = None
    for _ in range(num_batches):
      grads, accumulated_loss = _accumulate_batch(
          params, inputs, targets, grads, accumulated_loss
      )
    return grads, accumulated_loss

  if _USE_TORCH_COMPILE.value:
    _accumulate_batches = torch_compile_model(_accumulate_batches)

  def _jax_style_training_step(epoch, inputs, targets) -> float:
    logging.info(f"Epoch {epoch} : Zero Grad.")
    optimizer.zero_grad()
    logging.info(f"Epoch {epoch} : Accumulate batches.")
    grads, accumulated_loss = _accumulate_batches(
        dict(model.named_parameters()), inputs, targets
    )
    logging.info(f"Epoch {epoch} : Optimizer.")
    for name, param in model.named_parameters():
      param.grad = grads[name]
    optimizer_step_fn()
    logging.info(f"Epoch {epoch} : Loss aggregation.")
    step_loss = accumulated_loss.item()
    return step_loss

  return _jax_style_training_step


class Llama321BRandomDataTrainingTest(absltest.TestCase):

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
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)

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
    """Test llama3.2 tiny model random data training."""
    batch_size = _BATCH_SIZE.value
    seq_len = _SEQ_LEN.value
    device = get_torch_device()
    execution_mode.set_eager_mode(get_eager_mode())

    # Arrange
    model_config_path = runfiles.Create().Rlocation(
        f"{BASE_MODEL_CONFIG_PATH}/meta-llama/Llama-3.2-1B/config.json"
    )
    default_config = transformers.AutoConfig.from_pretrained(model_config_path)
    model = transformers.AutoModelForCausalLM.from_config(default_config)

    model = model.to(device)

    # Optimizer need to be in "capturable" if compiled, other wise the steps are
    # created on CPU and we will see CPU input to TPU graph module
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.1,
        capturable=_COMPILE_OPTIM.value and _USE_TORCH_COMPILE.value,
        # The non-fused version of adam will expand foreach into loops and
        # increase the graph size. The compile time increase is especially
        # obvious when compiling AdamW with torch.compile.
        # We should consider implementing 'aten::_fused_adamw_' to both
        # improve compile time and increase performance.
        fused=False,
    )  # Gigantic LR for testing.

    inputs = torch.full(
        (batch_size, seq_len),
        default_config.bos_token_id,
        dtype=torch.int64,
        device=device,
    )
    targets = torch.full(
        (batch_size, seq_len),
        42,
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

    if _TRAINING_STYLE.value == "PYTORCH":
      training_step_fn = _make_pytorch_style_training_step(
          num_batches, model, optimizer
      )
    elif _TRAINING_STYLE.value == "JAX":
      training_step_fn = _make_jax_style_training_step(
          num_batches, model, optimizer
      )
    else:
      raise ValueError(f"Unsupported training style: {_TRAINING_STYLE.value}")

    for epoch in range(num_epochs):
      with traceme.TraceMe(f"Epoch_{epoch}_Train"):
        step_start_time = time.time()

        step_loss = training_step_fn(epoch, inputs, targets)

        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        logging.info(f"Epoch {epoch}, loss: {step_loss}, time: {step_time}s")
        if _DEVICE.value in ("tpu", "xla_cuda"):
          hbm_usage_summary = getattr(torch, _DEVICE.value)._hbm_usage_summary()
          logging.info(f"Epoch {epoch}, HBM usage summary: {hbm_usage_summary}")
          cache_misses = getattr(torch, _DEVICE.value)._get_cache_misses()
          logging.info(
              f"Epoch {epoch}, compilation cache misses: {cache_misses}"
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
    else:
      pprint.pp(stats)

    if _DEVICE.value == "cuda":
      peak_memory_usage_mb = torch.cuda.memory.max_memory_allocated() / 1048576
      print(
          f"ACTUAL CUDA MEMORY USAGE: {peak_memory_usage_mb:.2f} MB",
          flush=True,
      )


if __name__ == "__main__":
  absltest.main()
