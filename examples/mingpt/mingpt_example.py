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

import time
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal.utils import utils
from examples.mingpt.impl import mingpt
from examples.mingpt.impl import mingpt_jax

CheckValueMode = utils.CheckValueMode

_DEVICE = flags.DEFINE_string(
    "device", "tpu", "Device: [tpu, cpu, xla_cuda, xla_cpu]"
)
GPT = mingpt.GPT

native_device_d = None

run_cpu = True
run_jax = False
run_torch = True


# TODO(bbahl): Update this to use mingpt_runner_lib.
def main(argv):
  native_device_d = torch.accelerator.current_accelerator()
  native_device = str(native_device_d)
  # TODO(bbahl): Investigate whether we need this flag at all
  assert native_device.split(":", 1)[0] == _DEVICE.value

  common_dtype = torch.float32
  # model = "gpt2-large"
  model = "tpu-optimal-l-hd128"
  # model = 'gpt-micro'

  torch.manual_seed(42)
  model_ref_cpu = None
  model_tpu = None
  x_input_cpu = None
  x_input_tpu = None

  if run_cpu:
    config = GPT.get_default_config()
    # config.model_type = "gpt-micro"
    config.model_type = model
    config.vocab_size = 100
    config.block_size = 1024

    model_ref_cpu = GPT(config, device="cpu").to(common_dtype)
    model_ref_cpu.eval()

    # Generate a single input tensor on CPU
    x_input_cpu = torch.randint(
        0, config.vocab_size, (2, config.block_size), dtype=torch.long
    )

    fx = torch.fx.experimental.proxy_tensor.make_fx(model_ref_cpu)(x_input_cpu)
    print(fx)

  if run_torch:
    config2 = GPT.get_default_config()
    # config2.model_type = "gpt-micro"
    config2.model_type = model
    config2.vocab_size = 100
    config2.block_size = 1024

    # We don't have all the init functions implemented for NATIVE_DEVICE yet, so we
    # need to load the weights from CPU.
    model_tpu = GPT(config2, device="cpu").to(common_dtype)

    if run_cpu:
      model_tpu.load_state_dict(
          model_ref_cpu.state_dict()
      )  # Ensure weights are identical
    model_tpu.to(native_device_d)
    model_tpu.eval()
    print("Model loaded")

    if run_cpu:
      x_input_tpu = x_input_cpu.to(native_device_d)
    else:
      x_input_tpu = torch.randint(
          0,
          config2.vocab_size,
          (2, config2.block_size),
          dtype=torch.long,
          device=native_device_d,
      )

  if run_jax:
    print("\n=== JAX Comparison ===")
    jax_config = mingpt_jax.get_default_config()
    jax_config.model_type = model
    jax_config.vocab_size = 100
    jax_config.block_size = 1024
    mingpt_jax.apply_model_type(jax_config)
    rng = jax.random.PRNGKey(0)
    gpt_params = mingpt_jax.init_gpt_params(rng, jax_config)

    if run_cpu:
      x_jax = np.array(x_input_cpu.numpy())
    else:
      x_jax = np.array(
          torch.randint(
              0,
              jax_config.vocab_size,
              (2, jax_config.block_size),
              dtype=torch.long,
          )
      )

  def run(model_ref_cpu, model_tpu, x_input_cpu, x_input_tpu, log):
    print("Running...")
    with torch.no_grad():
      if run_cpu:
        start_time_cpu = time.time()
        logits_ref_cpu, loss_cpu = model_ref_cpu(x_input_cpu)
        end_time_cpu = time.time()
        cpu_time = end_time_cpu - start_time_cpu
        if log:
          print(f"CPU time: {cpu_time:.4f} seconds")
      else:
        logits_ref_cpu = None

      if run_torch:
        start_time_tpu = time.time()
        logits_from_tpu, loss_tpu = model_tpu(x_input_tpu)
        end_time_tpu = time.time()
        tpu_time = end_time_tpu - start_time_tpu
        if log:
          print(f"NATIVE_DEVICE time: {tpu_time:.4f} seconds")
      else:
        logits_from_tpu = None

    if run_jax:
      start_time_jax = time.time()
      logits_jax, _ = mingpt_jax.gpt_forward(gpt_params, jnp.array(x_jax), None)
      end_time_jax = time.time()
      jax_time = end_time_jax - start_time_jax
      logits_jax = np.array(logits_jax)
      if log:
        print(f"JAX time: {jax_time:.4f} seconds")

    return logits_ref_cpu, logits_from_tpu

  # preheat
  run(model_ref_cpu, model_tpu, x_input_cpu, x_input_tpu, log=True)
  preheat_cache_misses = getattr(torch, native_device)._get_cache_misses()
  print(f"preheat_cache_misses={preheat_cache_misses}")

  num_runs = 4
  with jax.named_scope("real_runs"):
    for _ in range(num_runs):
      logits_ref_cpu, logits_from_tpu = run(
          model_ref_cpu, model_tpu, x_input_cpu, x_input_tpu, log=True
      )
      # force sync the async compilation, to show how long it takes,
      # even when we force a blocking wait.
      cpu_logits = logits_from_tpu.to("cpu")
      utils.assert_close(
          cpu_logits,
          logits_ref_cpu,
          check_value=CheckValueMode.LOOSE,
          rtol=1e-3,
          atol=1e-4,
      )

      # Check there are no new cache misses.
      cache_misses = getattr(torch, native_device)._get_cache_misses()
      print(f"cache_misses={cache_misses}")
      cache_stats = getattr(torch, native_device)._get_cache_stats()
      for entry in cache_stats.per_entry_stats:
        print(
            f" compilation_duration={entry.compilation_duration},"
            f" last_read={entry.last_read},"
            f" read_count={entry.read_count}"
        )

  if run_torch:
    assert logits_from_tpu.device.type == native_device_d.type
    logits_tpu_to_cpu = logits_from_tpu.cpu()

  if run_cpu and run_torch:
    print(f"logits_tpu_to_cpu.shape: {logits_tpu_to_cpu.shape}")
    print(f"logits_ref_cpu.shape: {logits_ref_cpu.shape}")

  utils.assert_close(
      actual=logits_tpu_to_cpu,
      expected=logits_ref_cpu,
      check_value=CheckValueMode.LOOSE,
      rtol=1e-3,
      atol=1e-4,
      preamble="MiniGPT outputs",
  )


if __name__ == "__main__":
  app.run(main)
