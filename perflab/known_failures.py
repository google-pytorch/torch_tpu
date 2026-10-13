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

"""Perflab model benchmarks that are known to fail.

Benchmark runners skip the executions that match an entry in `KNOWN_FAILURES`
the same way as the run modes in `BenchmarkSpec.skipped_run_modes`:
`torch_benchmark_test` and `torchax_benchmark_test` apply `--skip_behavior` to
both (e.g. `--skip_behavior=run_skipped` runs only the skipped executions to
check whether they still fail), and `distributed_runner` always skips them.
"""

import dataclasses

from perflab import common
from perflab import mode as mode_lib
from perflab import target as target_lib


@dataclasses.dataclass(frozen=True, kw_only=True)
class KnownFailure:
  """A model benchmark known to fail under specific execution coordinates.

  Attributes:
    benchmark: Registered `BenchmarkSpec.name` in `perflab/models` or
      `perflab/distributed_models`.
    framework: Framework (`TORCH` for TorchTPU/GPU or `TORCHAX` for TorchAX).
    device_kind: Optional device kind filter (e.g. `TPU` or `CUDA`). `None`
      matches all device kinds.
    run_modes: Optional set of `RunMode` values that fail. `None` matches all
      run modes.
    reason: Human-readable explanation of why the benchmark fails.
  """

  benchmark: str
  framework: mode_lib.Framework
  device_kind: target_lib.DeviceKind | None = None
  run_modes: frozenset[common.RunMode] | None = None
  reason: str

  def matches(
      self,
      benchmark: str,
      *,
      framework: mode_lib.Framework,
      device_kind: target_lib.DeviceKind,
      run_mode: common.RunMode,
  ) -> bool:
    """Returns whether this entry applies to the given execution coordinates."""
    if self.benchmark != benchmark or self.framework != framework:
      return False
    if self.device_kind is not None and self.device_kind != device_kind:
      return False
    if self.run_modes is not None and run_mode not in self.run_modes:
      return False
    return True


KNOWN_FAILURES: tuple[KnownFailure, ...] = (
    # TorchTPU.
    #
    # TODO(b/571252258): Remove once the model's on-device weight
    # initialization fits in HBM.
    KnownFailure(
        benchmark="qwen3_coder_30b_moe_forward",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
        reason=(
            "HBM OOM on v7_1x1x1 during on-device random weight initialization"
            " (peak ~1.6x the 57 GiB of bf16 weights). In eager_optimized it"
            " raises; in compiled and eager_default the executions failed"
            " silently and reported bogus metrics."
        ),
    ),
    # TODO(b/571695076): Remove once the 4096 parameter training step fits.
    KnownFailure(
        benchmark="synthetic_4096_params_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
        run_modes=frozenset({common.RunMode.EAGER_OPTIMIZED}),
        reason=(
            "the fused forward, backward and AdamW step needs more XLA"
            " parameters (20486) than supported (16314)"
        ),
    ),
    # The CPU smoke test (`--fake_tensor_mode`) builds these models under
    # FakeTensorMode, which cannot evaluate a Python scalar read from a fake
    # tensor (`DataDependentOutputException`).
    KnownFailure(
        benchmark="conformer_forward",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CPU,
        reason=(
            "int(lengths.max()) in _load_conformer reads a scalar from a fake"
            " tensor, which FakeTensorMode cannot evaluate"
        ),
    ),
    KnownFailure(
        benchmark="conformer_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CPU,
        reason=(
            "int(lengths.max()) in _load_conformer reads a scalar from a fake"
            " tensor, which FakeTensorMode cannot evaluate"
        ),
    ),
    KnownFailure(
        benchmark="timm_vit_small_dino_forward",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CPU,
        reason=(
            ".item() in timm calculate_drop_path_rates during model init reads"
            " a scalar from a fake tensor, which FakeTensorMode cannot evaluate"
        ),
    ),
    KnownFailure(
        benchmark="timm_vit_small_dino_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CPU,
        reason=(
            ".item() in timm calculate_drop_path_rates during model init reads"
            " a scalar from a fake tensor, which FakeTensorMode cannot evaluate"
        ),
    ),
    # TODO(b/568037427): Remove the TorchAX entries as their failures are fixed.
    #
    # `torch._grouped_mm`, used by the transformers MoE experts, has no
    # TorchAX lowering.
    KnownFailure(
        benchmark="gemma_4_26b_a4b_6_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "aten::_grouped_mm has no lowering (on v5e_1x1, HBM OOM while"
            " moving the model to the device)"
        ),
    ),
    KnownFailure(
        benchmark="gemma_4_26b_a4b_6_layers_training",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_120b_2_layers_training",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_120b_4_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_20b_4_layers_training",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_20b_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="nemotron_3_nano_30b_6_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="qwen3_coder_30b_moe_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="qwen3_5_397b_4_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    # The model forward needs a concrete value from a traced tensor (for a
    # shape or Python control flow), which fails under `jax.jit`.
    KnownFailure(
        benchmark="bart_large_mnli_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "torch.unique_consecutive() on a traced EOS count in"
            " modeling_bart.py"
        ),
    ),
    KnownFailure(
        benchmark="clap_htsat_fused_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason="torch.where() with a data-dependent size in modeling_clap.py",
    ),
    KnownFailure(
        benchmark="gemma_4_e4b_it_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "boolean indexing with a traced pooler_mask in modeling_gemma4.py"
        ),
    ),
    KnownFailure(
        benchmark="glm_ocr_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason="grid_thw.tolist() in transformers get_vision_position_ids()",
    ),
    KnownFailure(
        benchmark="janus_pro_1b_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "aten::_is_all_true in torch_compilable_check on a traced image"
            " token count in modeling_janus.py"
        ),
    ),
    KnownFailure(
        benchmark="lahja_sa_ahmad_v1_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason="torch.arange(predicted_lengths.max()) in modeling_vits.py",
    ),
    KnownFailure(
        benchmark="llava_1_5_7b_hf_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "torch_compilable_check and masked_scatter on a traced image mask"
            " in modeling_llava.py"
        ),
    ),
    KnownFailure(
        benchmark="sam3_litetext_s0_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "torch.arange() with a traced height in modeling_sam3_lite_text.py"
        ),
    ),
    KnownFailure(
        benchmark="vjepa2_vitg_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "the skip_predictor=True input becomes a traced bool under jax.jit,"
            " and `if not skip_predictor:` in modeling_vjepa2.py needs a"
            " concrete value"
        ),
    ),
    KnownFailure(
        benchmark="vjepa2_vitl_fpc64_256_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "Tensor.repeat() with a traced patch count in modeling_vjepa2.py"
        ),
    ),
    # Out of memory.
    KnownFailure(
        benchmark="gemma_4_31b_6_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="host OOM while moving the model to the device",
    ),
    # TODO(b/327271919): Remove once Inductor CPU codegen works on GPU hosts.
    #
    # The model forward creates a tensor without `device=`, which lands on CPU.
    # Inductor then emits a CPU C++ kernel for it, and the GPU benchmark hosts
    # have no C++ compiler (`InvalidCxxCompiler`).
    KnownFailure(
        benchmark="blip_vqa_capfilt_large_forward_gen",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CUDA,
        run_modes=frozenset({common.RunMode.COMPILED}),
        reason=(
            "torch.ones() without device= for the image attention mask in"
            " modeling_blip.py needs a C++ compiler for Inductor CPU codegen"
        ),
    ),
    KnownFailure(
        benchmark="deberta_base_mnli_forward_gen",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CUDA,
        run_modes=frozenset({common.RunMode.COMPILED}),
        reason=(
            "torch.tensor() without device= in scaled_size_sqrt() in"
            " modeling_deberta.py needs a C++ compiler for Inductor CPU codegen"
        ),
    ),
    # TODO(b/571695076): Remove once the 4096 parameter training step compiles
    # within the subprocess timeout.
    KnownFailure(
        benchmark="synthetic_4096_params_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CUDA,
        run_modes=frozenset({common.RunMode.COMPILED}),
        reason=(
            "Inductor compilation of the training step exceeds the 900 s"
            " subprocess timeout on b200_1"
        ),
    ),
    # TODO(b/563472806): Remove once internal builds pick up
    # https://github.com/llvm/llvm-project/pull/225808.
    #
    # The LLVM of internal builds expands a vectorized integer division by a
    # constant into a v2i16 smul_lohi or umul_lohi that its NVPTX backend cannot
    # select, so Triton aborts while compiling an Inductor kernel of these
    # models. The Triton wheel used in OSS is not affected.
    # copybara:uncomment_begin(internal builds only)
    # KnownFailure(
        # benchmark="conformer_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="dpt_hybrid_midas_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="gepard_1_0_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="gepard_1_0_train_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="glm_ocr_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="gpt2_train_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="lahja_sa_ahmad_v1_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="sam3_litetext_s0_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="timm_resnet_50_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="tiny_doc_qa_vision_encoder_decoder_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="wan_2_2_ti2v_5b_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # copybara:uncomment_end
)


def get_known_failure_reason(
    benchmark: str,
    *,
    framework: mode_lib.Framework,
    device_kind: target_lib.DeviceKind,
    run_mode: common.RunMode,
) -> str | None:
  """Returns the failure reason if the benchmark is known to fail, else None."""
  for failure in KNOWN_FAILURES:
    if failure.matches(
        benchmark,
        framework=framework,
        device_kind=device_kind,
        run_mode=run_mode,
    ):
      return failure.reason
  return None
