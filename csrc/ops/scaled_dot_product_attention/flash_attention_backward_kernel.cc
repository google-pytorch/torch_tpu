/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <optional>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_kernel.h"
#include "csrc/ops/scaled_dot_product_attention/util.h"

namespace torch_tpu {

namespace {

// The (qt, kt) intermediates shared by the dQ and dKV kernels for one
// (q block, kv block) pair.
struct SharedGrads {
  // Softmax probabilities with dropout applied, f32. Feeds dV.
  kiln::Tile p;
  // dS = P * (dP - di) in the input dtype. Feeds dQ and dK.
  kiln::Tile ds;
};

// Recomputes P from the forward statistics and derives dS:
//   s  = logits(q, k)                (log2-scaled, see ComputeLogits)
//   p  = exp2(s - lse)
//   dp = dO @ V^T
//   ds = p * (dp - di)
// `lse_ref` / `di_ref` hold one (1, 1, 1, qt) row per q block.
SharedGrads ComputeSharedGrads(
    kiln::Tile q, kiln::Tile k, kiln::Tile v, kiln::Tile do_,
    kiln::VmemRef lse_ref, kiln::VmemRef di_ref,
    std::optional<kiln::VmemRef> bias_ref,
    std::optional<kiln::SmemRef> seed_ref, kiln::Scalar batch,
    std::optional<kiln::Scalar> q_head, kiln::Scalar qi, kiln::Scalar ki,
    const FlashAttnConfig& config, const Tiling& tiling, kiln::DType dt) {
  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;
  // The statistics are stored as lane-major rows; turn them into per-row
  // columns so they broadcast along kt.
  kiln::Tile lse = kiln::Load(lse_ref(0, 0)).Reshape({qt, 1});
  kiln::Tile di = kiln::Load(di_ref(0, 0)).Reshape({qt, 1});

  kiln::Tile s = ComputeLogits(q, k, bias_ref, qi, ki, config, tiling);
  kiln::Tile lse_bcast = lse.BroadcastTo({qt, kt});
  kiln::Tile is_lse_ninf = lse_bcast == kiln::NInf(lse_bcast.shape());
  kiln::Tile safe_lse = Select(is_lse_ninf, 0.0f, lse_bcast);
  kiln::Tile p = Select(is_lse_ninf, 0.0f, Exp2(s - safe_lse));

  kiln::Tile dp =
      kiln::Dot(do_, v, {.transpose_rhs = true, .precision = config.precision});

  kiln::Tile p_dv = p;
  if (seed_ref) {
    kiln::Tile keep =
        DropoutKeepMask(config, tiling, batch, *q_head, qi, ki, *seed_ref);
    p_dv = ApplyDropout(p, keep, config);
    dp = ApplyDropout(dp, keep, config);
  }

  kiln::Tile ds = (p * (dp - di)).AsType(dt);
  return {p_dv, ds};
}

kiln::IndexIndices Block012IndexMap() {
  return {kiln::ProgramId(0), kiln::ProgramId(1), kiln::ProgramId(2), 0};
}

kiln::IndexIndices AuxBlockIndexMap() {
  return {kiln::ProgramId(0), kiln::ProgramId(1), 0, kiln::ProgramId(2)};
}

kiln::Scalar DkvQHead(int64_t head_group) {
  return kiln::ProgramId(1) * head_group + kiln::ProgramId(4);
}

void EmitDkvStep(const FlashAttnConfig& config, const Tiling& tiling,
                 std::optional<kiln::SmemRef> seed_ref, kiln::VmemRef q_ref,
                 kiln::VmemRef k_ref, kiln::VmemRef v_ref, kiln::VmemRef do_ref,
                 kiln::VmemRef lse_ref, kiln::VmemRef di_ref,
                 std::optional<kiln::VmemRef> bias_ref, kiln::VmemRef dk_ref,
                 kiln::VmemRef dv_ref, kiln::VmemRef dk_acc,
                 kiln::VmemRef dv_acc, int64_t head_group, kiln::Scalar qi,
                 kiln::Scalar ki) {
  kiln::DType dt = config.element_type;
  std::optional<kiln::ContractPrecision> precision = config.precision;

  kiln::Tile q = kiln::Load(q_ref(0, 0));
  kiln::Tile k = kiln::Load(k_ref(0, 0));
  kiln::Tile v = kiln::Load(v_ref(0, 0));
  kiln::Tile do_ = kiln::Load(do_ref(0, 0));

  std::optional<kiln::Scalar> q_head;
  if (seed_ref) {
    q_head = DkvQHead(head_group);
  }
  SharedGrads g = ComputeSharedGrads(q, k, v, do_, lse_ref, di_ref, bias_ref,
                                     seed_ref, kiln::ProgramId(0), q_head, qi,
                                     ki, config, tiling, dt);

  // dV += P^T @ dO ; dK += (dS^T @ Q) * scale
  kiln::Tile dv_step = kiln::Dot(
      g.p.AsType(dt), do_, {.transpose_lhs = true, .precision = precision});
  kiln::Tile dk_step =
      kiln::Dot(g.ds, q, {.transpose_lhs = true, .precision = precision}) *
      config.scale;
  kiln::Tile dk = kiln::Load(dk_acc) + dk_step;
  kiln::Store(dk_acc, dk);
  kiln::Tile dv = kiln::Load(dv_acc) + dv_step;
  kiln::Store(dv_acc, dv);
  kiln::Store(dk_ref(0, 0), dk);
  kiln::Store(dv_ref(0, 0), dv);
}

void EmitDqStep(const FlashAttnConfig& config, const Tiling& tiling,
                std::optional<kiln::SmemRef> seed_ref, kiln::VmemRef q_ref,
                kiln::VmemRef k_ref, kiln::VmemRef v_ref, kiln::VmemRef do_ref,
                kiln::VmemRef lse_ref, kiln::VmemRef di_ref,
                std::optional<kiln::VmemRef> bias_ref, kiln::VmemRef dq_ref,
                kiln::VmemRef dq_acc, kiln::Scalar qi, kiln::Scalar ki) {
  kiln::DType dt = config.element_type;
  std::optional<kiln::ContractPrecision> precision = config.precision;

  kiln::Tile q = kiln::Load(q_ref(0, 0));
  kiln::Tile k = kiln::Load(k_ref(0, 0));
  kiln::Tile v = kiln::Load(v_ref(0, 0));
  kiln::Tile do_ = kiln::Load(do_ref(0, 0));

  SharedGrads g = ComputeSharedGrads(
      q, k, v, do_, lse_ref, di_ref, bias_ref, seed_ref, kiln::ProgramId(0),
      kiln::ProgramId(1), qi, ki, config, tiling, dt);

  // dQ += (dS @ K) * scale
  kiln::Tile dq_step =
      kiln::Dot(g.ds, k, {.precision = precision}) * config.scale;
  kiln::Tile dq = kiln::Load(dq_acc) + dq_step;
  kiln::Store(dq_acc, dq);
  kiln::Store(dq_ref(0, 0), dq);
}

}  // namespace

absl::StatusOr<kiln::Kernel> CreateBackwardDkvKernel(
    const FlashAttnConfig& config, const Tiling& tiling) {
  TT_RETURN_IF_ERROR(ValidateFlashAttnConfig(config, tiling));

  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;
  const int64_t d_qk = config.qk_head_dim;
  const int64_t d_vo = config.vo_head_dim;
  const int64_t num_q_blocks = config.padded_q_sequence_length / qt;
  const int64_t num_kv_blocks = config.padded_kv_sequence_length / kt;
  const int64_t head_group = config.num_heads / config.kv_num_heads;
  kiln::DType dt = config.element_type;

  // ---------------------------------------------------------------------------
  // Grid & block specs.
  // ---------------------------------------------------------------------------
  // Grid: (batch, kv_head, kv_block, q_block, head_in_group). The last two are
  // reduction axes: every q block, and every query head sharing this kv head
  // (GQA / MQA), accumulates into the same dK / dV block.
  kiln::GridSpec grid{
      .name = "flash_backward_dkv",
      .grid = {config.batch_size, config.kv_num_heads, num_kv_blocks,
               num_q_blocks, head_group},
      .dimension_semantics = {kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kArbitrary,
                              kiln::DimensionSemantics::kArbitrary},
  };

  kiln::IndexMap kv_map(Block012IndexMap);
  kiln::IndexMap q_map = kiln::IndexMap([head_group]() -> kiln::IndexIndices {
    return {kiln::ProgramId(0), DkvQHead(head_group), kiln::ProgramId(3), 0};
  });
  kiln::IndexMap aux_map = kiln::IndexMap([head_group]() -> kiln::IndexIndices {
    return {kiln::ProgramId(0), DkvQHead(head_group), 0, kiln::ProgramId(3)};
  });

  std::optional<kiln::InSpec> bias_spec;
  if (config.has_attn_bias) {
    kiln::IndexMap bias_map = kiln::IndexMap(
        [head_group,
         dims = config.mask_broadcast_dims]() -> kiln::IndexIndices {
          kiln::IndexIndices idx{kiln::ProgramId(0), DkvQHead(head_group),
                                 kiln::ProgramId(3), kiln::ProgramId(2)};
          for (int64_t d : dims) idx[d] = 0;
          return idx;
        });
    bias_spec = kiln::InSpec{BiasBlockShape(config, tiling, dt), bias_map};
  }

  // ---------------------------------------------------------------------------
  // Kernel body.
  // ---------------------------------------------------------------------------
  TT_ASSIGN_OR_RETURN(
      kiln::Kernel kernel,
      kiln::CreateKernel(
          grid, DropoutSeedSpec(config),
          kiln::InSpec{dt(1, 1, qt, d_qk), q_map},        // q
          kiln::InSpec{dt(1, 1, kt, d_qk), kv_map},       // k
          kiln::InSpec{dt(1, 1, kt, d_vo), kv_map},       // v
          kiln::InSpec{dt(1, 1, qt, d_vo), q_map},        // dO
          kiln::InSpec{kiln::f32(1, 1, 1, qt), aux_map},  // lse
          kiln::InSpec{kiln::f32(1, 1, 1, qt), aux_map},  // di
          bias_spec,                                      //
          kiln::OutSpec{dt(1, 1, kt, d_qk), kv_map},      // dK
          kiln::OutSpec{dt(1, 1, kt, d_vo), kv_map},      // dV
          kiln::ScratchSpec{kiln::f32(kt, d_qk)},         // dK accumulator
          kiln::ScratchSpec{kiln::f32(kt, d_vo)},         // dV accumulator
          [&]  // REF_CAPTURE_OK=called synchronously
          (std::optional<kiln::SmemRef> seed_ref, kiln::VmemRef q_ref,
           kiln::VmemRef k_ref, kiln::VmemRef v_ref, kiln::VmemRef do_ref,
           kiln::VmemRef lse_ref, kiln::VmemRef di_ref,
           std::optional<kiln::VmemRef> bias_ref, kiln::VmemRef dk_ref,
           kiln::VmemRef dv_ref, kiln::VmemRef dk_acc, kiln::VmemRef dv_acc) {
            kiln::Scalar ki = kiln::ProgramId(2), qi = kiln::ProgramId(3);
            kiln::Scalar head_in_group = kiln::ProgramId(4);

            kiln::If((qi == 0) & (head_in_group == 0),
                     [&] {  // REF_CAPTURE_OK=called synchronously
                       kiln::Store(dk_acc, kiln::Zeros(kiln::f32(kt, d_qk)));
                       kiln::Store(dv_acc, kiln::Zeros(kiln::f32(kt, d_vo)));
                     });

            if (config.is_causal) {
              kiln::If(BelowOrOnDiagonal(qi, ki, tiling),
                       [&] {  // REF_CAPTURE_OK=called synchronously
                         EmitDkvStep(config, tiling, seed_ref, q_ref, k_ref,
                                     v_ref, do_ref, lse_ref, di_ref, bias_ref,
                                     dk_ref, dv_ref, dk_acc, dv_acc, head_group,
                                     qi, ki);
                       });
            } else {
              EmitDkvStep(config, tiling, seed_ref, q_ref, k_ref, v_ref, do_ref,
                          lse_ref, di_ref, bias_ref, dk_ref, dv_ref, dk_acc,
                          dv_acc, head_group, qi, ki);
            }
          }));

  ABSL_VLOG(1) << "Backward dKV kernel:\n" << kiln::ToString(kernel);
  return kernel;
}

absl::StatusOr<kiln::Kernel> CreateBackwardDqKernel(
    const FlashAttnConfig& config, const Tiling& tiling) {
  TT_RETURN_IF_ERROR(ValidateFlashAttnConfig(config, tiling));

  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;
  const int64_t d_qk = config.qk_head_dim;
  const int64_t d_vo = config.vo_head_dim;
  const int64_t num_q_blocks = config.padded_q_sequence_length / qt;
  const int64_t num_kv_blocks = config.padded_kv_sequence_length / kt;
  kiln::DType dt = config.element_type;

  // ---------------------------------------------------------------------------
  // Grid & block specs.
  // ---------------------------------------------------------------------------
  // Same grid as the forward kernel; the kv axis is the reduction axis.
  kiln::GridSpec grid{
      .name = "flash_backward_dq",
      .grid = {config.batch_size, config.num_heads, num_q_blocks,
               num_kv_blocks},
      .dimension_semantics = {kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kArbitrary},
  };

  kiln::IndexMap q_map(Block012IndexMap);
  kiln::IndexMap aux_map(AuxBlockIndexMap);
  kiln::IndexMap kv_map = KvBlockIndexMap(config, tiling);

  // ---------------------------------------------------------------------------
  // Kernel body.
  // ---------------------------------------------------------------------------
  TT_ASSIGN_OR_RETURN(
      kiln::Kernel kernel,
      kiln::CreateKernel(
          grid, DropoutSeedSpec(config),
          kiln::InSpec{dt(1, 1, qt, d_qk), q_map},        // q
          kiln::InSpec{dt(1, 1, kt, d_qk), kv_map},       // k
          kiln::InSpec{dt(1, 1, kt, d_vo), kv_map},       // v
          kiln::InSpec{dt(1, 1, qt, d_vo), q_map},        // dO
          kiln::InSpec{kiln::f32(1, 1, 1, qt), aux_map},  // lse
          kiln::InSpec{kiln::f32(1, 1, 1, qt), aux_map},  // di
          AttnBiasSpec(config, tiling, dt),               //
          // The forward output is part of the calling convention (see
          // CreateFlashAttentionBackwardKernel) but is not needed here: di is
          // computed outside the kernel.
          kiln::InSpec{dt(1, 1, qt, d_vo), q_map},   // out (unused)
          kiln::OutSpec{dt(1, 1, qt, d_qk), q_map},  // dQ
          kiln::ScratchSpec{kiln::f32(qt, d_qk)},    // dQ accumulator
          [&]  // REF_CAPTURE_OK=called synchronously
          (std::optional<kiln::SmemRef> seed_ref, kiln::VmemRef q_ref,
           kiln::VmemRef k_ref, kiln::VmemRef v_ref, kiln::VmemRef do_ref,
           kiln::VmemRef lse_ref, kiln::VmemRef di_ref,
           std::optional<kiln::VmemRef> bias_ref, kiln::VmemRef /*out_ref*/,
           kiln::VmemRef dq_ref, kiln::VmemRef dq_acc) {
            kiln::Scalar qi = kiln::ProgramId(2), ki = kiln::ProgramId(3);

            kiln::If(ki == 0,
                     [&] {  // REF_CAPTURE_OK=called synchronously
                       kiln::Store(dq_acc, kiln::Zeros(kiln::f32(qt, d_qk)));
                     });

            if (config.is_causal) {
              kiln::If(BelowOrOnDiagonal(qi, ki, tiling),
                       [&] {  // REF_CAPTURE_OK=called synchronously
                         EmitDqStep(config, tiling, seed_ref, q_ref, k_ref,
                                    v_ref, do_ref, lse_ref, di_ref, bias_ref,
                                    dq_ref, dq_acc, qi, ki);
                       });
            } else {
              EmitDqStep(config, tiling, seed_ref, q_ref, k_ref, v_ref, do_ref,
                         lse_ref, di_ref, bias_ref, dq_ref, dq_acc, qi, ki);
            }
          }));

  ABSL_VLOG(1) << "Backward dQ kernel:\n" << kiln::ToString(kernel);
  return kernel;
}

}  // namespace torch_tpu
