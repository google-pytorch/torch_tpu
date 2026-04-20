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

#include "torch_tpu/ops/ctc_loss/ctc_loss_aten_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/tensor.h"
#include "ATen/ops/zeros.h"
#include "c10/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/ctc_loss/ctc_loss.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor> AtenCtcLoss(const at::Tensor& log_probs,
                                               const at::Tensor& targets,
                                               at::IntArrayRef input_lengths,
                                               at::IntArrayRef target_lengths,
                                               int64_t blank,
                                               bool zero_infinity) {
  TT_KERNEL(OpName::kCtcLoss, _,
            (log_probs, targets,
             IgnoreInCacheKey(input_lengths, "delegates to AtenCtcLossTensor"),
             IgnoreInCacheKey(target_lengths, "delegates to AtenCtcLossTensor"),
             IgnoreInCacheKey(blank, "delegates to AtenCtcLossTensor"),
             IgnoreInCacheKey(zero_infinity, "delegates to AtenCtcLossTensor")),
            {
              at::Tensor input_lengths_tensor =
                  at::tensor(input_lengths, at::kLong).to(log_probs.device());
              at::Tensor target_lengths_tensor =
                  at::tensor(target_lengths, at::kLong).to(log_probs.device());
              return AtenCtcLossTensor(log_probs, targets, input_lengths_tensor,
                                       target_lengths_tensor, blank,
                                       zero_infinity);
            });
}

std::tuple<at::Tensor, at::Tensor> AtenCtcLossTensor(
    const at::Tensor& log_probs, const at::Tensor& targets,
    const at::Tensor& input_lengths, const at::Tensor& target_lengths,
    int64_t blank, bool zero_infinity) {
  TT_KERNEL(
      OpName::kCtcLossTensor, param_keys,
      (log_probs, targets, input_lengths, target_lengths, blank, zero_infinity),
      {
        TT_CHECK_THROW(log_probs.dim() == 3, error::kInvalidArgument)
            << "expected log_probs to be 3-D, got " << log_probs.dim() << "-D";
        TT_CHECK_THROW(targets.dim() == 1 || targets.dim() == 2,
                       error::kInvalidArgument)
            << "expected targets to be 1-D or 2-D, got " << targets.dim()
            << "-D";

        const int64_t batch_size = log_probs.size(1);
        TT_CHECK_THROW(input_lengths.numel() == batch_size,
                       error::kInvalidArgument)
            << "expected input_lengths to have batch_size (" << batch_size
            << ") elements, got " << input_lengths.numel();
        TT_CHECK_THROW(target_lengths.numel() == batch_size,
                       error::kInvalidArgument)
            << "expected target_lengths to have batch_size (" << batch_size
            << ") elements, got " << target_lengths.numel();

        const int64_t N = log_probs.size(1);
        const int64_t T = log_probs.size(0);

        at::Tensor padded_targets = targets;
        if (targets.dim() == 1) {
          int64_t max_target_length = 0;
          at::Tensor target_lengths_cpu =
              target_lengths.to(at::kLong).cpu().contiguous();
          const int64_t* lengths_data =
              static_cast<const int64_t*>(target_lengths_cpu.const_data_ptr());

          for (int64_t i = 0; i < N; ++i) {
            max_target_length = std::max(max_target_length, lengths_data[i]);
          }

          padded_targets = at::zeros({N, max_target_length}, targets.options());
          int64_t offset = 0;
          for (int64_t i = 0; i < N; ++i) {
            int64_t len = lengths_data[i];
            if (len > 0) {
              padded_targets.select(0, i).narrow(0, 0, len).copy_(
                  targets.narrow(0, offset, len));
            }
            offset += len;
          }
        }

        const int64_t S = padded_targets.size(1);
        const int64_t L = 2 * S + 1;

        TT_ASSIGN_OR_THROW(
            mlir::ElementType output_dtype,
            ConvertTo<mlir::ElementType>(log_probs.scalar_type()));
        const std::array<mlir::ElementType, 2> out_dtypes = {output_dtype,
                                                             output_dtype};

        const Dimensions loss_dims = {N};
        const Dimensions log_alpha_dims = {N, T, L};

        const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
            absl::MakeConstSpan(loss_dims),
            absl::MakeConstSpan(log_alpha_dims)};

        DispatchOpOptions<2> options = {
            .out_dtypes = out_dtypes,
            .out_dims_list = out_dims_list,
            .op_param_cache_keys = std::move(param_keys),
        };

        auto op_builder = [blank,
                           zero_infinity](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<MlirOpResults<2>> {
          auto& [log_probs_op, targets_op, input_lengths_op,
                 target_lengths_op] = inputs;
          auto& builder = log_probs_op.getBuilder();
          TT_ASSIGN_OR_RETURN(
              auto results,
              BuildCtcLossShlo(
                  log_probs_op, targets_op, input_lengths_op, target_lengths_op,
                  blank, zero_infinity ? ZeroInfinity::kYes : ZeroInfinity::kNo,
                  builder));
          return MlirOpResults<2>{results[0], results[1]};
        };

        TT_ASSIGN_OR_THROW(auto output_bufs,
                           (DispatchOp<4, 2>(std::move(op_builder),
                                             {log_probs, padded_targets,
                                              input_lengths, target_lengths},
                                             std::move(options))));

        at::Tensor loss = at::empty(loss_dims, log_probs.options());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_bufs[0]), loss));

        at::Tensor log_alpha = at::empty(log_alpha_dims, log_probs.options());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(output_bufs[1]), log_alpha));

        return std::make_tuple(loss, log_alpha);
      });
}

}  // namespace torch_tpu
