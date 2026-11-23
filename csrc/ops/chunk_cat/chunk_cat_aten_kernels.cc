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

#include "csrc/ops/chunk_cat/chunk_cat_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/TensorShape.h"
#include "absl/types/span.h"
#include "c10/util/accumulate.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

at::Tensor& AtenChunkCatOut(at::TensorList tensors, int64_t dim,
                            int64_t num_chunks, at::Tensor& out) {
  TT_KERNEL(OpName::kChunkCatOut, param_keys, (tensors, dim, num_chunks, out), {
    // PyTorch's own checks, so that errors match the CPU.
    const int64_t wrapped_dim =
        at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
    TT_THROW_IF_ERROR(ValidateOutDtype(out, tensors[0].scalar_type()));
    // FSDP2's reduce-scatter copy-in calls _chunk_cat with dim 0 and
    // out = flat_buffer.view(num_chunks, -1). The default decomposition
    // reshapes every input to (num_chunks, -1) and concatenates along dim
    // 1, and XLA takes minutes to compile that for a vocabulary-sized
    // gradient. This kernel builds out as one flat array instead, with
    // slices and a concatenation, so that neither the compile time nor
    // the host work grows with num_chunks * tensors.size().
    //
    // Each input is viewed as (num_rows, -1), num_rows being the product
    // of the dims before wrapped_dim (the same for all inputs). Zero-
    // padding wrapped_dim to a multiple of num_chunks then appends zeros
    // to every row, and out's row is the concatenation, over the chunks
    // and then the inputs, of each input's chunk.
    const int64_t num_rows =
        c10::multiply_integers(tensors[0].sizes().slice(0, wrapped_dim));
    Indices chunk_numels;
    chunk_numels.reserve(tensors.size());
    int64_t chunk_row_numel = 0;
    for (const at::Tensor& tensor : tensors) {
      const int64_t size = tensor.size(wrapped_dim);
      const int64_t inner_numel = tensor.numel() / (num_rows * size);
      chunk_numels.push_back((size + num_chunks - 1) / num_chunks *
                             inner_numel);
      chunk_row_numel += chunk_numels.back();
    }
    Dimensions out_shape =
        CopyIntVector(tensors[0].sizes().slice(0, wrapped_dim));
    out_shape.push_back(num_chunks);
    out_shape.push_back(chunk_row_numel);
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, out_shape));
    // When out covers its whole storage contiguously (FSDP2's flat
    // buffer), the flat result becomes its storage directly. Otherwise
    // the result takes out's shape, so that it can be written through
    // out's view.
    const int64_t out_numel = c10::multiply_integers(out_shape);
    const bool flat_result = out.is_contiguous() && out.storage_offset() == 0 &&
                             out.storage().nbytes() == out.nbytes();
    Dimensions result_dims = flat_result ? Dimensions{out_numel} : out_shape;
    TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    auto op_builder = [num_rows, num_chunks, chunk_numels, result_dims](
                          absl::Span<mlir::MlirOp> inputs,
                          mlir::MlirBuilder& builder) {
      // A single row (dim 0, as FSDP2 calls it) is kept 1-D: a (1, N) array
      // makes the copy slower on TPU.
      const int64_t axis = num_rows == 1 ? 0 : 1;
      const auto row_shape = [num_rows](int64_t row_numel) {
        return num_rows == 1 ? Dimensions{row_numel}
                             : Dimensions{num_rows, row_numel};
      };
      const Indices stride(axis + 1, 1);
      const Dimensions no_padding(axis + 1, 0);
      std::vector<mlir::MlirOp> rows;
      rows.reserve(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        const mlir::RankedTensorType type = GetTensorTypeOrDie(inputs[i]);
        const int64_t row_numel = type.getNumElements() / num_rows;
        mlir::MlirOp row =
            mlir::stablehlo::Reshape(inputs[i], row_shape(row_numel));
        Dimensions padding = no_padding;
        padding[axis] = num_chunks * chunk_numels[i] - row_numel;
        if (padding[axis] > 0) {
          mlir::MlirOp zero =
              MakeScalarConstant(builder, 0, type.getElementType());
          row = mlir::stablehlo::Pad(row, zero, /*edge_padding_low=*/no_padding,
                                     /*edge_padding_high=*/padding,
                                     /*interior_padding=*/no_padding);
        }
        rows.push_back(row);
      }
      std::vector<mlir::MlirOp> chunks;
      chunks.reserve(num_chunks * inputs.size());
      for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
        for (size_t i = 0; i < inputs.size(); ++i) {
          Indices begin(axis + 1, 0);
          begin[axis] = chunk * chunk_numels[i];
          const Indices limit = row_shape((chunk + 1) * chunk_numels[i]);
          chunks.push_back(
              mlir::stablehlo::Slice(rows[i], begin, limit, stride));
        }
      }
      mlir::MlirOp result = mlir::stablehlo::Concatenate(builder, chunks, axis);
      return mlir::stablehlo::Reshape(result, result_dims);
    };
    // DispatchOp rather than DispatchOpOut: no input aliases out, so there is
    // nothing to donate, and DispatchOpOut would resize out to the result's
    // shape. out keeps its shape [leading..., num_chunks, chunk_row_numel]
    // (resized above), while the result is 1-D {out_numel} when out covers
    // its storage contiguously. Both describe the same bytes, so
    // AssignBufferToAtTensor adopts the 1-D buffer as out's storage without
    // changing out's shape, and the reduce-scatter that reads that storage as
    // 1-D needs no relayout. The inputs keep their dtype until the result is
    // cast to out's dtype, so the slices and concatenation move fewer bytes.
    TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                       DispatchOp<kDynamicSize>(
                           std::move(op_builder),
                           tensors.vec(),  // VEC_OK
                           {.out_dtype = out_dtype,
                            .out_dims = result_dims,
                            .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
