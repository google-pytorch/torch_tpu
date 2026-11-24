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
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/accumulate.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

at::Tensor& AtenChunkCatOut(at::TensorList tensors, int64_t dim,
                            int64_t num_chunks, at::Tensor& out) {
  TT_KERNEL(
      OpName::kChunkCatOut, param_keys,
      (tensors, IgnoreInCacheKey(dim, "keyed as wrapped_dim"), num_chunks, out),
      {
        // PyTorch's own checks, so that errors match the CPU.
        const int64_t wrapped_dim =
            at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
        TT_THROW_IF_ERROR(param_keys.SetParam("wrapped_dim", wrapped_dim));
        // preprocess_chunk_cat_inputs checks the inputs' devices, not out's.
        TT_CHECK_THROW(out.device() == tensors[0].device(),
                       error::kInvalidArgument)
            << "expected out to be on " << tensors[0].device().str() << ", got "
            << out.device().str();
        TT_THROW_IF_ERROR(ValidateOutDtype(out, tensors[0].scalar_type()));
        // TPU stores float4_e2m1fn_x2 with its last dimension unpacked, which
        // the element counts below do not account for.
        for (const at::ScalarType dtype :
             {tensors[0].scalar_type(), out.scalar_type()}) {
          TT_CHECK_THROW(dtype != at::kFloat4_e2m1fn_x2,
                         error::kPythonNotImplementedError)
              << "not implemented for " << ToString(dtype);
        }
        // The program below is built for static shapes. The base buffer
        // carries a marked tensor's dynamic dimensions without recording a
        // view op; inputs whose dynamic dimensions are only known when the
        // program is built (deferred ops) are checked in op_builder.
        for (const at::Tensor& tensor : tensors) {
          TT_ASSIGN_OR_THROW(const DeviceBufferRef buffer,
                             GetBaseBuffer(tensor));
          TT_CHECK_THROW(buffer.dynamic_dimensions().empty() &&
                             !buffer.on_device_shape_is_dynamic(),
                         error::kPythonNotImplementedError)
              << "not implemented for inputs with dynamic dimensions";
        }
        // FSDP2's reduce-scatter copy-in calls _chunk_cat with dim 0 and
        // out = flat_buffer.view(num_chunks, -1). The default decomposition
        // reshapes every input to (num_chunks, -1) and concatenates along dim
        // 1, and XLA takes minutes to compile that for a vocabulary-sized
        // gradient. This kernel builds out as one flat array instead, with
        // slices and a concatenation. It compiles in under a second, and its
        // host work does not grow with num_chunks * tensors.size() (the
        // program's size does).
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
          // ceil(size / num_chunks) rows, without overflowing for a huge
          // num_chunks
          TT_ASSIGN_OR_THROW(
              const int64_t chunk_numel,
              SafeMultiply((size - 1) / num_chunks + 1, inner_numel));
          chunk_numels.push_back(chunk_numel);
          TT_ASSIGN_OR_THROW(chunk_row_numel,
                             SafeAdd(chunk_row_numel, chunk_numel));
        }
        Dimensions out_shape =
            CopyIntVector(tensors[0].sizes().slice(0, wrapped_dim));
        out_shape.push_back(num_chunks);
        out_shape.push_back(chunk_row_numel);
        // When out already has out_shape and covers its whole storage
        // contiguously (FSDP2's flat buffer), the flat result becomes its
        // storage directly. Otherwise the result takes out's shape, so that it
        // can be written through out's view.
        TT_ASSIGN_OR_THROW(const int64_t out_numel, NumElements(out_shape));
        const bool flat_result =
            out.sizes() ==
                c10::IntArrayRef(out_shape.data(), out_shape.size()) &&
            out.is_contiguous() && !out.is_conj() &&
            out.storage_offset() == 0 && out.storage().nbytes() == out.nbytes();
        const Dimensions result_dims =
            flat_result ? Dimensions{out_numel} : out_shape;
        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_THROW_IF_ERROR(
            ValidateTensorByteSize(
                c10::IntArrayRef(out_shape.data(), out_shape.size()), out_dtype)
                .status());
        // XLA rejects a direct real-to-complex convert, which the dispatcher's
        // cast to out_dtype would emit.
        const bool real_to_complex =
            at::isComplexType(out.scalar_type()) &&
            !at::isComplexType(tensors[0].scalar_type());
        auto op_builder =
            [num_rows, num_chunks, chunk_numels = std::move(chunk_numels),
             result_dims, out_dtype, real_to_complex](
                absl::Span<mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
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
            TT_RET_CHECK(type.hasStaticShape(),
                         error::kPythonNotImplementedError)
                << "not implemented for inputs with dynamic dimensions";
            const int64_t row_numel = type.getNumElements() / num_rows;
            mlir::MlirOp row =
                mlir::stablehlo::Reshape(inputs[i], row_shape(row_numel));
            Dimensions padding = no_padding;
            padding[axis] = num_chunks * chunk_numels[i] - row_numel;
            if (padding[axis] > 0) {
              mlir::MlirOp zero =
                  MakeScalarConstant(builder, 0, type.getElementType());
              row = mlir::stablehlo::Pad(row, zero,
                                         /*edge_padding_low=*/no_padding,
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
          mlir::MlirOp result =
              mlir::stablehlo::Concatenate(builder, chunks, axis);
          result = mlir::stablehlo::Reshape(result, result_dims);
          if (real_to_complex) {
            TT_ASSIGN_OR_RETURN(
                result, ComplexSafeConvertElementType(result, out_dtype));
          }
          return result;
        };
        // DispatchOp rather than DispatchOpOut, which would resize out to the
        // result's shape. out keeps its shape [leading..., num_chunks,
        // chunk_row_numel], while the result is 1-D {out_numel} when out covers
        // its storage contiguously. Both describe the same bytes, so
        // AssignBufferToAtTensor adopts the 1-D buffer as out's storage without
        // changing out's shape, and the reduce-scatter that reads that storage
        // as 1-D needs no relayout. The inputs keep their dtype until the
        // result is cast to out's dtype, so the slices and concatenation move
        // fewer bytes.
        TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                           DispatchOp<kDynamicSize>(
                               std::move(op_builder),
                               tensors.vec(),  // VEC_OK
                               {.out_dtype = out_dtype,
                                .out_dims = result_dims,
                                .op_param_cache_keys = std::move(param_keys)}));
        // Resized only now, so that an input aliasing out is read before out
        // changes, and a failed dispatch leaves out untouched.
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, out_shape));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
