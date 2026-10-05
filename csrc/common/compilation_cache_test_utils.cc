/*
 * Copyright 2025 Google LLC
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

#include "csrc/common/compilation_cache_test_utils.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

bool IsCompatible(absl::Span<const Shape> shapes) {
  return shapes.size() == 1 && shapes[0].dimensions().size() == 2;
}

}  // namespace

UnboundedDynamicExecutionMetadata MakeDummyDynamicMetadata() {
  UnboundedDynamicExecutionMetadata dynamic_meta;
  dynamic_meta.is_compatible = IsCompatible;
  return dynamic_meta;
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CreateDummyExecutable(xla::PjRtClient& client,
                      UniqueCompileOptions compile_options) {
  xla::XlaBuilder builder("dynamic_metadata_test");
  xla::Shape data_shape =
      xla::ShapeUtil::MakeShape(xla::F32, /*dimensions=*/{1});
  xla::LayoutUtil::SetToDefaultLayout(&data_shape);

  xla::XlaOp p0 = xla::Parameter(&builder, 0, data_shape, "input_0");

  TT_ASSIGN_OR_RETURN(xla::XlaComputation computation, builder.Build(p0));
  return client.CompileAndLoad(computation, std::move(*compile_options));
}

}  // namespace torch_tpu
