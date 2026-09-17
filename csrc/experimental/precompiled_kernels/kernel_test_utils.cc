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

#include "csrc/experimental/precompiled_kernels/kernel_test_utils.h"

#include <memory>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> CreateKernelTestClient() {
  TT_ASSIGN_OR_RETURN(bool initialized,
                      AdaptXlaError(pjrt::IsPjrtPluginInitialized("tpu"),
                                    "failed to query test plugin"));
  if (!initialized) {
    TT_RETURN_IF_ERROR(AdaptXlaError(pjrt::InitializePjrtPlugin("tpu"),
                                     "failed to initialize test plugin"));
  }
  TT_ASSIGN_OR_RETURN(
      auto client,
      AdaptXlaError(xla::GetCApiClient("tpu"), "failed to create test client"));
  return client;
}

}  // namespace torch_tpu
