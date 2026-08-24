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

#include "torch_tpu/csrc/eager/device_buffer_utils.h"

#include <map>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/fingerprint_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::Pair;

TEST(DeviceBufferUtilsTest,
     ComputeConstantDeviceBufferRefOpParamCacheKeysMatchesGoldenMap) {
  const std::vector<char> data = {'a', 'b', 'c', 'd'};
  const Dimensions dims = {2, 3};
  const auto dtype = mlir::ElementType::F32;

  TF_ASSERT_OK_AND_ASSIGN(
      const OpParamCacheKeys keys,
      internal::ComputeConstantDeviceBufferRefOpParamCacheKeys(data, dims,
                                                               dtype));

  const std::map<std::string, FingerprintType> actual_map(keys.begin(),
                                                          keys.end());

  EXPECT_THAT(actual_map, ElementsAre(
                              // go/keep-sorted start
                              Pair("data", 1897425971756105985ULL),
                              Pair("dimensions", FingerprintCat("", 2, 3)),
                              Pair("element_type",
                                   Fingerprint("f32"))  //
                                                        // go/keep-sorted end
                              ));
}

}  // namespace
}  // namespace torch_tpu
