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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_DTYPE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_DTYPE_H_

#include <cstdint>
#include <ostream>
#include <string_view>

#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {

using Shape = llvm::SmallVector<int64_t, 4>;

// An element data type.
class DType {
 public:
  enum class Kind : uint8_t {
    kInvalid = 0,
    kBool,
    kI8,
    kI16,
    kI32,
    kI64,
    kF16,
    kBF16,
    kF32,
  };

  constexpr DType() = default;
  constexpr explicit DType(Kind kind) : kind_(kind) {}

  constexpr Kind kind() const { return kind_; }

  constexpr explicit operator bool() const { return kind_ != Kind::kInvalid; }

  constexpr bool IsBool() const { return kind_ == Kind::kBool; }

  constexpr bool IsFloat() const {
    return kind_ == Kind::kF16 || kind_ == Kind::kBF16 || kind_ == Kind::kF32;
  }
  constexpr bool IsInt() const {
    return kind_ == Kind::kI8 || kind_ == Kind::kI16 || kind_ == Kind::kI32 ||
           kind_ == Kind::kI64;
  }
  constexpr bool IsIntOrFloat() const { return IsInt() || IsFloat(); }
  constexpr bool IsInteger(unsigned width) const {
    return IsInt() && GetIntOrFloatBitWidth() == width;
  }
  constexpr unsigned GetIntOrFloatBitWidth() const {
    switch (kind_) {
      case Kind::kInvalid:
        return 0;
      case Kind::kBool:
        return 1;
      case Kind::kI8:
        return 8;
      case Kind::kI16:
      case Kind::kF16:
      case Kind::kBF16:
        return 16;
      case Kind::kI32:
      case Kind::kF32:
        return 32;
      case Kind::kI64:
        return 64;
    }
  }

  constexpr std::string_view ToString() const {
    switch (kind_) {
      case Kind::kInvalid:
        return "invalid";
      case Kind::kBool:
        return "bool";
      case Kind::kI8:
        return "i8";
      case Kind::kI16:
        return "i16";
      case Kind::kI32:
        return "i32";
      case Kind::kI64:
        return "i64";
      case Kind::kF16:
        return "f16";
      case Kind::kBF16:
        return "bf16";
      case Kind::kF32:
        return "f32";
    }
  }

  constexpr bool operator==(DType other) const { return kind_ == other.kind_; }
  constexpr bool operator!=(DType other) const { return kind_ != other.kind_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DType& dtype) {
    sink.Append(dtype.ToString());
  }

 private:
  Kind kind_ = Kind::kInvalid;
};

inline std::ostream& operator<<(std::ostream& os, DType dtype) {
  return os << dtype.ToString();
}

// Types
constexpr DType BoolType() { return DType(DType::Kind::kBool); }
constexpr DType Int8Type() { return DType(DType::Kind::kI8); }
constexpr DType Int16Type() { return DType(DType::Kind::kI16); }
constexpr DType Int32Type() { return DType(DType::Kind::kI32); }
constexpr DType Int64Type() { return DType(DType::Kind::kI64); }
constexpr DType Float16Type() { return DType(DType::Kind::kF16); }
constexpr DType BFloat16Type() { return DType(DType::Kind::kBF16); }
constexpr DType Float32Type() { return DType(DType::Kind::kF32); }

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_DTYPE_H_
