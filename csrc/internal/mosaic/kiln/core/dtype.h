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

#include <concepts>
#include <cstdint>
#include <ostream>
#include <string_view>

#include "llvm/ADT/ArrayRef.h"

namespace torch_tpu::kiln {

class Shape;

// An element data type.
class DType {
 public:
  // TODO(willfroom): replace this with a bit field of:
  // - log2(width): 3 bits
  // - float vs int: 1 bit
  // - signedness: 1 bit
  enum class Kind : uint8_t {
    kBool,
    kI8,
    kI16,
    kI32,
    kI64,
    kU8,
    kU16,
    kU32,
    kU64,
    kF16,
    kBF16,
    kF32,
  };

  constexpr explicit DType(Kind kind) : kind_(kind) {}

  constexpr DType(const DType&) = default;
  constexpr DType& operator=(const DType&) = default;
  constexpr DType(DType&&) noexcept = default;
  constexpr DType& operator=(DType&&) noexcept = default;

  constexpr Kind kind() const { return kind_; }

  Shape operator()(llvm::ArrayRef<int64_t> dims) const;

  template <std::convertible_to<int64_t>... Dims>
  Shape operator()(Dims... dims) const;

  constexpr bool IsBool() const { return kind_ == Kind::kBool; }

  constexpr bool IsFloat() const {
    return kind_ == Kind::kF16 || kind_ == Kind::kBF16 || kind_ == Kind::kF32;
  }
  constexpr bool IsSigned() const {
    return kind_ == Kind::kI8 || kind_ == Kind::kI16 || kind_ == Kind::kI32 ||
           kind_ == Kind::kI64;
  }
  constexpr bool IsUnsigned() const {
    return kind_ == Kind::kU8 || kind_ == Kind::kU16 || kind_ == Kind::kU32 ||
           kind_ == Kind::kU64;
  }
  constexpr bool IsInt() const { return IsSigned() || IsUnsigned(); }
  constexpr bool IsIntOrFloat() const { return IsInt() || IsFloat(); }
  constexpr bool IsInteger(unsigned width) const {
    return IsInt() && GetIntOrFloatBitWidth() == width;
  }
  constexpr bool IsSignedInteger(unsigned width) const {
    return IsSigned() && GetIntOrFloatBitWidth() == width;
  }
  constexpr bool IsUnsignedInteger(unsigned width) const {
    return IsUnsigned() && GetIntOrFloatBitWidth() == width;
  }
  constexpr unsigned GetIntOrFloatBitWidth() const {
    switch (kind_) {
      case Kind::kBool:
        return 1;
      case Kind::kI8:
      case Kind::kU8:
        return 8;
      case Kind::kI16:
      case Kind::kU16:
      case Kind::kF16:
      case Kind::kBF16:
        return 16;
      case Kind::kI32:
      case Kind::kU32:
      case Kind::kF32:
        return 32;
      case Kind::kI64:
      case Kind::kU64:
        return 64;
    }
  }

  constexpr std::string_view ToString() const {
    switch (kind_) {
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
      case Kind::kU8:
        return "u8";
      case Kind::kU16:
        return "u16";
      case Kind::kU32:
        return "u32";
      case Kind::kU64:
        return "u64";
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
  Kind kind_;
};

inline std::ostream& operator<<(std::ostream& os, DType dtype) {
  return os << dtype.ToString();
}

// Types.
// Intentionally deviate from Google style guide (k-prefix for constexpr values)
// to improve readability of the DSL.
inline constexpr DType bool_(DType::Kind::kBool);
inline constexpr DType i8(DType::Kind::kI8);
inline constexpr DType i16(DType::Kind::kI16);
inline constexpr DType i32(DType::Kind::kI32);
inline constexpr DType i64(DType::Kind::kI64);
inline constexpr DType u8(DType::Kind::kU8);
inline constexpr DType u16(DType::Kind::kU16);
inline constexpr DType u32(DType::Kind::kU32);
inline constexpr DType u64(DType::Kind::kU64);
inline constexpr DType f16(DType::Kind::kF16);
inline constexpr DType bf16(DType::Kind::kBF16);
inline constexpr DType f32(DType::Kind::kF32);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_DTYPE_H_
