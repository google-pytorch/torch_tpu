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

#ifndef TORCH_TPU_OPS_FOREACH_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FOREACH_ATEN_KERNELS_H_

#include <optional>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

std::vector<at::Tensor> AtenForeachAbs(at::TensorList self);
void AtenForeachAbs_(at::TensorList self);

std::vector<at::Tensor> AtenForeachAcos(at::TensorList self);
void AtenForeachAcos_(at::TensorList self);

std::vector<at::Tensor> AtenForeachAsin(at::TensorList self);
void AtenForeachAsin_(at::TensorList self);

std::vector<at::Tensor> AtenForeachAtan(at::TensorList self);
void AtenForeachAtan_(at::TensorList self);

std::vector<at::Tensor> AtenForeachCeil(at::TensorList self);
void AtenForeachCeil_(at::TensorList self);

std::vector<at::Tensor> AtenForeachCos(at::TensorList self);
void AtenForeachCos_(at::TensorList self);

std::vector<at::Tensor> AtenForeachCosh(at::TensorList self);
void AtenForeachCosh_(at::TensorList self);

std::vector<at::Tensor> AtenForeachErf(at::TensorList self);
void AtenForeachErf_(at::TensorList self);

std::vector<at::Tensor> AtenForeachErfc(at::TensorList self);
void AtenForeachErfc_(at::TensorList self);

std::vector<at::Tensor> AtenForeachExp(at::TensorList self);
void AtenForeachExp_(at::TensorList self);

std::vector<at::Tensor> AtenForeachExpm1(at::TensorList self);
void AtenForeachExpm1_(at::TensorList self);

std::vector<at::Tensor> AtenForeachFloor(at::TensorList self);
void AtenForeachFloor_(at::TensorList self);

std::vector<at::Tensor> AtenForeachFrac(at::TensorList self);
void AtenForeachFrac_(at::TensorList self);

std::vector<at::Tensor> AtenForeachLgamma(at::TensorList self);
void AtenForeachLgamma_(at::TensorList self);

std::vector<at::Tensor> AtenForeachLog(at::TensorList self);
void AtenForeachLog_(at::TensorList self);

std::vector<at::Tensor> AtenForeachLog10(at::TensorList self);
void AtenForeachLog10_(at::TensorList self);

std::vector<at::Tensor> AtenForeachLog1p(at::TensorList self);
void AtenForeachLog1p_(at::TensorList self);

std::vector<at::Tensor> AtenForeachLog2(at::TensorList self);
void AtenForeachLog2_(at::TensorList self);

std::vector<at::Tensor> AtenForeachNeg(at::TensorList self);
void AtenForeachNeg_(at::TensorList self);

std::vector<at::Tensor> AtenForeachReciprocal(at::TensorList self);
void AtenForeachReciprocal_(at::TensorList self);

std::vector<at::Tensor> AtenForeachRound(at::TensorList self);
void AtenForeachRound_(at::TensorList self);

std::vector<at::Tensor> AtenForeachRsqrt(at::TensorList self);
void AtenForeachRsqrt_(at::TensorList self);

std::vector<at::Tensor> AtenForeachSigmoid(at::TensorList self);
void AtenForeachSigmoid_(at::TensorList self);

std::vector<at::Tensor> AtenForeachSign(at::TensorList self);
void AtenForeachSign_(at::TensorList self);

std::vector<at::Tensor> AtenForeachSin(at::TensorList self);
void AtenForeachSin_(at::TensorList self);

std::vector<at::Tensor> AtenForeachSinh(at::TensorList self);
void AtenForeachSinh_(at::TensorList self);

std::vector<at::Tensor> AtenForeachSqrt(at::TensorList self);
void AtenForeachSqrt_(at::TensorList self);

std::vector<at::Tensor> AtenForeachTan(at::TensorList self);
void AtenForeachTan_(at::TensorList self);

std::vector<at::Tensor> AtenForeachTanh(at::TensorList self);
void AtenForeachTanh_(at::TensorList self);

std::vector<at::Tensor> AtenForeachTrunc(at::TensorList self);
void AtenForeachTrunc_(at::TensorList self);

void AtenForeachZero_(at::TensorList self);

std::vector<at::Tensor> AtenForeachAddList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha);
std::vector<at::Tensor> AtenForeachAddScalar(at::TensorList self,
                                             const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachAddScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
std::vector<at::Tensor> AtenForeachAddTensor(at::TensorList self,
                                             const at::Tensor& other,
                                             const at::Scalar& alpha);

void AtenForeachAdd_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha);
void AtenForeachAdd_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachAdd_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars);
void AtenForeachAdd_Tensor(at::TensorList self, const at::Tensor& other,
                           const at::Scalar& alpha);

std::vector<at::Tensor> AtenForeachAddcdivScalar(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Scalar& value);
std::vector<at::Tensor> AtenForeachAddcdivScalarList(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars);
std::vector<at::Tensor> AtenForeachAddcdivTensor(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Tensor& scalars);

void AtenForeachAddcdiv_Scalar(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2, const at::Scalar& value);
void AtenForeachAddcdiv_ScalarList(at::TensorList self, at::TensorList tensor1,
                                   at::TensorList tensor2,
                                   at::ArrayRef<at::Scalar> scalars);
void AtenForeachAddcdiv_Tensor(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Tensor& scalars);

std::vector<at::Tensor> AtenForeachAddcmulScalar(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Scalar& value);
std::vector<at::Tensor> AtenForeachAddcmulScalarList(
    at::TensorList self, at::TensorList tensor1, at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars);
std::vector<at::Tensor> AtenForeachAddcmulTensor(at::TensorList self,
                                                 at::TensorList tensor1,
                                                 at::TensorList tensor2,
                                                 const at::Tensor& scalars);

void AtenForeachAddcmul_Scalar(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2, const at::Scalar& value);
void AtenForeachAddcmul_ScalarList(at::TensorList self, at::TensorList tensor1,
                                   at::TensorList tensor2,
                                   at::ArrayRef<at::Scalar> scalars);
void AtenForeachAddcmul_Tensor(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2,
                               const at::Tensor& scalars);

std::vector<at::Tensor> AtenForeachDivList(at::TensorList self,
                                           at::TensorList other);
std::vector<at::Tensor> AtenForeachDivScalar(at::TensorList self,
                                             const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachDivScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
std::vector<at::Tensor> AtenForeachDivTensor(at::TensorList self,
                                             const at::Tensor& other);
void AtenForeachDiv_List(at::TensorList self, at::TensorList other);
void AtenForeachDiv_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachDiv_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars);
void AtenForeachDiv_Tensor(at::TensorList self, const at::Tensor& other);

std::vector<at::Tensor> AtenForeachLerpList(at::TensorList self,
                                            at::TensorList other,
                                            at::TensorList weight);
std::vector<at::Tensor> AtenForeachLerpScalar(at::TensorList self,
                                              at::TensorList other,
                                              const at::Scalar& weight);
std::vector<at::Tensor> AtenForeachLerpScalarList(
    at::TensorList self, at::TensorList other,
    at::ArrayRef<at::Scalar> scalars);
void AtenForeachLerp_List(at::TensorList self, at::TensorList other,
                          at::TensorList weight);
void AtenForeachLerp_Scalar(at::TensorList self, at::TensorList other,
                            const at::Scalar& weight);
void AtenForeachLerp_ScalarList(at::TensorList self, at::TensorList other,
                                at::ArrayRef<at::Scalar> scalars);

std::vector<at::Tensor> AtenForeachMulList(at::TensorList self,
                                           at::TensorList other);
std::vector<at::Tensor> AtenForeachMulScalar(at::TensorList self,
                                             const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachMulScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
std::vector<at::Tensor> AtenForeachMulTensor(at::TensorList self,
                                             const at::Tensor& other);

void AtenForeachMul_List(at::TensorList self, at::TensorList other);
void AtenForeachMul_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachMul_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars);
void AtenForeachMul_Tensor(at::TensorList self, const at::Tensor& other);

std::vector<at::Tensor> AtenForeachSubList(at::TensorList self,
                                           at::TensorList other,
                                           const at::Scalar& alpha);
std::vector<at::Tensor> AtenForeachSubScalar(at::TensorList self,
                                             const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachSubScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);

void AtenForeachSub_List(at::TensorList self, at::TensorList other,
                         const at::Scalar& alpha);
void AtenForeachSub_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachSub_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> scalars);

std::vector<at::Tensor> AtenForeachClampMaxScalar(at::TensorList self,
                                                  const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachClampMaxList(at::TensorList self,
                                                at::TensorList other);
std::vector<at::Tensor> AtenForeachClampMaxScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
void AtenForeachClampMax_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachClampMax_List(at::TensorList self, at::TensorList other);
void AtenForeachClampMax_ScalarList(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars);

std::vector<at::Tensor> AtenForeachClampMinScalar(at::TensorList self,
                                                  const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachClampMinList(at::TensorList self,
                                                at::TensorList other);
std::vector<at::Tensor> AtenForeachClampMinScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
void AtenForeachClampMin_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachClampMin_List(at::TensorList self, at::TensorList other);
void AtenForeachClampMin_ScalarList(at::TensorList self,
                                    at::ArrayRef<at::Scalar> scalars);

void AtenForeachCopy_(at::TensorList self, at::TensorList src,
                      bool non_blocking);

std::vector<at::Tensor> AtenForeachNormScalar(
    at::TensorList self, const at::Scalar& ord,
    const std::optional<c10::ScalarType> dtype);

std::vector<at::Tensor> AtenForeachMax(at::TensorList self);

std::vector<at::Tensor> AtenForeachMaximumScalar(at::TensorList self,
                                                 const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachMaximumList(at::TensorList self,
                                               at::TensorList other);
std::vector<at::Tensor> AtenForeachMaximumScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
void AtenForeachMaximum_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachMaximum_List(at::TensorList self, at::TensorList other);
void AtenForeachMaximum_ScalarList(at::TensorList self,
                                   at::ArrayRef<at::Scalar> scalars);

std::vector<at::Tensor> AtenForeachMinimumScalar(at::TensorList self,
                                                 const at::Scalar& scalar);
std::vector<at::Tensor> AtenForeachMinimumList(at::TensorList self,
                                               at::TensorList other);
std::vector<at::Tensor> AtenForeachMinimumScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> scalars);
void AtenForeachMinimum_Scalar(at::TensorList self, const at::Scalar& scalar);
void AtenForeachMinimum_List(at::TensorList self, at::TensorList other);
void AtenForeachMinimum_ScalarList(at::TensorList self,
                                   at::ArrayRef<at::Scalar> scalars);

std::vector<at::Tensor> AtenForeachPowList(at::TensorList self,
                                           at::TensorList exponent);
std::vector<at::Tensor> AtenForeachPowScalar(at::TensorList self,
                                             const at::Scalar& exponent);
std::vector<at::Tensor> AtenForeachPowScalarList(
    at::TensorList self, at::ArrayRef<at::Scalar> exponent);
std::vector<at::Tensor> AtenForeachPowScalarAndTensor(const at::Scalar& self,
                                                      at::TensorList exponent);
void AtenForeachPow_List(at::TensorList self, at::TensorList exponent);
void AtenForeachPow_Scalar(at::TensorList self, const at::Scalar& exponent);
void AtenForeachPow_ScalarList(at::TensorList self,
                               at::ArrayRef<at::Scalar> exponent);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FOREACH_ATEN_KERNELS_H_
