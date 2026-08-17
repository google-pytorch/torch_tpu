# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

td_library(
    name = "torch_tpu_mosaic_dialect_td_files",
    srcs = ["dialect/torch_tpu_mosaic_dialect.td"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

td_library(
    name = "torch_tpu_mosaic_attrs_td_files",
    srcs = ["dialect/torch_tpu_mosaic_attrs.td"],
    deps = [
        ":torch_tpu_mosaic_dialect_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

td_library(
    name = "torch_tpu_mosaic_ops_td_files",
    srcs = ["dialect/torch_tpu_mosaic_ops.td"],
    deps = [
        ":torch_tpu_mosaic_dialect_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "torch_tpu_mosaic_dialect_inc_gen",
    tbl_outs = {
        "dialect/torch_tpu_mosaic_dialect.h.inc": ["-gen-dialect-decls"],
        "dialect/torch_tpu_mosaic_dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/torch_tpu_mosaic_dialect.td",
    deps = [":torch_tpu_mosaic_dialect_td_files"],
)

gentbl_cc_library(
    name = "torch_tpu_mosaic_attrs_inc_gen",
    tbl_outs = {
        "dialect/torch_tpu_mosaic_attrs.h.inc": ["-gen-attrdef-decls"],
        "dialect/torch_tpu_mosaic_attrs.cc.inc": ["-gen-attrdef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/torch_tpu_mosaic_attrs.td",
    deps = [":torch_tpu_mosaic_attrs_td_files"],
)

gentbl_cc_library(
    name = "torch_tpu_mosaic_ops_inc_gen",
    tbl_outs = {
        "dialect/torch_tpu_mosaic_ops.h.inc": ["-gen-op-decls"],
        "dialect/torch_tpu_mosaic_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/torch_tpu_mosaic_ops.td",
    deps = [":torch_tpu_mosaic_ops_td_files"],
)

torch_tpu_cc_library(
    name = "torch_tpu_mosaic_dialect",
    srcs = ["dialect/torch_tpu_mosaic_dialect.cc"],
    hdrs = ["dialect/torch_tpu_mosaic_dialect.h"],
    deps = [
        ":torch_tpu_mosaic_attrs_inc_gen",
        ":torch_tpu_mosaic_dialect_inc_gen",
        ":torch_tpu_mosaic_ops_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_library(
    name = "torch_tpu_mosaic_attrs",
    srcs = ["dialect/torch_tpu_mosaic_attrs.cc"],
    hdrs = ["dialect/torch_tpu_mosaic_attrs.h"],
    deps = [
        ":torch_tpu_mosaic_attrs_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

torch_tpu_cc_library(
    name = "torch_tpu_mosaic_ops",
    srcs = ["dialect/torch_tpu_mosaic_ops.cc"],
    hdrs = ["dialect/torch_tpu_mosaic_ops.h"],
    deps = [":torch_tpu_mosaic_ops_inc_gen"],
)

torch_tpu_package_end()
