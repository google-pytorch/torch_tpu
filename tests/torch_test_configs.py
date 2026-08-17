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

"""TorchTPU capability configuration for upstream device-generic suites.

Declared through ``PrivateUse1TestBase.set_test_configs`` (torch 2.13+;
reference usage: pytorch ``test/cpp_extensions/open_registration_extension/
torch_openreg/tests/test_testing.py``). Every entry below is a test whose
lowering fatally crashes the process on TPU (a failure would be fine, a
crash discards the rest of the shard); ordinary failures stay visible.
Keep each entry annotated with its cause; remove entries as fixes land.

Layers, all applied by ``install(module)`` before the upstream module is
imported:

1. Backend dtype support: every OpInfo gets ``dtypesIf["tpu"]`` — the
   per-backend dtype declaration ``OpInfo.supported_dtypes()`` is built
   around — equal to its generic dtypes minus ``_CRASHING_DTYPES``;
   ``PrivateUse1TestBase._get_dtypes`` (consulted for ``@dtypes`` tests)
   is filtered the same way. Excluded variants are never instantiated.
   ``_MODULE_CRASHING_DTYPES`` extends the set per module, for suites whose
   own reference machinery upcasts a dtype on device.
2. ``_TEST_EXCLUSIONS`` (``set_test_configs(test_exclusions=...)``): whole
   generic methods or classes ("*"), keyed by the generic class name.
3. ``_OP_OVERRIDES`` (``set_test_configs(op_overrides=...)``): per-op
   ``DecorateInfo(unittest.skip)`` entries, optionally scoped to a generic
   class / test name / dtypes — upstream's own mechanism for op_db-driven
   suites. Expanded from ``_OP_OVERRIDE_RULES`` (op selectors may be exact
   ``OpInfo.full_name``s or ``prefix.*``).
4. ``_TEST_NAME_EXCLUSIONS``: the residue set_test_configs cannot reach —
   tests in plain (non-device-generic) classes, and device-generic
   variants parametrized by something other than op_db (module_db,
   layouts). Regexes over ``ClassName.test_name``, applied in the driver's
   loader; per module.

Class naming: ``install()`` runs ``PrivateUse1TestBase.setUpClass()`` up
front, which renames the base's device_type from "privateuse1" to "tpu".
Upstream would otherwise do that lazily on the first instantiated test, so
the first device-generic class in a module would be named ``FooPRIVATEUSE1``
and later ones ``FooTPU`` — and op_overrides (device-matched against
"tpu") would silently miss the first class. With the up-front rename every
instantiated class is ``<Generic>TPU``, which is the name the regexes in
layer 4 use.

The dominant crash cause is a libtpu compiler fatal on any conversion into
complex128 (docs/libtpu_c128_cvt_crash.md; 203 of 324 shards in the
2026-08-21 baseline), which is why complex128 is excluded wholesale.
"""

import functools
import unittest

# Dtypes whose lowering fatally crashes the TPU process, in every suite.
_CRASHING_DTYPES = frozenset({
    "complex128",  # libtpu CVT-into-c128 FATAL (docs/libtpu_c128_cvt_crash.md)
})

# Per-module additions to _CRASHING_DTYPES: the suite's own reference,
# decomposition, meta or redispatch machinery upcasts the dtype into a
# crashing one on device (complex64 -> complex128; float64 -> complex128
# through complex-producing ops), across so many ops that per-op rules
# would be the whole op_db.
_MODULE_CRASHING_DTYPES = {
    "complex_tensor/test_complex_tensor": {"complex64"},
    "test_binary_ufuncs": {"complex64"},  # two-axis c64 x f64 sweeps
    "test_decomp": {"complex64"},
    "test_meta": {"complex64"},
    "test_modules": {"float64"},
    "test_ops": {"complex64"},
    "test_ops_jit": {"complex64"},
    "test_optim": {"complex64"},
    "test_overrides": {"complex64"},
    "test_reductions": {"complex64"},
    "test_spectral_ops": {"float64", "complex64"},
}

# Whole generic methods / classes, keyed by the generic (template) class
# name. Each was identified from the last started test in a crashed
# shard's verbose log.
_TEST_EXCLUSIONS = {
    # Probe ops on device across all dtypes / error inputs / wrong out=
    # buffers, independent of declared support: c128 CVT fatal, SIGFPE and
    # LLVM aborts on error inputs, SIGSEGV on mismatched out= buffers.
    "TestCommon": [
        "test_dtypes",
        "test_errors",
        "test_out",
        "test_out_warning",
    ],
    # c128 reached through internal dtype conversion or storage/copy paths;
    # test_storage* SIGSEGV; scatter_add deterministic path aborts.
    "TestTorchDeviceType": [
        "test_clone_all_dtypes_and_devices",
        "test_copy_",
        "test_deepcopy_scalar",
        "test_deterministic_resize",
        "test_gather_backward_deterministic_path",
        "test_gradient_all",
        "test_gradient_extreme_cases",
        "test_gradient_type_promotion",
        "test_lazy_clone_view_materialize",
        "test_scatter_add_one_dim_deterministic",
        "test_set_storage",
        "test_storage",
        "test_storage_setitem",
        "test_tensor_from_storage",
    ],
    # Internal all-dtypes copy sweeps reach complex128.
    "TestAsArray": [
        "test_copy_from_dlpack",
        "test_copy_tensor",
    ],
    # In-body promotion sweeps (c64 x f64, complex scalar ** real) -> c128.
    "TestBinaryUfuncsDevice": [
        "test_add",
        "test_float_power",  # bf16 x bf16 computes in c128
        "test_float_scalar_pow_float_tensor",  # SIGILL
        "test_complex_scalar_pow_tensor",
        "test_type_promotion",
    ],
    "TestFFT": [
        "test_fft_round_trip",  # f32 round trip reaches c128
        "test_fftn_round_trip",  # LLVM ERROR in the nd path
        # OpDTypes.unsupported probing executes bf16/half FFTs expecting a
        # clean error; the lowering aborts instead.
        "test_fft_half_and_bfloat16_errors",
        "test_fft_plan_repeatable",  # c128 CVT fatal
        "test_fftshift_numpy",  # c128 CVT fatal
        "test_hfftn",  # LLVM ERROR in the nd hermitian path
    ],
    # Internal all-dtypes loops reach c128; linspace lowering crashes.
    "TestTensorCreation": [
        "test_block_diag",
        "test_cat_all_dtypes_and_devices",
        "test_complex_type_conversions",
        "test_linspace",
        "test_torch_complex",  # f64 pair -> c128
        "test_torch_polar",  # f32 polar -> c128
    ],
    "TestAutogradDeviceType": [
        "test_copy_r_to_c",  # hardcodes cdouble
        "test_pyscalar_conversions",  # python complex scalar -> c128
        "test_scatter_index_reduce_amin_amax_aminmax_backprops_to_all_values",
        "test_scatter_index_reduce_prod_gradgrad_error",
        "test_sparse_backward",  # f64 sparse backward reaches c128
    ],
    "TestReductions": [
        "test_minmax_illegal_dtype",  # illegal-dtype probe reaches c128
        "test_scalar_tensor_dim_compiled_mode",  # compiled_mode CHECK
    ],
    "TestLinalg": [
        "test_addr_type_promotion",  # c64 x f64 -> c128
        "test_bmm",  # SIGSEGV (f16/bf16/f64)
        "test_dot_invalid_args",  # LLVM ERROR: failed to infer result type
        "test_vdot_invalid_args",
        "test_vector_norm",  # c64/f64 norms reach c128
    ],
    "TestMeta": [
        "test_meta__fused_moving_avg_obs_fq_helper",  # LLVM ERROR
    ],
    "TestMultiprocessingDeviceType": [
        "test_leaf_variable_sharing",  # forked child aborts on TPU init
        "test_parameter_sharing",  # same: child aborts, parent hangs
        "test_variable_sharing",  # same
    ],
    "TestNNDeviceType": [
        "test_ctc_loss_error",  # CompositeOpCheck fatal on ctc_loss.Tensor
        "test_pad",  # LLVM ERROR in reflection_pad2d lowering
    ],
    "TestPoolingNNDeviceType": [
        "test_adaptive_pooling_empty_output_size",  # SIGFPE (div by zero)
        "test_max_pool2d_nhwc",  # SIGILL in channels-last paths
        "test_max_pool3d_ndhwc",
        "test_pooling_max_nhwc",
        "test_pooling_zero_stride",  # SIGFPE (div by zero)
    ],
    "TestVmapBatchedGradientLegacy": [
        "test_sigmoid",  # LLVM ERROR: batched backward multiply
        "test_threshold",
    ],
    # backward() on nested tensors hits aten::_nested_tensor_size with no
    # NestedTensor-key kernel; the c10::NotImplementedError escapes on the
    # autograd engine thread -> std::terminate.
    "TestNestedTensorAutograd": "*",
    "TestFloat4Dtype": [
        "test_float4_e2m1fn_x2",  # tensor_to_buffer CHECK on packed fp4
    ],
    "TestTypePromotion": [
        "test_bfloat16",
        "test_cat_different_dtypes",
        "test_comparison_ops_with_type_promotion",
        "test_complex_scalar_mult_tensor_promotion",
        "test_many_promotions",
        "test_result_type",
        "test_ternary_out_promotion",
        "test_unary_op_out_casting",
    ],
    "TestShapeOps": ["test_flip"],
    "TestUnaryUfuncs": [
        "test_nan_to_num_complex",  # c64 nan_to_num reaches c128
    ],
    "TestViewOps": [
        "test_view_tensor_dsplit",
        "test_view_tensor_hsplit",
        "test_view_tensor_vsplit",
    ],
    "TestMasked": [
        "test_mask_layout",
        "test_reference_masked",
    ],
    # SIGABRT on boolean/byte mask indexing; index_copy deterministic path
    # is OOM-killed.
    "TestIndexing": [
        "test_bool_indices",
        "test_byte_mask",
        "test_byte_mask_accumulate",
        "test_empty_ndim_index",
        "test_index_copy_deterministic",
        "test_multi_dimensional_bool_mask",
    ],
    # torch_tpu tensor_to_buffer.cc fatal CHECK: DLPack import of a
    # PrivateUse1 tensor not owned by the TPU allocator.
    "TestTorchDlPack": "*",
}

# Op-level skips, expanded into _OP_OVERRIDES at install():
#   (op selector, dtype names or None for all, generic class or None,
#    generic test name or None, reason)
_OP_OVERRIDE_RULES = [
    # Ops whose semantics reach complex128 from non-complex128 inputs.
    ("cdouble", None, None, None, "converts any input to complex128"),
    ("_refs._conversions.cdouble", None, None, None, "converts to c128"),
    ("cfloat", None, None, None, "reference paths upcast to complex128"),
    ("_refs._conversions.cfloat", None, None, None, "reference -> c128"),
    ("complex", ["float64", "float32"], None, None, "complex(f, f) -> c128"),
    ("_refs._conversions.complex", ["float64", "float32"], None, None,
     "complex(f, f) ref -> c128"),
    ("polar", None, None, None, "reference paths reach c128"),
    ("_refs._conversions.polar", None, None, None, "reference -> c128"),
    ("view_as_complex", ["float64"], None, None, "f64 view -> c128"),
    ("fft.*", ["float64"], None, None, "float64 FFT promotes to c128"),
    # The _refs.* mirrors carry the same semantics and crash the same
    # way in the python-ref suites.
    ("_refs.fft.*", ["float64"], None, None, "float64 FFT ref -> c128"),
    ("stft", ["float64"], None, None, "float64 STFT promotes to c128"),
    ("float_power", None, None, None, "computes in f64/c128 by definition"),
    ("_refs.float_power", None, None, None, "computes in f64/c128"),
    # Backend kernel crashes, everywhere they are instantiated.
    ("max_pool2d_with_indices_backward", None, None, None, "SIGILL"),
    ("combinations", None, None, None, "LLVM ERROR / c128 CVT"),
    ("nn.functional.unfold", None, None, None, "LLVM ERROR"),
    ("nn.functional.glu", None, None, None, "SIGILL"),
    ("polygamma*", ["float64"], None, None, "float64 polygamma OOM-kills"),
    ("_refs.polygamma*", ["float64"], None, None, "float64 OOM-kills"),
    ("_refs.lgamma", ["float64"], None, None, "abort"),
    ("nansum", None, "TestMeta", None, "SIGILL under meta dispatch"),
    # _refs mirrors crashing in the python-ref suites (TestCommon).
    ("_refs.fliplr", None, None, None, "c128 CVT in ref path"),
    ("_refs.fmax", None, None, None, "SIGILL in ref path"),
    ("_refs.fmod", None, None, None, "c128 CVT in ref path"),
    ("_refs.ge", None, None, None, "c128 CVT in ref path"),
    ("_refs.view_as_complex", ["float64"], None, None, "f64 -> c128"),
    ("_refs.xlogy", None, None, None, "SIGILL in ref path"),
    ("_refs.stft", None, None, None, "SIGILL in ref path"),
    ("_refs.zeros", None, None, None, "LLVM ERROR in ref path"),
    # Backward/neg-view lowerings of the nn.functional family SIGILL or
    # LLVM-ERROR at float64 wherever they are exercised.
    ("nn.functional.max_pool*", None, None, None, "SIGILL"),
    ("nn.functional.max_unpool*", None, None, None, "SIGILL"),
    ("nn.functional.*", ["float64"], "TestMathBits", None,
     "neg-view lowering SIGILL / LLVM ERROR at float64"),
    ("nn.functional.pad*", None, "TestCompositeCompliance", None,
     "COW-input pad lowering LLVM ERROR"),
    # Decomposition-only crashers (the eager kernels are fine).
    ("combinations", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("grid_sampler_2d", None, "TestDecomp", None, "decomp abort"),
    ("masked_var", None, "TestDecomp", None, "decomp abort"),
    ("repeat", None, "TestDecomp", None, "decomp abort"),
    ("take", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("to", None, "TestDecomp", None, "decomp abort"),
    ("nn.functional.*", None, "TestDecomp", None,
     "decomposed nn kernels abort (max_pool/loss/grid_sample/interpolate)"),
    # Decompositions that LLVM-ERROR, SIGILL or hang past the 1h timeout.
    ("put", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("tile", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("transpose", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("unfold_copy", None, "TestDecomp", None, "LLVM ERROR / SIGILL"),
    ("xlogy", None, "TestDecomp", None, "LLVM ERROR / hang"),
    ("masked.var", None, "TestDecomp", None, "hang past timeout"),
    ("var_mean", None, "TestDecomp", None, "hang past timeout"),
    ("mvlgamma*", None, "TestDecomp", None, "hang past timeout"),
    ("split", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("diagonal_scatter", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("nanquantile", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("signbit", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("tensor_split", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("torch.ops.aten._safe_softmax*", None, "TestDecomp", None,
     "LLVM ERROR in decomp"),
    ("view", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("_unsafe_masked_index*", None, "TestDecomp", None, "SIGILL / LLVM"),
    ("eq", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    ("rot90", None, "TestDecomp", None, "LLVM ERROR in decomp"),
    # Meta-dispatch crashers (CompositeOpCheck fatals, SIGILL).
    ("nn.functional.batch_norm", None, "TestMeta", None, "CompositeOpCheck"),
    ("nn.functional.cross_entropy", None, "TestMeta", None, "abort"),
    ("nn.functional.instance_norm", None, "TestMeta", None, "abort"),
    ("nn.functional.linear_cross_entropy*", None, "TestMeta", None, "abort"),
    ("nn.functional.nll_loss", None, "TestMeta", None, "abort"),
    ("nn.functional.max_pool*", None, "TestMeta", None, "SIGILL"),
    ("nn.functional.max_unpool*", None, "TestMeta", None, "SIGILL"),
    ("unique_consecutive", None, "TestMeta", None, "abort"),
    # float64 backward lowering of nn ops / sigmoid-family: LLVM ERROR or
    # CHECK failures during gradcheck.
    ("nn.functional.*", ["float64"], "TestBwdGradients", None, "f64 grad"),
    ("nn.functional.*", ["float64"], "TestFwdGradients", None, "f64 grad"),
    ("sigmoid", ["float64"], "TestBwdGradients", None, "LLVM ERROR"),
    ("sigmoid", ["float64"], "TestFwdGradients", None, "LLVM ERROR"),
    ("combinations", ["float64"], "TestBwdGradients", None, "c128 CVT"),
    ("combinations", ["float64"], "TestFwdGradients", None, "c128 CVT"),
    ("grid_sampler_3d", ["float64"], "TestBwdGradients", None, "abort"),
    # foreach: parity sweeps reach c128; scalar-list error cases SIGSEGV
    # in GetOutputDtypeFromTensorAndScalar.
    ("_foreach_*", None, "TestForeach", "test_parity", "c128 CVT"),
    ("_foreach_*", None, "TestForeach", "test_binary_op_list_error_cases",
     "SIGSEGV on mismatched ScalarList"),
    ("_foreach_pow", None, "TestForeach",
     "test_binary_op_with_scalar_self_support", "SIGSEGV"),
]

# Residue set_test_configs cannot express: plain (non-device-generic)
# classes, and device-generic variants parametrized by module_db or
# layouts. Regexes over "ClassName.test_name" (device-generic classes are
# always <Generic>TPU, see the module docstring), per upstream module.
_TEST_NAME_EXCLUSIONS = {
    # WithCompiledAutograd-wrapped reruns and the bwd-scan HOP test abort.
    "inductor/test_compiled_autograd": [
        r"^(HigherOrderOpTests|FuncTorchHigherOrderOpTests"
        r"|ActivationCheckpointingTests)WithCompiledAutograd\.",
        r"^TestCompiledAutogradOpInfoTPU\.test_hops_in_bwd_scan_simple",
    ],
    # CPU-compiled graphs handed to the TPU compiled-mode buffer path ->
    # compiled_mode.cc CHECK abort.
    "inductor/test_torchinductor_dynamic_shapes": [
        r"^DynamicShapesCpuTests\.test_(graph_partition_\w+"
        r"|like_channels_last_dynamic_shapes_cpu)",
    ],
    # DataLoader workers cannot init the TPU (device busy) -> hang.
    "profiler/test_profiler": [
        r"^TestExperimentalUtils\.test_profiler_synchronized_dataloader_pattern",
    ],
    # fork()ed children abort on TPU re-init; parents hang.
    "test_dataloader": [
        r"^TestConvAfterFork\.test_conv_after_fork",
        r"^TestDataLoader\.test_(batch_sampler|builtin_collection_conversion|bulk_loading_nobatch|chain_iterable_style_dataset)",
    ],
    "test_multiprocessing": [
        r"^TestMultiprocessing\.test_(autograd_fine_with_spawn"
        r"|parameter_sharing)",
    ],
    "test_python_dispatch": [
        r"^TestPythonDispatch\."
        r"test_wrapper_subclass_multiprocessing_preserves_dtype",
    ],
    # module_db-parametrized: CTC / NLL / linear-cross-entropy modules abort.
    "test_modules": [
        r"^TestModuleTPU\.test_\w*_nn_(NLLLoss|CTCLoss|LinearCrossEntropyLoss)",
    ],
    # CPU tensor into compiled mode; c10 error on the autograd thread.
    "test_nestedtensor": [
        r"^TestNestedTensorSubclassTPU\.test_(dummy_mha_with_nt|sdpa)",
    ],
    # Block-sparse sum at complex64 compiles a c128 upcast.
    "test_sparse_csr": [
        r"^TestSparseCompressedTPU\."
        r"test_consistency_SparseBS[RC]_sum_tpu_complex64",
    ],
    # Single dtype variants of otherwise-fine methods reaching c128.
    "test_torch": [
        r"^TestTorchDeviceTypeTPU\.test_deepcopy_tpu_complex64$",
    ],
    "test_linalg": [
        r"^TestLinalgTPU\.test_(v?dot_vs_numpy|inner|norm_dtype)"
        r"_tpu_complex64$",
        r"^TestLinalgTPU\.test_outer_type_promotion_tpu_"
        r"(float64_complex64|complex64_float64)$",
    ],
}

# When set, only these ops generate @ops variants. None = all ops.
_OP_ALLOWLIST = None


@functools.lru_cache(maxsize=1)
def get_tpu_supported_dtypes():
  """Returns the dtypes the TPU test envelope allows globally, as a list."""
  from torch.testing._internal import common_dtype  # pylint: disable=g-import-not-at-top

  crashing = _crashing_dtypes(None)
  return [
      dt
      for dt in common_dtype.get_all_dtypes(include_complex32=True)
      if dt not in crashing
  ]


def loader_filters(module):
  """(excluded dtype names, name regexes) for the driver's loader filter.

  The loader filter is the backstop for parametrizers that take dtype
  lists straight from their decorator arguments (@optims, ...) and for
  the _TEST_NAME_EXCLUSIONS residue.
  """
  names = set(_CRASHING_DTYPES) | _MODULE_CRASHING_DTYPES.get(module, set())
  return sorted(names), list(_TEST_NAME_EXCLUSIONS.get(module, []))


def _crashing_dtypes(module):
  import torch  # pylint: disable=g-import-not-at-top

  names = set(_CRASHING_DTYPES) | _MODULE_CRASHING_DTYPES.get(module, set())
  return frozenset(getattr(torch, name) for name in names)


def _op_selector_matches(selector, full_name):
  if selector.endswith("*"):
    return full_name.startswith(selector[:-1])
  return full_name == selector


def _all_op_infos():
  from torch.testing._internal import common_methods_invocations as cmi  # pylint: disable=g-import-not-at-top
  from torch.testing._internal.opinfo import core as opinfo_core  # pylint: disable=g-import-not-at-top

  seen = set()
  for value in vars(cmi).values():
    if not (isinstance(value, list) and value
            and isinstance(value[0], opinfo_core.OpInfo)):
      continue
    for op in value:
      if id(op) not in seen:
        seen.add(id(op))
        yield op


def _declare_backend_dtype_support(crashing):
  """Declares TPU dtype support (generic dtypes minus crashers).

  OpInfos get ``dtypesIf["tpu"]`` (consulted by ``supported_dtypes`` for
  the renamed PrivateUse1 backend); ``backward_dtypes`` and ModuleInfo
  ``dtypes`` have no per-backend hook and are filtered in place — safe
  because each upstream module runs in its own process.
  """
  from torch.testing._internal import common_dtype  # pylint: disable=g-import-not-at-top
  from torch.testing._internal import common_modules  # pylint: disable=g-import-not-at-top

  def filtered(dtypes):
    return common_dtype._dispatch_dtypes(  # pylint: disable=protected-access
        dt for dt in dtypes if dt not in crashing
    )

  for op in _all_op_infos():
    op.dtypesIf["tpu"] = filtered(op.supported_dtypes("tpu"))
    if op.backward_dtypes:
      op.backward_dtypes = filtered(op.backward_dtypes)
  for module_info in common_modules.module_db:
    module_info.dtypes = filtered(module_info.dtypes)


def _install_dtypes_decorator_filter(crashing):
  """Filters @dtypes-parametrized variants down to supported dtypes."""
  from torch.testing._internal import common_device_type  # pylint: disable=g-import-not-at-top

  base = common_device_type.PrivateUse1TestBase
  original = base._get_dtypes.__func__  # pylint: disable=protected-access

  def allowed(entry):
    # Entries are single dtypes or tuples (multi-axis @dtypes(*product())).
    if isinstance(entry, (tuple, list)):
      return not any(dt in crashing for dt in entry)
    return entry not in crashing

  @classmethod
  def _get_dtypes(cls, test):
    dtypes = original(cls, test)
    if dtypes is None:
      return None
    return [entry for entry in dtypes if allowed(entry)]

  base._get_dtypes = _get_dtypes  # pylint: disable=protected-access


def _build_op_overrides():
  """Expands _OP_OVERRIDE_RULES into per-op DecorateInfo skip lists."""
  import torch  # pylint: disable=g-import-not-at-top
  from torch.testing._internal.opinfo.core import DecorateInfo  # pylint: disable=g-import-not-at-top

  overrides = {}
  for op in _all_op_infos():
    decorators = []
    for selector, dtypes, cls_name, test_name, reason in _OP_OVERRIDE_RULES:
      if not _op_selector_matches(selector, op.full_name):
        continue
      decorators.append(DecorateInfo(
          unittest.skip(f"TorchTPU: {reason}"),
          cls_name,
          test_name,
          dtypes=(
              tuple(getattr(torch, n) for n in dtypes) if dtypes else None
          ),
      ))
    if decorators:
      overrides[op.full_name] = decorators
  return overrides


def install(module):
  """Applies the TorchTPU test configuration for one upstream module.

  Call after importing torch_tpu and before importing the upstream module
  (``instantiate_device_type_tests`` consumes it at import time).
  """
  from torch.testing._internal import common_device_type  # pylint: disable=g-import-not-at-top

  base = common_device_type.PrivateUse1TestBase
  # Uniform "<Generic>TPU" class names and correct op_overrides device
  # matching for every class (see the module docstring).
  base.setUpClass()

  crashing = _crashing_dtypes(module)
  _declare_backend_dtype_support(crashing)
  _install_dtypes_decorator_filter(crashing)
  base.set_test_configs(
      op_overrides=_build_op_overrides(),
      op_allowlist=_OP_ALLOWLIST,
      test_exclusions=_TEST_EXCLUSIONS,
  )
