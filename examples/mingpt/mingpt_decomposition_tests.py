# Copyright 2025 Google LLC
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

from functools import partial
import math
import time

from absl import app
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_tpu._internal.device import _device_module
from examples.mingpt.impl import mingpt

GPT = mingpt.GPT
CausalSelfAttention_pt = mingpt.CausalSelfAttention
Block = mingpt.Block

tpu_backend_name = "tpu"
pt_tpu_device_global = None

MODEL_CMP_RTOL = 1e-3
MODEL_CMP_ATOL = 1e-3
LAYER_PRECISION_RTOL = 1e-5
LAYER_PRECISION_ATOL = 1e-5
MATMUL_ISOLATED_RTOL = 1e-5
MATMUL_ISOLATED_ATOL = 1e-5
STEP_RTOL = 1e-4
STEP_ATOL = 1e-4


def compare_tensors(
    tensor_a,
    tensor_b,
    test_name_suffix,
    platform_a_name,
    platform_b_name,
    rtol=MODEL_CMP_RTOL,
    atol=MODEL_CMP_ATOL,
    exact=False,
    verbose=True,
):
  if tensor_a is None and tensor_b is None:
    if verbose:
      print(
          f"\n--- Skipping Comparison: {platform_a_name} vs"
          f" {platform_b_name} ({test_name_suffix}) as both tensors are"
          " None ---"
      )
    return True, 0.0, 0.0
  if tensor_a is None or tensor_b is None:
    if verbose:
      print(
          f"\n--- ERROR: Comparison Failure: {platform_a_name} vs"
          f" {platform_b_name} ({test_name_suffix}) - one tensor is None ---"
      )
    if verbose:
      print(
          f"  {platform_a_name} is None: {tensor_a is None},"
          f" {platform_b_name} is None: {tensor_b is None}"
      )
    return False, float("nan"), float("nan")

  if verbose:
    print(
        f"\n--- Comparing: {platform_a_name} vs"
        f" {platform_b_name} ({test_name_suffix}) ---"
    )
  if tensor_a.shape != tensor_b.shape:
    if verbose:
      print(
          f"  Shape Mismatch: {platform_a_name}: {tensor_a.shape},"
          f" {platform_b_name}: {tensor_b.shape}"
      )
    return False, float("nan"), float("nan")

  tensor_a_comp = (
      tensor_a.detach().cpu()
      if isinstance(tensor_a, torch.Tensor)
      else torch.tensor(tensor_a).cpu()
  )
  tensor_b_comp = (
      tensor_b.detach().cpu()
      if isinstance(tensor_b, torch.Tensor)
      else torch.tensor(tensor_b).cpu()
  )

  if tensor_a_comp.dtype != tensor_b_comp.dtype:
    common_dtype = torch.float32

    if tensor_a_comp.dtype == torch.bool and tensor_b_comp.is_floating_point():
      common_dtype = tensor_b_comp.dtype
    elif (
        tensor_b_comp.dtype == torch.bool and tensor_a_comp.is_floating_point()
    ):
      common_dtype = tensor_a_comp.dtype
    elif (
        tensor_a_comp.dtype == torch.bool and tensor_b_comp.is_signed_integer()
    ):
      common_dtype = torch.promote_types(torch.int64, tensor_b_comp.dtype)
    elif (
        tensor_b_comp.dtype == torch.bool and tensor_a_comp.is_signed_integer()
    ):
      common_dtype = torch.promote_types(torch.int64, tensor_a_comp.dtype)
    elif tensor_a_comp.is_floating_point() or tensor_b_comp.is_floating_point():

      type1 = tensor_a_comp.dtype
      type2 = tensor_b_comp.dtype
      if type1.is_floating_point and type2.is_floating_point:
        common_dtype = torch.promote_types(type1, type2)
      elif type1.is_floating_point:
        common_dtype = type1
      elif type2.is_floating_point:
        common_dtype = type2
      else:
        common_dtype = torch.promote_types(type1, type2)

      if common_dtype == torch.float16 or common_dtype == torch.bfloat16:
        common_dtype = torch.float32
      elif not common_dtype.is_floating_point and (
          type1.is_floating_point or type2.is_floating_point
      ):
        common_dtype = torch.float32

    if verbose:
      print(
          f"  Promoting dtypes for comparison: {tensor_a_comp.dtype} and"
          f" {tensor_b_comp.dtype} to {common_dtype}"
      )
    tensor_a_comp = tensor_a_comp.to(common_dtype)
    tensor_b_comp = tensor_b_comp.to(common_dtype)

  is_exact_comp = exact
  passed = False
  if is_exact_comp:
    passed = torch.equal(tensor_a_comp, tensor_b_comp)
  else:
    try:
      passed = torch.allclose(
          tensor_a_comp, tensor_b_comp, rtol=rtol, atol=atol
      )
    except RuntimeError as e:
      if verbose:
        print(
            f"  torch.allclose failed with RuntimeError: {e}. Comparing dtypes:"
            f" {tensor_a_comp.dtype} vs {tensor_b_comp.dtype}"
        )
      passed = False

  if verbose:
    print(
        f"  Comparison PASSED (rtol={rtol:.1e}, atol={atol:.1e},"
        f" exact={is_exact_comp}): {passed}"
    )
  max_abs_diff_val, max_rel_diff_val = 0.0, 0.0

  if not passed and verbose:
    if tensor_a_comp.is_floating_point() and tensor_b_comp.is_floating_point():
      abs_diff = torch.abs(tensor_a_comp - tensor_b_comp)
      if abs_diff.numel() > 0:
        max_abs_diff_val = torch.max(abs_diff).item()

      abs_b_val = torch.abs(tensor_b_comp)
      meaningful_denom_threshold = max(
          torch.finfo(tensor_b_comp.dtype).eps * 100
          if tensor_b_comp.is_floating_point()
          else 1e-20,
          1e-12,
      )

      rel_diff_elements = torch.zeros_like(abs_diff, dtype=torch.float32)
      meaningful_indices = abs_b_val > meaningful_denom_threshold

      if meaningful_indices.any():

        rel_diff_elements[meaningful_indices] = abs_diff[meaningful_indices].to(
            torch.float32
        ) / abs_b_val[meaningful_indices].to(torch.float32)
        if rel_diff_elements[meaningful_indices].numel() > 0:
          max_rel_diff_val = torch.max(
              rel_diff_elements[meaningful_indices]
          ).item()
        else:
          max_rel_diff_val = float("inf") if max_abs_diff_val > atol else 0.0
      else:
        max_rel_diff_val = float("inf") if max_abs_diff_val > atol else 0.0

      print(f"  Max absolute difference: {max_abs_diff_val:.6e}")
      print(
          f"  Max relative difference: {max_rel_diff_val:.6e} (calculated where"
          f" |B| > {meaningful_denom_threshold:.1e})"
      )

      diff_indices_cond = (
          torch.logical_not(
              torch.isclose(tensor_a_comp, tensor_b_comp, rtol=rtol, atol=atol)
          )
          if not is_exact_comp
          else (tensor_a_comp != tensor_b_comp)
      )
      diff_indices = torch.nonzero(diff_indices_cond)
      if diff_indices.numel() > 0:
        print(
            f"  First few differing values ({platform_a_name} vs"
            f" {platform_b_name}) (Total:"
            f" {diff_indices.shape[0]}/{tensor_b_comp.numel()}):"
        )
        for i in range(min(5, diff_indices.shape[0])):
          idx = tuple(diff_indices[i].tolist())
          val_a, val_b = tensor_a_comp[idx].item(), tensor_b_comp[idx].item()
          abs_d = abs(val_a - val_b)
          rel_d_denom = abs_b_val[idx].item()
          rel_d = (
              abs_d / abs(rel_d_denom)
              if abs(rel_d_denom) > meaningful_denom_threshold
              else (float("inf") if abs_d > atol else 0.0)
          )
          print(
              f"    at {idx}: {platform_a_name}={val_a:.6e},"
              f" {platform_b_name}={val_b:.6e}, AbsD={abs_d:.3e},"
              f" RelD={rel_d:.3e}"
          )
    elif is_exact_comp and not passed:
      diff_indices = torch.nonzero(tensor_a_comp != tensor_b_comp)
      if diff_indices.numel() > 0:
        print(
            "  First few differing integer/boolean values"
            f" ({platform_a_name} vs {platform_b_name}):"
        )
        for i in range(min(5, diff_indices.shape[0])):
          idx = tuple(diff_indices[i].tolist())
          print(
              f"    at {idx}: {platform_a_name}={tensor_a_comp[idx].item()},"
              f" {platform_b_name}={tensor_b_comp[idx].item()}"
          )
  return passed, max_abs_diff_val, max_rel_diff_val


def subtest_layernorm_precision(
    x_pt_cpu: torch.Tensor,
    ln_layer_module_cpu: nn.LayerNorm,
    ln_layer_module_tpu: nn.LayerNorm,
    pt_tpu_device: torch.device,
    test_name_prefix: str,
    rtol=LAYER_PRECISION_RTOL,
    atol=LAYER_PRECISION_ATOL,
):
  print(f"\n--- Subtest: {test_name_prefix} (LayerNorm Precision) ---")
  overall_passed = True
  results_cpu = {"y": None, "mean": None, "rstd": None}
  results_tpu_copied_to_cpu = {"y": None, "mean": None, "rstd": None}
  y_output_from_tpu_on_cpu_for_chaining = None

  ln_layer_module_cpu.eval()
  print("  Executing LayerNorm on CPU (PyTorch native)...")
  normalized_shape_cpu = (
      list(ln_layer_module_cpu.normalized_shape)
      if isinstance(ln_layer_module_cpu.normalized_shape, torch.Size)
      else ln_layer_module_cpu.normalized_shape
  )

  try:
    with torch.no_grad():

      y_cpu, mean_cpu, rstd_cpu = torch.native_layer_norm(
          x_pt_cpu,
          normalized_shape_cpu,
          ln_layer_module_cpu.weight,
          ln_layer_module_cpu.bias,
          ln_layer_module_cpu.eps,
      )
    results_cpu["y"], results_cpu["mean"], results_cpu["rstd"] = (
        y_cpu,
        mean_cpu,
        rstd_cpu,
    )
    print(
        f"    PT CPU Y: {y_cpu.shape}, Mean: {mean_cpu.shape}, Rstd:"
        f" {rstd_cpu.shape}"
    )
  except Exception as e:
    print(f"    PyTorch CPU LayerNorm FAILED: {e}")
    overall_passed = False

  if pt_tpu_device and ln_layer_module_tpu and results_cpu["y"] is not None:
    ln_layer_module_tpu.eval()
    print("  Executing LayerNorm on TorchTPU (via backend)...")
    x_pt_tpu = x_pt_cpu.clone().to(pt_tpu_device)
    normalized_shape_tpu = (
        list(ln_layer_module_tpu.normalized_shape)
        if isinstance(ln_layer_module_tpu.normalized_shape, torch.Size)
        else ln_layer_module_tpu.normalized_shape
    )
    try:
      with torch.no_grad():
        y_tpu, mean_tpu, rstd_tpu = torch.native_layer_norm(
            x_pt_tpu,
            normalized_shape_tpu,
            ln_layer_module_tpu.weight,
            ln_layer_module_tpu.bias,
            ln_layer_module_tpu.eps,
        )
      results_tpu_copied_to_cpu["y"] = y_tpu.cpu()
      results_tpu_copied_to_cpu["mean"] = mean_tpu.cpu()
      results_tpu_copied_to_cpu["rstd"] = rstd_tpu.cpu()
      y_output_from_tpu_on_cpu_for_chaining = results_tpu_copied_to_cpu["y"]
      print(
          f"    PT TPU Y: {results_tpu_copied_to_cpu['y'].shape}, Mean:"
          f" {results_tpu_copied_to_cpu['mean'].shape}, Rstd:"
          f" {results_tpu_copied_to_cpu['rstd'].shape}"
      )
    except Exception as e:
      print(f"    TorchTPU LayerNorm FAILED: {e}")
      overall_passed = False
  elif results_cpu["y"] is None:
    print("  Skipping PT TPU LayerNorm as CPU reference failed.")
    overall_passed = False
  else:
    print("  Skipping PT TPU LayerNorm (TPU device or module not available).")
    if pt_tpu_device and ln_layer_module_tpu:
      overall_passed = False

  if (
      results_cpu["y"] is not None
      and results_tpu_copied_to_cpu["y"] is not None
  ):
    print("  Comparing Y (Normalized Output):")
    passed_y, _, _ = compare_tensors(
        results_tpu_copied_to_cpu["y"],
        results_cpu["y"],
        f"{test_name_prefix}_Y",
        "PT_TPU",
        "PT_CPU",
        rtol=rtol,
        atol=atol,
    )
    if not passed_y:
      overall_passed = False

    print("  Comparing Mean:")
    passed_mean, _, _ = compare_tensors(
        results_tpu_copied_to_cpu["mean"],
        results_cpu["mean"],
        f"{test_name_prefix}_Mean",
        "PT_TPU",
        "PT_CPU",
        rtol=rtol,
        atol=atol,
    )
    if not passed_mean:
      overall_passed = False

    print("  Comparing Rstd:")
    passed_rstd, _, _ = compare_tensors(
        results_tpu_copied_to_cpu["rstd"],
        results_cpu["rstd"],
        f"{test_name_prefix}_Rstd",
        "PT_TPU",
        "PT_CPU",
        rtol=rtol,
        atol=atol,
    )
    if not passed_rstd:
      overall_passed = False
  elif results_cpu["y"] is not None and not (
      pt_tpu_device and ln_layer_module_tpu
  ):
    print(
        "  Cannot compare LayerNorm outputs as TPU execution was skipped (TPU"
        " not available)."
    )
  elif results_cpu["y"] is None:
    print("  Cannot compare LayerNorm outputs as CPU reference failed.")

  if overall_passed:
    print(
        f"--- Subtest: {test_name_prefix} (LayerNorm Precision) PASSED"
        f" (rtol={rtol:.1e}, atol={atol:.1e}) ---"
    )
  else:
    print(
        f"!!! Subtest: {test_name_prefix} (LayerNorm Precision) FAILED"
        f" (rtol={rtol:.1e}, atol={atol:.1e}). Check logs. !!!"
    )

  return overall_passed, y_output_from_tpu_on_cpu_for_chaining, results_cpu


def subtest_view_after_transpose_contiguous(
    base_data_tensor_cpu: torch.Tensor,
    transpose_dim0: int,
    transpose_dim1: int,
    view_shape: tuple,
    pt_tpu_device: torch.device,
    test_name_suffix: str,
    rtol=1e-5,
    atol=1e-8,
):
  """Tests the sequence: tensor.transpose(dim0, dim1).contiguous().view(view_shape).cpu()

  This is a common pattern that can expose issues in view metadata handling
  and device-to-host copies.

  Args:
      base_data_tensor_cpu: A CPU tensor with the desired initial data and
        shape.
      transpose_dim0: The first dimension to transpose.
      transpose_dim1: The second dimension to transpose.
      view_shape: The target shape for the .view() operation.
      pt_tpu_device: The TorchTPU device (e.g., torch.device("tpu")).
      test_name_suffix: A string suffix for test naming in logs.
      rtol: Relative tolerance for comparison.
      atol: Absolute tolerance for comparison.

  Returns:
      A tuple: (passed_boolean, cpu_reference_tensor, tpu_result_on_cpu_tensor)
  """
  print(
      f"\n--- Subtest: {test_name_suffix} (View after Transpose then"
      " Contiguous) ---"
  )
  print(f"  Base Tensor Shape (from input data): {base_data_tensor_cpu.shape}")
  print(f"  Transpose Dims: ({transpose_dim0}, {transpose_dim1})")
  print(f"  Target View Shape: {view_shape}")

  overall_passed = True
  view_cpu_reference = None
  view_tpu_copied_to_cpu = None

  print("  Executing CPU path (reference)...")
  try:
    cpu_base = base_data_tensor_cpu.clone().detach()
    print(f"    CPU Base: shape {cpu_base.shape}, dtype {cpu_base.dtype}")

    cpu_transposed = cpu_base.transpose(transpose_dim0, transpose_dim1)
    print(
        f"    CPU Transposed: shape {cpu_transposed.shape}, strides"
        f" {cpu_transposed.stride()}, is_contiguous"
        f" {cpu_transposed.is_contiguous()}"
    )

    cpu_contiguous_after_transpose = cpu_transposed.contiguous()
    print(
        "    CPU Contiguous after Transpose: shape"
        f" {cpu_contiguous_after_transpose.shape}, strides"
        f" {cpu_contiguous_after_transpose.stride()}, is_contiguous"
        f" {cpu_contiguous_after_transpose.is_contiguous()}"
    )

    view_cpu_reference = cpu_contiguous_after_transpose.view(view_shape)
    print(
        f"    CPU View (Reference): shape {view_cpu_reference.shape}, strides"
        f" {view_cpu_reference.stride()}, is_contiguous"
        f" {view_cpu_reference.is_contiguous()}"
    )
  except Exception as e:
    print(f"    CPU Path FAILED: {e}")
    import traceback

    traceback.print_exc()
    overall_passed = False

  if pt_tpu_device and overall_passed:
    print(f"\n  Executing TPU path (device: {pt_tpu_device})...")
    try:
      with torch.no_grad():
        tpu_base = base_data_tensor_cpu.clone().to(pt_tpu_device)
        print(
            f"    TPU Base: shape {tpu_base.shape}, dtype {tpu_base.dtype},"
            f" device {tpu_base.device}"
        )

        tpu_transposed = tpu_base.transpose(transpose_dim0, transpose_dim1)
        print(
            f"    TPU Transposed: shape {tpu_transposed.shape}, strides"
            f" {tpu_transposed.stride()}, is_contiguous"
            f" {tpu_transposed.is_contiguous()}, device {tpu_transposed.device}"
        )

        tpu_contiguous_after_transpose = tpu_transposed.contiguous()
        print(
            "    TPU Contiguous after Transpose: shape"
            f" {tpu_contiguous_after_transpose.shape}, strides"
            f" {tpu_contiguous_after_transpose.stride()}, is_contiguous"
            f" {tpu_contiguous_after_transpose.is_contiguous()}, device"
            f" {tpu_contiguous_after_transpose.device}"
        )

        tpu_view = tpu_contiguous_after_transpose.view(view_shape)
        print(
            f"    TPU View: shape {tpu_view.shape}, strides"
            f" {tpu_view.stride()}, is_contiguous {tpu_view.is_contiguous()},"
            f" device {tpu_view.device}"
        )

        view_tpu_copied_to_cpu = tpu_view.cpu()
        print(
            "    TPU View (Copied to CPU): shape"
            f" {view_tpu_copied_to_cpu.shape}, strides"
            f" {view_tpu_copied_to_cpu.stride()}, is_contiguous"
            f" {view_tpu_copied_to_cpu.is_contiguous()}, device"
            f" {view_tpu_copied_to_cpu.device}"
        )

    except Exception as e:
      print(f"    TPU Path FAILED: {e}")
      import traceback

      traceback.print_exc()
      overall_passed = False
      if hasattr(e, "extra_cuda_debug_state"):
        print(
            f"    TPU Exception Extra Debug State: {e.extra_cuda_debug_state}"
        )

  elif not pt_tpu_device:
    print("  Skipping TPU path as pt_tpu_device is None.")
    overall_passed = False
  elif not overall_passed:
    print("  Skipping TPU path as CPU reference path failed.")

  if (
      overall_passed
      and view_cpu_reference is not None
      and view_tpu_copied_to_cpu is not None
  ):

    passed_comparison, _, _ = compare_tensors(
        view_tpu_copied_to_cpu,
        view_cpu_reference,
        test_name_suffix,
        "PT_TPU",
        "PT_CPU_Ref",
        rtol=rtol,
        atol=atol,
    )
    if not passed_comparison:
      overall_passed = False
  elif overall_passed and (
      view_cpu_reference is None or view_tpu_copied_to_cpu is None
  ):
    print(
        f"  Comparison SKIPPED for {test_name_suffix} due to missing tensors"
        " (TPU execution likely failed)."
    )
    overall_passed = False

  if overall_passed:
    print(
        f"--- Subtest: {test_name_suffix} PASSED (rtol={rtol:.1e},"
        f" atol={atol:.1e}) ---"
    )
  else:
    print(
        f"!!! Subtest: {test_name_suffix} FAILED (rtol={rtol:.1e},"
        f" atol={atol:.1e}). Check logs. !!!"
    )

  return overall_passed, view_cpu_reference, view_tpu_copied_to_cpu


def test_causal_self_attention_stepwise(
    attn_module_cpu: CausalSelfAttention_pt,
    attn_module_tpu: CausalSelfAttention_pt,
    x_input_cpu_ref: torch.Tensor,
    x_input_tpu_path_start: torch.Tensor,
    pt_tpu_device: torch.device,
    test_prefix: str = "CSA_Stepwise.",
    force_identical_inputs_for_first_matmul: bool = False,
    force_identical_inputs_for_qkT_matmul: bool = False,
    force_identical_inputs_for_v_matmul: bool = False,
    force_identical_inputs_for_c_proj: bool = False,
):
  print(f"\n--- Subtest: {test_prefix} (CausalSelfAttention Stepwise) ---")
  print(
      "  force_identical_inputs_for_first_matmul (c_attn):"
      f" {force_identical_inputs_for_first_matmul}"
  )
  print(
      "  force_identical_inputs_for_qkT_matmul:"
      f" {force_identical_inputs_for_qkT_matmul}"
  )
  print(
      "  force_identical_inputs_for_v_matmul:"
      f" {force_identical_inputs_for_v_matmul}"
  )
  print(
      "  force_identical_inputs_for_c_proj:"
      f" {force_identical_inputs_for_c_proj}"
  )

  overall_passed_flag = [True]
  B, T, C = x_input_cpu_ref.size()
  n_head = attn_module_cpu.n_head
  n_embd = attn_module_cpu.n_embd
  hs = n_embd // n_head

  val = {}

  def check(cond, message=""):
    if not cond:
      print(f"!!! {test_prefix} FAILED Check: {message} !!!")
      overall_passed_flag[0] = False
    return cond

  def compare_step(
      step_name, tensor_tpu_cpu, tensor_cpu_ref, rtol=STEP_RTOL, atol=STEP_ATOL
  ):
    if tensor_tpu_cpu is not None and tensor_cpu_ref is not None:
      passed, _, _ = compare_tensors(
          tensor_tpu_cpu,
          tensor_cpu_ref,
          f"{test_prefix}_{step_name}",
          "PT_TPU",
          "PT_CPU",
          rtol=rtol,
          atol=atol,
      )
      check(passed, f"Step {step_name} (rtol={rtol:.1e}, atol={atol:.1e})")
    elif tensor_cpu_ref is None:
      print(f"  Skipping comparison for {step_name} as CPU reference is None.")
      check(False, f"Step {step_name} - CPU reference missing")
    elif tensor_tpu_cpu is None and pt_tpu_device:
      print(
          f"  Skipping comparison for {step_name} as TPU result is None (but"
          " TPU was expected)."
      )
      check(False, f"Step {step_name} - TPU result missing")

  with torch.no_grad():
    val["x_cpu_path"] = x_input_cpu_ref.clone()
    if pt_tpu_device and x_input_tpu_path_start is not None:
      val["x_tpu_path_start_on_cpu"] = x_input_tpu_path_start.clone()

    step_name = "1_c_attn"

    val[f"{step_name}_cpu_path_out"] = attn_module_cpu.c_attn(val["x_cpu_path"])

    if pt_tpu_device and attn_module_tpu and "x_tpu_path_start_on_cpu" in val:
      x_for_tpu_c_attn = (
          (
              val["x_cpu_path"]
              if force_identical_inputs_for_first_matmul
              else val["x_tpu_path_start_on_cpu"]
          )
          .clone()
          .to(pt_tpu_device)
      )
      val[f"{step_name}_tpu_path_out_on_tpu"] = attn_module_tpu.c_attn(
          x_for_tpu_c_attn
      )
      val[f"{step_name}_tpu_path_out_on_cpu"] = val[
          f"{step_name}_tpu_path_out_on_tpu"
      ].cpu()
      rtol_c, atol_c = (
          (MATMUL_ISOLATED_RTOL, MATMUL_ISOLATED_ATOL)
          if force_identical_inputs_for_first_matmul
          else (STEP_RTOL, STEP_ATOL)
      )
      compare_step(
          step_name,
          val[f"{step_name}_tpu_path_out_on_cpu"],
          val[f"{step_name}_cpu_path_out"],
          rtol=rtol_c,
          atol=atol_c,
      )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    ops_after_c_attn = [
        "2_q_split",
        "2_k_split",
        "2_v_split",
        "3_q_rt",
        "3_k_rt",
        "3_v_rt",
    ]

    q_c, k_c, v_c = val["1_c_attn_cpu_path_out"].split(n_embd, dim=2)
    (
        val["2_q_split_cpu_path_out"],
        val["2_k_split_cpu_path_out"],
        val["2_v_split_cpu_path_out"],
    ) = (q_c, k_c, v_c)
    val["3_q_rt_cpu_path_out"] = q_c.view(B, T, n_head, hs).transpose(1, 2)
    val["3_k_rt_cpu_path_out"] = k_c.view(B, T, n_head, hs).transpose(1, 2)
    val["3_v_rt_cpu_path_out"] = v_c.view(B, T, n_head, hs).transpose(1, 2)

    if (
        pt_tpu_device
        and attn_module_tpu
        and "1_c_attn_tpu_path_out_on_tpu" in val
    ):
      q_t, k_t, v_t = val["1_c_attn_tpu_path_out_on_tpu"].split(n_embd, dim=2)
      (
          val["2_q_split_tpu_path_out_on_tpu"],
          val["2_k_split_tpu_path_out_on_tpu"],
          val["2_v_split_tpu_path_out_on_tpu"],
      ) = (q_t, k_t, v_t)
      val["3_q_rt_tpu_path_out_on_tpu"] = q_t.view(B, T, n_head, hs).transpose(
          1, 2
      )
      val["3_k_rt_tpu_path_out_on_tpu"] = k_t.view(B, T, n_head, hs).transpose(
          1, 2
      )
      val["3_v_rt_tpu_path_out_on_tpu"] = v_t.view(B, T, n_head, hs).transpose(
          1, 2
      )

      for op_suf in ops_after_c_attn:

        compare_step(
            op_suf,
            val[op_suf + "_tpu_path_out_on_tpu"].cpu(),
            val[op_suf + "_cpu_path_out"],
            rtol=STEP_RTOL,
            atol=STEP_ATOL,
        )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    step_name = "4_att_raw_scores"

    val[f"{step_name}_cpu_path_out"] = val["3_q_rt_cpu_path_out"] @ val[
        "3_k_rt_cpu_path_out"
    ].transpose(-2, -1)

    if (
        pt_tpu_device
        and attn_module_tpu
        and "3_q_rt_tpu_path_out_on_tpu" in val
        and "3_k_rt_tpu_path_out_on_tpu" in val
    ):
      q_in_tpu = (
          val["3_q_rt_cpu_path_out"].clone().to(pt_tpu_device)
          if force_identical_inputs_for_qkT_matmul
          else val["3_q_rt_tpu_path_out_on_tpu"]
      )
      k_in_tpu = (
          val["3_k_rt_cpu_path_out"].clone().to(pt_tpu_device)
          if force_identical_inputs_for_qkT_matmul
          else val["3_k_rt_tpu_path_out_on_tpu"]
      )
      val[f"{step_name}_tpu_path_out_on_tpu"] = q_in_tpu @ k_in_tpu.transpose(
          -2, -1
      )
      val[f"{step_name}_tpu_path_out_on_cpu"] = val[
          f"{step_name}_tpu_path_out_on_tpu"
      ].cpu()
      rtol_m, atol_m = (
          (MATMUL_ISOLATED_RTOL, MATMUL_ISOLATED_ATOL)
          if force_identical_inputs_for_qkT_matmul
          else (STEP_RTOL, STEP_ATOL)
      )
      compare_step(
          step_name,
          val[f"{step_name}_tpu_path_out_on_cpu"],
          val[f"{step_name}_cpu_path_out"],
          rtol=rtol_m,
          atol=atol_m,
      )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    current_cpu = val["4_att_raw_scores_cpu_path_out"]
    current_tpu_on_tpu = val.get("4_att_raw_scores_tpu_path_out_on_tpu")

    scale = 1.0 / math.sqrt(hs)
    current_cpu = current_cpu * scale
    if current_tpu_on_tpu is not None:
      current_tpu_on_tpu = current_tpu_on_tpu * scale
    compare_step(
        "5_att_scaled",
        current_tpu_on_tpu.cpu() if current_tpu_on_tpu is not None else None,
        current_cpu,
    )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    bias_mask_cpu = attn_module_cpu.bias[:, :, :T, :T]
    current_cpu = current_cpu.masked_fill(bias_mask_cpu == 0, float("-inf"))
    if current_tpu_on_tpu is not None:
      bias_mask_tpu = attn_module_tpu.bias[:, :, :T, :T].to(pt_tpu_device)
      current_tpu_on_tpu = current_tpu_on_tpu.masked_fill(
          bias_mask_tpu == 0, float("-inf")
      )
    compare_step(
        "6_att_masked",
        current_tpu_on_tpu.cpu() if current_tpu_on_tpu is not None else None,
        current_cpu,
    )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    current_cpu = F.softmax(current_cpu, dim=-1)
    if current_tpu_on_tpu is not None:
      current_tpu_on_tpu = F.softmax(current_tpu_on_tpu, dim=-1)
    compare_step(
        "7_att_softmax",
        current_tpu_on_tpu.cpu() if current_tpu_on_tpu is not None else None,
        current_cpu,
    )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    current_cpu = attn_module_cpu.attn_dropout(current_cpu)
    if current_tpu_on_tpu is not None:
      current_tpu_on_tpu = attn_module_tpu.attn_dropout(current_tpu_on_tpu)
    compare_step(
        "8_att_dropout",
        current_tpu_on_tpu.cpu() if current_tpu_on_tpu is not None else None,
        current_cpu,
    )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    v_cpu_in = val["3_v_rt_cpu_path_out"]
    current_cpu = current_cpu @ v_cpu_in

    if current_tpu_on_tpu is not None and "3_v_rt_tpu_path_out_on_tpu" in val:
      v_tpu_in = (
          val["3_v_rt_cpu_path_out"].clone().to(pt_tpu_device)
          if force_identical_inputs_for_v_matmul
          else val["3_v_rt_tpu_path_out_on_tpu"]
      )
      current_tpu_on_tpu = current_tpu_on_tpu @ v_tpu_in
      rtol_v, atol_v = (
          (MATMUL_ISOLATED_RTOL, MATMUL_ISOLATED_ATOL)
          if force_identical_inputs_for_v_matmul
          else (STEP_RTOL, STEP_ATOL)
      )
      compare_step(
          "9_y_intermediate_matmul",
          current_tpu_on_tpu.cpu(),
          current_cpu,
          rtol=rtol_v,
          atol=atol_v,
      )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None

    current_cpu = current_cpu.transpose(1, 2).contiguous().view(B, T, C)
    if current_tpu_on_tpu is not None:
      current_tpu_on_tpu = (
          current_tpu_on_tpu.transpose(1, 2).contiguous().view(B, T, C)
      )
    compare_step(
        "10_y_reshaped_heads",
        current_tpu_on_tpu.cpu() if current_tpu_on_tpu is not None else None,
        current_cpu,
        rtol=1e-5,
        atol=1e-5,
    )
    if not overall_passed_flag[0] and pt_tpu_device:
      return False, None, None


def test_matmul_arange_data(
    config, pt_tpu_device, test_name_prefix="Matmul_Arange_Transpose"
):
  print(f"\n--- Subtest: {test_name_prefix} ---")
  B, H, M, K_dim, N = (
      1,
      1,
      2,
      3,
      4,
  )
  dtype = torch.float32

  lhs_cpu = torch.arange(B * H * M * K_dim, dtype=dtype).reshape(B, H, M, K_dim)

  rhs_orig_cpu = (
      torch.arange(B * H * N * K_dim, dtype=dtype).reshape(B, H, N, K_dim) * 0.5
  )
  rhs_transposed_cpu = rhs_orig_cpu.transpose(-2, -1)

  print(
      f"  {test_name_prefix} LHS (CPU): shape {lhs_cpu.shape}, first few:"
      f" {lhs_cpu.flatten()[:5].tolist()}"
  )
  print(
      f"  {test_name_prefix} RHS_orig (CPU): shape {rhs_orig_cpu.shape}, first"
      f" few: {rhs_orig_cpu.flatten()[:5].tolist()}"
  )
  print(
      f"  {test_name_prefix} RHS_transposed (CPU): shape"
      f" {rhs_transposed_cpu.shape}, first few:"
      f" {rhs_transposed_cpu.flatten()[:5].tolist()}"
  )

  expected_output_cpu = torch.matmul(lhs_cpu, rhs_transposed_cpu)
  print(
      f"  {test_name_prefix} Expected Output (CPU matmul): shape"
      f" {expected_output_cpu.shape}, first few:"
      f" {expected_output_cpu.flatten()[:5].tolist()}"
  )

  output_tpu_cpu = None
  overall_passed = False

  if pt_tpu_device:
    try:
      lhs_tpu = lhs_cpu.clone().to(pt_tpu_device)

      rhs_orig_tpu = rhs_orig_cpu.clone().to(pt_tpu_device)
      rhs_transposed_tpu = rhs_orig_tpu.transpose(-2, -1)

      print(
          f"  {test_name_prefix} LHS_tpu shape: {lhs_tpu.shape}, is_contiguous:"
          f" {lhs_tpu.is_contiguous()}"
      )
      print(
          f"  {test_name_prefix} RHS_orig_tpu shape: {rhs_orig_tpu.shape},"
          f" is_contiguous: {rhs_orig_tpu.is_contiguous()}"
      )
      print(
          f"  {test_name_prefix} RHS_transposed_tpu shape:"
          f" {rhs_transposed_tpu.shape}, is_contiguous:"
          f" {rhs_transposed_tpu.is_contiguous()}"
      )

      output_tpu_on_tpu = torch.matmul(lhs_tpu, rhs_transposed_tpu)
      output_tpu_cpu = output_tpu_on_tpu.cpu()

      print(
          f"  {test_name_prefix} Output (TPU matmul, on CPU): shape"
          f" {output_tpu_cpu.shape}, first few:"
          f" {output_tpu_cpu.flatten()[:5].tolist()}"
      )

      passed, abs_diff, rel_diff = compare_tensors(
          output_tpu_cpu,
          expected_output_cpu,
          test_name_prefix,
          "TPU",
          "CPU_Ref",
          rtol=MATMUL_ISOLATED_RTOL,
          atol=MATMUL_ISOLATED_ATOL,
      )
      overall_passed = passed
      if passed:
        print(
            f"--- Subtest: {test_name_prefix} PASSED"
            f" (rtol={MATMUL_ISOLATED_RTOL:.1e},"
            f" atol={MATMUL_ISOLATED_ATOL:.1e}) ---"
        )
      else:
        print(
            f"!!! Subtest: {test_name_prefix} FAILED"
            f" (rtol={MATMUL_ISOLATED_RTOL:.1e},"
            f" atol={MATMUL_ISOLATED_ATOL:.1e}) !!!"
        )
    except Exception as e:
      print(f"  {test_name_prefix} TPU execution FAILED with exception: {e}")
      overall_passed = False
  else:
    print(
        f"  Skipping {test_name_prefix} TPU execution as pt_tpu_device is None."
    )

  return overall_passed


def main(argv):
  global pt_tpu_device_global, tpu_backend_name

  torch.utils.rename_privateuse1_backend(tpu_backend_name)
  print(f"PT backend renamed to '{tpu_backend_name}'.")
  torch._register_device_module(tpu_backend_name, _device_module._DeviceModule)
  print(f"Registered Python module for '{tpu_backend_name}'.")
  try:
    getattr(torch, tpu_backend_name)._init_runtime_options()  # pylint: disable=protected-access
    pt_tpu_device_global = torch.device(tpu_backend_name)
    print(
        f"PT Device type: {pt_tpu_device_global.type}, Index:"
        f" {pt_tpu_device_global.index if pt_tpu_device_global.index is not None else 'N/A'}"
    )
  except Exception as e:
    print(f"WARNING: Could not initialize TorchTPU runtime or device: {e}")
    pt_tpu_device_global = None

  torch.set_float32_matmul_precision("highest")
  torch.manual_seed(42)
  np.random.seed(42)

  config = GPT.get_default_config()
  config.model_type = "gpt-micro"
  config.vocab_size = 100
  # known crash if this is too small, due to layouts and tiling
  config.block_size = 1024
  config.n_embd = 32
  config.n_head = 8
  config.attn_pdrop = 0.0
  config.resid_pdrop = 0.0
  config.embd_pdrop = 0.0

  B, T_current, C_embd = 2, 32, config.n_embd
  if T_current > config.block_size:
    T_current = config.block_size

  print(f"\n--- Test Configuration ---")
  print(
      f"  B: {B}, T: {T_current}, C: {C_embd}, n_head: {config.n_head},"
      f" block_size: {config.block_size}"
  )

  model_pt_cpu = (
      Block(config, device=torch.device("cpu")).to(torch.float32).eval()
  )
  model_pt_tpu = None
  if pt_tpu_device_global:
    model_pt_tpu = (
        Block(config, device=pt_tpu_device_global).to(torch.float32).eval()
    )
    model_pt_tpu.load_state_dict(model_pt_cpu.state_dict())
    model_pt_tpu.to(pt_tpu_device_global)
    print("CPU and TPU models (Block instances) created and weights synced.")

  x_pt_cpu_input = torch.randn(B, T_current, C_embd, dtype=torch.float32)
  print(f"\nInput x shape for tests: {x_pt_cpu_input.shape}")

  test_matmul_arange_data(config, pt_tpu_device_global)

  ln_1_cpu_module = model_pt_cpu.ln_1
  ln_1_tpu_module = model_pt_tpu.ln_1 if model_pt_tpu else None
  y_ln_from_tpu_on_cpu_for_chaining, ln_cpu_ref_outputs_dict = None, None
  layernorm_standalone_passed = False
  if ln_1_cpu_module and (not pt_tpu_device_global or ln_1_tpu_module):
    (
        layernorm_standalone_passed,
        y_ln_from_tpu_on_cpu_for_chaining,
        ln_cpu_ref_outputs_dict,
    ) = subtest_layernorm_precision(
        x_pt_cpu_input.clone(),
        ln_1_cpu_module,
        ln_1_tpu_module,
        pt_tpu_device_global,
        "Focused_LN1_Precision",
    )

  if (
      layernorm_standalone_passed
      and ln_cpu_ref_outputs_dict
      and ln_cpu_ref_outputs_dict.get("y") is not None
  ):
    print("\n--- Test: 'Pure' c_attn (TpuAddmm) with IDENTICAL inputs ---")
    x_identical_for_c_attn_cpu = ln_cpu_ref_outputs_dict["y"].clone()
    attn_cpu_mod = model_pt_cpu.attn
    c_attn_out_cpu_pure = attn_cpu_mod.c_attn(x_identical_for_c_attn_cpu)
    c_attn_out_tpu_pure_cpu = None
    if pt_tpu_device_global and model_pt_tpu:
      attn_tpu_mod = model_pt_tpu.attn
      x_identical_for_c_attn_tpu = x_identical_for_c_attn_cpu.clone().to(
          pt_tpu_device_global
      )
      c_attn_out_tpu_pure_tpu = attn_tpu_mod.c_attn(x_identical_for_c_attn_tpu)
      c_attn_out_tpu_pure_cpu = c_attn_out_tpu_pure_tpu.cpu()

    if c_attn_out_tpu_pure_cpu is not None:
      passed_pure_c_attn, _, _ = compare_tensors(
          c_attn_out_tpu_pure_cpu,
          c_attn_out_cpu_pure,
          "Pure_c_attn",
          "TPU_Pure",
          "CPU_Ref",
          rtol=MATMUL_ISOLATED_RTOL,
          atol=MATMUL_ISOLATED_ATOL,
      )
      if passed_pure_c_attn:
        print("--- 'Pure' c_attn test PASSED ---")
      else:
        print("!!! 'Pure' c_attn test FAILED !!!")
    elif pt_tpu_device_global:
      print("!!! 'Pure' c_attn test SKIPPED/FAILED (TPU output None) !!!")

  print(
      "\n--- Test: 'Pure' Q@K.T (TpuMatmulNdInternal) with IDENTICAL Q,K"
      " inputs ---"
  )
  hs_calc = C_embd // config.n_head
  q_pure_cpu = torch.randn(
      B, config.n_head, T_current, hs_calc, dtype=torch.float32
  )
  k_pure_cpu = torch.randn(
      B, config.n_head, T_current, hs_calc, dtype=torch.float32
  )
  att_scores_cpu_pure = q_pure_cpu @ k_pure_cpu.transpose(-2, -1)
  att_scores_tpu_pure_cpu = None
  if pt_tpu_device_global:
    q_pure_tpu = q_pure_cpu.clone().to(pt_tpu_device_global)
    k_pure_tpu = k_pure_cpu.clone().to(pt_tpu_device_global)
    try:
      att_scores_tpu_pure_tpu = torch.matmul(
          q_pure_tpu, k_pure_tpu.transpose(-2, -1)
      )
      att_scores_tpu_pure_cpu = att_scores_tpu_pure_tpu.cpu()
    except Exception as e:
      print(f"  'Pure' Q@K.T matmul on TPU FAILED with exception: {e}")

  if att_scores_tpu_pure_cpu is not None:
    passed_pure_qkT, _, _ = compare_tensors(
        att_scores_tpu_pure_cpu,
        att_scores_cpu_pure,
        "Pure_Q@K.T",
        "TPU_Pure",
        "CPU_Ref",
        rtol=MATMUL_ISOLATED_RTOL,
        atol=MATMUL_ISOLATED_ATOL,
    )
    if passed_pure_qkT:
      print("--- 'Pure' Q@K.T Matmul test PASSED ---")
    else:
      print("!!! 'Pure' Q@K.T Matmul test FAILED !!!")
  elif pt_tpu_device_global:
    print("!!! 'Pure' Q@K.T Matmul test SKIPPED/FAILED (TPU output None) !!!")

  print("PRE CSA TESTS!!!!")
  B_test, nH_test, T_test, hs_test = (
      2,
      config.n_head,
      T_current,
      C_embd // config.n_head,
  )
  C_test = nH_test * hs_test

  base_shape_scenario1 = (B_test, nH_test, T_test, hs_test)
  transpose_dim0_scenario1 = 1
  transpose_dim1_scenario1 = 2

  view_shape_scenario1 = (B_test, T_test, C_test)

  print(
      f"  Scenario 1 - Base Shape: {base_shape_scenario1}, Transpose:"
      f" ({transpose_dim0_scenario1},{transpose_dim1_scenario1}), View Shape:"
      f" {view_shape_scenario1}"
  )

  torch.manual_seed(4242)

  base_data_cpu_s1 = torch.arange(
      float(np.prod(base_shape_scenario1)), dtype=torch.float32
  ).reshape(base_shape_scenario1)
  base_data_cpu_s1 = (
      base_data_cpu_s1 - base_data_cpu_s1.mean()
  ) / base_data_cpu_s1.std()

  view_test_rtol_s1 = 1e-5
  view_test_atol_s1 = 1e-5

  passed_s1, _, _ = subtest_view_after_transpose_contiguous(
      base_data_tensor_cpu=base_data_cpu_s1,
      transpose_dim0=transpose_dim0_scenario1,
      transpose_dim1=transpose_dim1_scenario1,
      view_shape=view_shape_scenario1,
      pt_tpu_device=pt_tpu_device_global,
      test_name_suffix="ViewOp_AttentionReshapeMimic",
      rtol=view_test_rtol_s1,
      atol=view_test_atol_s1,
  )
  if not passed_s1:
    print(
        "!!! ISOLATED VIEW TEST (AttentionReshapeMimic) FAILED. This is likely"
        " the core issue. !!!"
    )
  else:
    print(
        "--- ISOLATED VIEW TEST (AttentionReshapeMimic) PASSED. The issue might"
        " be upstream or different. ---"
    )

  base_shape_scenario2 = (4, 6)
  transpose_dim0_scenario2 = 0
  transpose_dim1_scenario2 = 1

  view_shape_scenario2 = (2, 12)

  print(
      f"  Scenario 2 - Base Shape: {base_shape_scenario2}, Transpose:"
      f" ({transpose_dim0_scenario2},{transpose_dim1_scenario2}), View Shape:"
      f" {view_shape_scenario2}"
  )
  torch.manual_seed(4343)
  base_data_cpu_s2 = torch.arange(
      float(np.prod(base_shape_scenario2)), dtype=torch.float32
  ).reshape(base_shape_scenario2)

  passed_s2, _, _ = subtest_view_after_transpose_contiguous(
      base_data_tensor_cpu=base_data_cpu_s2,
      transpose_dim0=transpose_dim0_scenario2,
      transpose_dim1=transpose_dim1_scenario2,
      view_shape=view_shape_scenario2,
      pt_tpu_device=pt_tpu_device_global,
      test_name_suffix="ViewOp_Simple2D",
      rtol=1e-5,
      atol=1e-5,
  )
  if not passed_s2:
    print("!!! ISOLATED VIEW TEST (Simple2D) FAILED. !!!")

  if (
      layernorm_standalone_passed
      and y_ln_from_tpu_on_cpu_for_chaining is not None
      and ln_cpu_ref_outputs_dict
      and ln_cpu_ref_outputs_dict.get("y") is not None
      and model_pt_cpu.attn
      and (not pt_tpu_device_global or (model_pt_tpu and model_pt_tpu.attn))
  ):
    test_causal_self_attention_stepwise(
        model_pt_cpu.attn,
        model_pt_tpu.attn if model_pt_tpu else None,
        ln_cpu_ref_outputs_dict["y"].clone(),
        y_ln_from_tpu_on_cpu_for_chaining.clone(),
        pt_tpu_device_global,
        test_prefix="AttnStepwise_Propagated.",
        force_identical_inputs_for_first_matmul=False,
        force_identical_inputs_for_qkT_matmul=False,
        force_identical_inputs_for_v_matmul=False,
        force_identical_inputs_for_c_proj=False,
    )

  if (
      layernorm_standalone_passed
      and y_ln_from_tpu_on_cpu_for_chaining is not None
      and ln_cpu_ref_outputs_dict
      and ln_cpu_ref_outputs_dict.get("y") is not None
      and model_pt_cpu.attn
      and (not pt_tpu_device_global or (model_pt_tpu and model_pt_tpu.attn))
  ):
    test_causal_self_attention_stepwise(
        model_pt_cpu.attn,
        model_pt_tpu.attn if model_pt_tpu else None,
        ln_cpu_ref_outputs_dict["y"].clone(),
        y_ln_from_tpu_on_cpu_for_chaining.clone(),
        pt_tpu_device_global,
        test_prefix="AttnStepwise_ForceIdenticalQKT.",
        force_identical_inputs_for_first_matmul=False,
        force_identical_inputs_for_qkT_matmul=True,
        force_identical_inputs_for_v_matmul=False,
        force_identical_inputs_for_c_proj=False,
    )

  if (
      layernorm_standalone_passed
      and y_ln_from_tpu_on_cpu_for_chaining is not None
      and ln_cpu_ref_outputs_dict
      and ln_cpu_ref_outputs_dict.get("y") is not None
      and model_pt_cpu.attn
      and (not pt_tpu_device_global or (model_pt_tpu and model_pt_tpu.attn))
  ):
    test_causal_self_attention_stepwise(
        model_pt_cpu.attn,
        model_pt_tpu.attn if model_pt_tpu else None,
        ln_cpu_ref_outputs_dict["y"].clone(),
        y_ln_from_tpu_on_cpu_for_chaining.clone(),
        pt_tpu_device_global,
        test_prefix="AttnStepwise_AllMatmulsForcedIdentical.",
        force_identical_inputs_for_first_matmul=True,
        force_identical_inputs_for_qkT_matmul=True,
        force_identical_inputs_for_v_matmul=True,
        force_identical_inputs_for_c_proj=True,
    )

  print("\n--- Test: Full Block Operation model(x) (End-to-End) ---")
  block_output_cpu_ref = None
  try:
    with torch.no_grad():
      block_output_cpu_ref = model_pt_cpu(x_pt_cpu_input.clone())
  except Exception as e:
    print(f"  Full Block CPU execution FAILED: {e}")
  block_output_tpu_to_cpu = None
  if pt_tpu_device_global and model_pt_tpu:
    x_for_block_tpu = x_pt_cpu_input.clone().to(pt_tpu_device_global)
    try:
      with torch.no_grad():
        block_output_tpu_on_tpu = model_pt_tpu(x_for_block_tpu)
      block_output_tpu_to_cpu = block_output_tpu_on_tpu.cpu()
    except Exception as e:
      print(f"  Full Block TPU execution FAILED: {e}")

  if block_output_cpu_ref is not None and block_output_tpu_to_cpu is not None:
    passed_tmp, _, _ = compare_tensors(
        block_output_tpu_to_cpu,
        block_output_cpu_ref,
        "Full_Block_EndToEnd",
        "TPU",
        "CPU",
        rtol=MODEL_CMP_RTOL,
        atol=MODEL_CMP_ATOL,
    )
    if passed_tmp:
      print("--- Full Block(x) End-to-End test PASSED ---")
    else:
      print("!!! Full Block(x) End-to-End test FAILED !!!")
  elif pt_tpu_device_global and block_output_cpu_ref is not None:
    print(
        "!!! Full Block(x) End-to-End test FAILED (TPU output None or CPU"
        " failed) !!!"
    )
  elif block_output_cpu_ref is None:
    print("!!! Full Block(x) End-to-End test FAILED (CPU reference failed) !!!")

  print(
      "\n--- Test: 'Pure' Q@K.T (TpuMatmulNdInternal via torch.matmul) with"
      " IDENTICAL Q,K inputs (WITH PYTHON DATA LOG) ---"
  )
  hs_calc = C_embd // config.n_head
  q_pure_cpu = torch.randn(
      B, config.n_head, T_current, hs_calc, dtype=torch.float32
  )
  k_pure_cpu = torch.randn(
      B, config.n_head, T_current, hs_calc, dtype=torch.float32
  )

  print(
      "  PURE_QKT_TEST CPU Q input (first 10 flat):"
      f" {q_pure_cpu.flatten()[:10].tolist()}"
  )
  print(
      "  PURE_QKT_TEST CPU K input (for K.T) (first 10 flat):"
      f" {k_pure_cpu.flatten()[:10].tolist()}"
  )

  rhs_for_matmul_cpu = k_pure_cpu.transpose(-2, -1)
  att_scores_cpu_pure = q_pure_cpu @ rhs_for_matmul_cpu
  expected_output_dims_qkT = list(att_scores_cpu_pure.shape)
  print(
      "  PURE_QKT_TEST CPU Matmul Output (first 10 flat):"
      f" {att_scores_cpu_pure.flatten()[:10].tolist()}"
  )

  att_scores_tpu_pure_cpu = None
  if pt_tpu_device_global:
    q_pure_tpu = q_pure_cpu.clone().contiguous().to(pt_tpu_device_global)

    k_transposed_tpu = (
        rhs_for_matmul_cpu.clone().contiguous().to(pt_tpu_device_global)
    )

    try:

      att_scores_tpu_pure_on_tpu = torch.matmul(q_pure_tpu, k_transposed_tpu)
      att_scores_tpu_pure_cpu = att_scores_tpu_pure_on_tpu.cpu()
      print(
          "  PURE_QKT_TEST TPU Matmul Output (first 10 flat, from CPU):"
          f" {att_scores_tpu_pure_cpu.flatten()[:10].tolist()}"
      )
    except Exception as e:
      print(
          "  'Pure' Q@K.T matmul on TPU (torch.matmul dispatch) FAILED with"
          f" exception: {e}"
      )

  if att_scores_tpu_pure_cpu is not None:
    passed_pure_qkT, _, _ = compare_tensors(
        att_scores_tpu_pure_cpu,
        att_scores_cpu_pure,
        "Pure_Q@K.T_dispatch",
        "TPU_Dispatch",
        "CPU_Ref",
        rtol=MATMUL_ISOLATED_RTOL,
        atol=MATMUL_ISOLATED_ATOL,
    )
    if passed_pure_qkT:
      print("--- 'Pure' Q@K.T Matmul test (torch.matmul dispatch) PASSED ---")
    else:
      print("!!! 'Pure' Q@K.T Matmul test (torch.matmul dispatch) FAILED !!!")
  elif pt_tpu_device_global:
    print(
        "!!! 'Pure' Q@K.T Matmul test (torch.matmul dispatch) SKIPPED/FAILED"
        " (TPU output None) !!!"
    )

  print("\n--- Full Script Finished ---")
  return


if __name__ == "__main__":
  app.run(main)
