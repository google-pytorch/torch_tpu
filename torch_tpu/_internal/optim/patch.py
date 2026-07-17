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
"""Utilities for patching PyTorch optimizers to prevent graph breaks."""

# pylint: disable=protected-access

import functools
import torch
import torch.optim.optimizer as optimizer_module

_original_use_grad_for_differentiable = None
_original_class_steps = {}
_closure_restorations = []


def use_grad_for_differentiable(func):
  """A safe decorator for step() that replaces _use_grad_for_differentiable.

  It avoids inserting dynamo graph breaks.

  Args:
    func: The optimizer step function to decorate.

  Returns:
    The decorated step function.
  """

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    self = args[0]
    prev_grad = torch.is_grad_enabled()
    try:
      torch.set_grad_enabled(self.defaults.get("differentiable", False))
      ret = func(*args, **kwargs)
    finally:
      torch.set_grad_enabled(prev_grad)
    return ret

  return wrapper


def _get_wrapped_function(use_grad_fn):
  """Retrieves the immediate function wrapped by the _use_grad decorator."""
  if (
      hasattr(use_grad_fn, "__closure__")
      and use_grad_fn.__closure__ is not None
  ):
    for cell in use_grad_fn.__closure__:
      contents = cell.cell_contents
      if callable(contents):
        return contents
  return None


def _patch_closure_chain(method, patched_decorator_fn):
  """Recursively traverses closure cells to find and replace _use_grad wrapper.

  Args:
    method: The method to check.
    patched_decorator_fn: The patched decorator function.

  Returns:
    True if the wrapper was found and replaced, False otherwise.
  """
  if hasattr(method, "__closure__") and method.__closure__ is not None:
    for cell in method.__closure__:
      contents = cell.cell_contents
      if callable(contents):
        # This recursive search can potentially unwrap other decorators
        # if they also place the inner function in a closure and are
        # named "step". To make this safer, ideally we would only proceed
        # if 'contents' is confirmed to be a wrapper from the specific
        # optimizer_module._use_grad_for_differentiable decorator.
        code_obj = getattr(contents, "__code__", None)
        if (
            code_obj is not None
            and getattr(code_obj, "co_name", None) == "_use_grad"
        ):
          # Only retrieve the immediate wrapped function to avoid unwrapping
          # other decorators inside the closure.
          wrapped_step = _get_wrapped_function(contents)
          if wrapped_step is not None:
            patched_step = patched_decorator_fn(wrapped_step)
            # Re-apply profile_hook_step wrapper if it was hooked
            if getattr(contents, "hooked", False):
              new_step = torch.optim.Optimizer.profile_hook_step(patched_step)
              new_step.hooked = True
            else:
              new_step = patched_step

            # Save for restoration
            _closure_restorations.append((cell, contents))
            cell.cell_contents = new_step
            return True
        else:
          if _patch_closure_chain(contents, patched_decorator_fn):
            return True
  return False


def patch_optimizer_graph_breaks() -> None:
  """Monkey-patches PyTorch optimizers to avoid graph breaks.

  Removes the explicit `torch._dynamo.graph_break()` inserted by PyTorch's
  `_use_grad_for_differentiable` decorator.
  """
  global _original_use_grad_for_differentiable

  if _original_use_grad_for_differentiable is None:
    _original_use_grad_for_differentiable = (
        optimizer_module._use_grad_for_differentiable
    )

  # 1. Patch the decorator itself for future definitions
  optimizer_module._use_grad_for_differentiable = use_grad_for_differentiable

  # 2. Patch existing subclasses of Optimizer
  def get_all_subclasses(cls):
    subclasses = set(cls.__subclasses__())
    all_subclasses = set(subclasses)
    for c in subclasses:
      all_subclasses.update(get_all_subclasses(c))
    return all_subclasses

  all_opt_classes = get_all_subclasses(torch.optim.Optimizer)

  for opt_class in all_opt_classes:
    # Safe check: only patch if 'step' is directly defined in this
    # subclass's __dict__
    if "step" not in opt_class.__dict__:
      continue

    step_method = opt_class.step
    if (
        getattr(step_method, "__code__", None) is not None
        and step_method.__code__.co_name == "_use_grad"
    ):
      # Only retrieve the immediate wrapped function to avoid unwrapping
      # other decorators inside the closure.
      wrapped_step = _get_wrapped_function(step_method)
      if wrapped_step is not None:
        if opt_class not in _original_class_steps:
          _original_class_steps[opt_class] = step_method

        patched_step = use_grad_for_differentiable(wrapped_step)
        # Re-apply profile_hook_step wrapper if it was hooked
        if getattr(step_method, "hooked", False):
          new_step = torch.optim.Optimizer.profile_hook_step(patched_step)
          new_step.hooked = True
        else:
          new_step = patched_step
        opt_class.step = new_step
    else:
      # If step is wrapped by other decorators, patch the inner _use_grad in
      # the closure chain
      _patch_closure_chain(step_method, use_grad_for_differentiable)


def unpatch_optimizer_graph_breaks() -> None:
  """Restores the original PyTorch optimizer decorator and class methods."""
  global _original_use_grad_for_differentiable

  # 1. Restore the decorator
  if _original_use_grad_for_differentiable is not None:
    optimizer_module._use_grad_for_differentiable = (
        _original_use_grad_for_differentiable
    )
    _original_use_grad_for_differentiable = None

  # 2. Restore class methods
  for opt_class, original_step in _original_class_steps.items():
    opt_class.step = original_step
  _original_class_steps.clear()

  # 3. Restore mutated closure cells
  for cell, original_contents in _closure_restorations:
    cell.cell_contents = original_contents
  _closure_restorations.clear()
