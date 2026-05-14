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

"""Utilities to work with the ActivationTracer.

This module contains a pretty printer function and a tool to
replay the collected data and modules on CPU to check for numerical differences.
"""
from collections.abc import Sequence
import copy
from typing import Any, TypeVar, cast
import torch
from torch_tpu._internal.utils import utils


def pformat_op_tracer(tracer: utils.OpTracer) -> str:
  """Pretty-formats the collected data from OpTracer.

  Args:
    tracer: The OpTracer instance containing the collected data.

  Returns:
    A string representation of the collected data.
  """
  return tracer._pformat()  # pylint: disable=protected-access


def _pformat_value(
    prefix: str,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    output: Any | None = None,
) -> list[str]:
  """Pretty-formats the collected data from ActivationTracer.

  Args:
    prefix: The prefix to use for each line.
    args: The arguments to the module.
    kwargs: The keyword arguments to the module.
    output: The output of the module.

  Returns:
    A list of strings representing the formatted data.
  """
  args = args or ()
  kwargs = kwargs or {}
  output = output if output is not None else ()

  lines = []

  def _pformat_arg(key, item):
    if isinstance(item, torch.Tensor):
      summary = utils.get_tensor_summary(item)
    else:
      summary = type(item).__name__
    lines.append("{}Args[{}]:   {}".format(prefix, key, summary))

  def _pformat_output(item):
    if isinstance(item, torch.Tensor):
      summary = utils.get_tensor_summary(item)
    else:
      summary = type(item).__name__
    lines.append("{}Output:    {}".format(prefix, summary))

  def _pformat_outputs(idx, item):
    if isinstance(item, torch.Tensor):
      summary = utils.get_tensor_summary(item)
    else:
      summary = type(item).__name__
    lines.append("{}Output[{}]: {}".format(prefix, idx, summary))

  # Args is always a list, per the definition of
  # torch.nn.Module.register_forward_pre_hook / register_forward_hook.
  for idx, item in enumerate(args):
    _pformat_arg(idx, item)

  for key, value in kwargs.items():
    _pformat_arg(key, value)

  # Return values are frequently tensors, tuples, or lists of tensors.
  # TODO: Support HuggingFace ModelOutput types.
  if isinstance(output, (list, tuple)):
    for idx, item in enumerate(output):
      _pformat_outputs(idx, item)
  else:
    _pformat_output(output)

  return lines


def link_events(
    pre_log: Sequence[utils.Event], log: Sequence[utils.Event]
) -> None:
  """Adds 'pre_event' and 'post_event' keys to link events.

  Args:
    pre_log: The pre-forward hook log from ActivationTracer.
    log: The forward log from ActivationTracer.

  Raises:
    RuntimeError: If we fail to link all pre and post events.
  """
  if "post_event" in pre_log[0]:
    return

  pre_item_stack = []
  post_idx = 0
  for pre_item in pre_log:
    pre_item_stack.append(pre_item)
    while (
        pre_item_stack
        and pre_item_stack[-1]["module"] is log[post_idx]["module"]
    ):
      popped_pre_item = pre_item_stack.pop()
      post_item = log[post_idx]
      popped_pre_item["post_event"] = post_item
      post_item["pre_event"] = popped_pre_item
      post_idx += 1
  if post_idx != len(log) or pre_item_stack:
    raise RuntimeError(
        "Failed to link all pre and post events,"
        f" post_idx={post_idx}, len(log)={len(log)},"
        f" stack_size={len(pre_item_stack)}"
    )


def pformat_activation_tracer(tracer) -> str:
  """Pretty-formats the collected data from ActivationTracer.

  TODO: Add time information to the output.

  Args:
    tracer: The ActivationTracer instance containing the collected data.

  Returns:
    A string representation of the collected data.

  Raises:
    RuntimeError: the implementation is incorrect and the internal
      data structures are not consistent.
  """
  if len(tracer.forward_pre_log) != len(tracer.forward_log):
    raise RuntimeError(
        "Pre-hook data and hook data are not the same length. "
        f"{len(tracer.forward_pre_log)=}, {len(tracer.forward_log)=}"
    )
  link_events(tracer.forward_pre_log, tracer.forward_log)

  lines = ["ActivationTracer Collected Data:"]

  pre_item_stack = []
  post_idx = 0  # This is a pointer into the list for traversal.

  for pre_item in tracer.forward_pre_log:
    module = pre_item["module"]
    name = module.__class__.__name__
    args = pre_item["args"]
    kwargs = pre_item["kwargs"]
    depth = pre_item["depth"]

    prefix = "|  " * depth
    deeper_prefix = "|  " * (depth + 1)

    lines.append("{}+- {} (#{})".format(prefix, name, pre_item["idx"]))
    val_lines = _pformat_value(deeper_prefix, args, kwargs)
    lines.extend(val_lines)
    lines.append("{}|".format(prefix))
    pre_item_stack.append(pre_item)

    # Process completed modules from forward_log.
    while pre_item_stack[-1]["post_event"] is tracer.forward_log[post_idx]:
      popped_pre_item = pre_item_stack.pop()
      post_item = popped_pre_item["post_event"]
      module = post_item["module"]
      name = type(module).__name__
      output = post_item["output"]
      depth = post_item["depth"]
      prefix = "|  " * depth
      deeper_prefix = "|  " * (depth + 1)

      lines.extend(_pformat_value(deeper_prefix, output=output))
      lines.append(prefix + "+- {} END (#{})".format(name, post_item["idx"]))
      if not pre_item_stack:
        break
      lines.append(prefix)
      post_idx += 1

  return "\n".join(lines)


def replay_log(
    log: Sequence[utils.Event], device: str
) -> Sequence[utils.Event | Exception]:
  """Reruns the modules and inputs from the forward log.

  Although it regenerates a post-order traveral of the computation DAG, it does
  so by iterating over the log (which is a sequence). The original log is also
  post-order, but is generated by an actual post-order traversal of the
  computation DAG.

  A reminder that the module tree is not equivalent to the computation DAG,
  because modules can have control flow.

  TODO: Support backward.

  Args:
    log: The forward_log from ActivationTracer.
    device: The device to replay the log on.

  Returns:
    The replayed log, which may include exceptions if the replay fails.
  """
  if log[-1]["idx"] != len(log) * 2 - 1:
    raise ValueError(f"{log[-1]['idx']=} != {(len(log)*2-1)=}.")

  log = tuple(log)  # Ignore mutations to the log.

  new_log = []
  for event in log:
    module = copy.deepcopy(event["module"].to(device))
    args = copy.copy(event["args"])
    kwargs = copy.copy(event["kwargs"])
    args = tuple(
        a.to(device) if isinstance(a, torch.Tensor) else a for a in args
    )
    kwargs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
    }
    try:
      output = module(*args, **kwargs)
      item = dict(module=module, args=args, kwargs=kwargs, output=output)
    except Exception as e:  # pylint: disable=broad-except
      item = e

    new_log.append(item)

  return new_log


def _helper_pformat_module_header(
    lines: list[str],
    event: dict[str, Any],
    start_time: float,
    pre_event: dict[str, Any],
) -> None:
  """Pretty-formats a single module.

  Args:
    lines: The list of lines to append to.
    event: The event to pretty-format.
    start_time: The start time of the model forward pass.
    pre_event: The pre-forward event for the module.
  """
  module = event["module"]
  name = type(module).__name__
  args = event["args"]
  kwargs = event["kwargs"]
  output = event["output"]
  depth = event["depth"]
  pre_idx = pre_event["idx"]
  post_idx = event["idx"]
  zero_based_start_time = pre_event["time"] - start_time
  end_time = event["time"] - start_time
  duration = event["time"] - pre_event["time"]

  lines.append(f"--- Module: {name} (#{pre_idx}/#{post_idx}) ---")
  lines.append(f"  Depth:         {depth}")
  lines.append(f"  Start time:    {zero_based_start_time}")
  lines.append(f"  End time:      {end_time}")
  lines.append(f"  Duration:      {duration}")
  lines.extend(_pformat_value("  Original:      ", args, kwargs))
  lines.extend(_pformat_value("  Original:      ", output=output))


# Enforces that both output and cpu_output are the same type.
_TensorOrTensors = TypeVar(
    "_TensorOrTensors", torch.Tensor, Sequence[torch.Tensor]
)


def _helper_render_delta(
    lines: list[str],
    output: _TensorOrTensors,
    cpu_output: _TensorOrTensors,
) -> None:
  """Renders the delta between the CPU and TPU outputs.

  Args:
    lines: The list of lines to append to.
    output: The output of the TPU module.
    cpu_output: The output of the CPU module.
  """
  lines.extend(_pformat_value("  CPU:           ", output=cpu_output))

  if isinstance(cpu_output, torch.Tensor):
    delta = cpu_output - cast(torch.Tensor, output).cpu()
    lines.extend(_pformat_value("  Delta:         ", output=delta))
  elif isinstance(cpu_output, (list, tuple)):
    for a, b in zip(cpu_output, output):
      if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        lines.append("  Delta: n/a. Some items of list/tuple aren't tensors.")
        return
    deltas = [a - b.cpu() for a, b in zip(cpu_output, output)]
    lines.extend(_pformat_value("  Delta:         ", output=deltas))
  else:
    lines.append("  Delta:         n/a. Not tensor, list, or tuple.")


def pformat_replay(
    log: Sequence[utils.Event],
    pre_log: Sequence[utils.Event],
    replayed_log: Sequence[utils.Event | Exception],
) -> str:
  """Pretty-formats diff of ActivationTracer forward_log and its replay.

  Args:
    log: The forward_log from ActivationTracer.
    pre_log: The forward_pre_log from ActivationTracer.
    replayed_log: The replayed log from replay_log().

  Returns:
    A string representation of the collected data.

  Raises:
    RuntimeError: If a module instance appears more than once in the log.
      This is a reasonable but unsupported use case.
  """
  lines = ["ActivationTracer Collected Data:"]
  if len(log) != len(replayed_log):
    raise RuntimeError(f"{len(log)=} != {len(replayed_log)=}")

  link_events(pre_log, log)
  start_time = pre_log[0]["time"]

  for event, replayed_event in zip(log, replayed_log):
    pre_event = event["pre_event"]
    _helper_pformat_module_header(lines, event, start_time, pre_event)
    if isinstance(replayed_event, Exception):
      lines.append(f"  Replay Error:  {replayed_event}")
      continue

    try:
      _helper_render_delta(lines, event["output"], replayed_event["output"])
    except TypeError:
      lines.append("  CPU:           n/a. Some inputs aren't tensors.")
      lines.append("  Delta:         n/a. Some inputs aren't tensors.")

  return "\n".join(lines)
