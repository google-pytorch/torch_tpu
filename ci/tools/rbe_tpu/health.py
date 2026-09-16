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

"""Deciding whether the chip under an agent is still usable.

The rule that shapes everything here: a quarantine that fires wrongly
removes a working chip from a 28-chip fleet, so output text alone never
quarantines anything. Suspicious output only asks the probe to look. The
probe decides. The one exception is a missing device node, which is not
ambiguous.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import shlex
import subprocess
import time

from ci.tools.rbe_tpu import protocol

# Env var holding a shell command that exits 0 when the chip is usable.
# The default probe below is deliberately weak, and swapping in a real
# libtpu handshake later should not mean touching this file.
PROBE_COMMAND_ENV = "TORCH_TPU_RELAY_CHIP_PROBE"

PROBE_TIMEOUT_SECONDS = 60.0

# How often an idle agent looks at its chip unprompted.
IDLE_PROBE_SECONDS = 60.0

_DEVICE_GLOBS = ("/dev/accel*", "/dev/vfio/[0-9]*")

# Strings that mean the runtime could not get to the chip at all. A test
# asserting on its own tensors cannot plausibly print these. Anything
# that a legitimate test failure could also produce is left out on
# purpose: RESOURCE_EXHAUSTED is an ordinary out-of-memory, and
# DEADLINE_EXCEEDED is an ordinary slow test.
_INFRA_MARKERS = (
    "Failed to initialize TPU",
    "TPU initialization failed",
    "Unable to initialize backend 'tpu'",
    "libtpu.so: cannot open shared object",
    "No visible TPU devices",
    "no TPU devices found",
    "Failed to open /dev/vfio",
    "Failed to open TPU device",
    "vfio: error",
)


@dataclasses.dataclass(frozen=True)
class ProbeResult:
  """What a probe saw."""

  healthy: bool
  detector: str
  detail: str = ""


def classify_output(text: str) -> str:
  """Looks for evidence that the chip, not the test, is at fault.

  Args:
    text: Combined stdout and stderr from a finished item.

  Returns:
    The marker that matched, or an empty string when nothing did. A
    non-empty result is a reason to probe, never a reason to quarantine.
  """
  for marker in _INFRA_MARKERS:
    if marker in text:
      return marker
  return ""


def device_node_probe() -> ProbeResult:
  """Checks that the accelerator device nodes are present.

  Presence only. Opening the device would be a stronger check and a
  wrong one: while a test is running the chip is legitimately busy, and
  a failed open would report a healthy chip as broken.

  Returns:
    A healthy result when at least one node matches.
  """
  found = [path for pattern in _DEVICE_GLOBS for path in glob.glob(pattern)]
  if found:
    return ProbeResult(True, "device_nodes", f"found {len(found)} node(s)")
  return ProbeResult(
      False, "device_nodes", f"no device nodes matching {_DEVICE_GLOBS}"
  )


def command_probe(command: str) -> ProbeResult:
  """Runs an operator-supplied probe command.

  Args:
    command: Shell words; exit status 0 means the chip is usable.

  Returns:
    The probe outcome, with the command's own output as the detail.
  """
  try:
    finished = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
        check=False,
    )
  except subprocess.TimeoutExpired:
    return ProbeResult(False, "probe_command", "probe timed out")
  except OSError as error:
    return ProbeResult(False, "probe_command", f"probe failed: {error}")
  detail = (finished.stdout + finished.stderr).strip()[:500]
  return ProbeResult(finished.returncode == 0, "probe_command", detail)


def default_probe() -> ProbeResult:
  """Runs the configured probe, falling back to the device node check."""
  command = os.environ.get(PROBE_COMMAND_ENV, "").strip()
  if command:
    return command_probe(command)
  return device_node_probe()


class HealthMonitor:
  """Tracks one chip's health and decides when to pull it from service.

  The agent owns this. It reports transitions rather than acting on
  them, so the caller stays in charge of the revocation sequence.
  """

  def __init__(self, probe=default_probe, *, time_fn=time.time):
    self._probe = probe
    self._time_fn = time_fn
    self._consecutive_infra = 0
    self._consecutive_clean_probes = 0
    self._last_probe_at = 0.0
    self.quarantined = False
    self.reason = ""
    self.detector = ""
    self.quarantine_count = 0
    self.quarantined_until = 0.0

  def check_at_boot(self) -> bool:
    """Probes before taking any work. Returns True when healthy."""
    return self._apply(self._run_probe())

  def observe_success(self) -> None:
    """Records an item that finished without infrastructure trouble."""
    self._consecutive_infra = 0

  def observe_failure(self, text: str) -> bool:
    """Records a finished item and decides whether to probe.

    Args:
      text: Combined output from the item.

    Returns:
      True when the chip is still usable, False when it was just
      quarantined.
    """
    marker = classify_output(text)
    if not marker:
      self._consecutive_infra = 0
      return not self.quarantined
    self._consecutive_infra += 1
    self.reason = marker
    if self._consecutive_infra < protocol.INFRA_FAILURES_BEFORE_PROBE:
      return not self.quarantined
    return self._apply(self._run_probe())

  def poll(self) -> bool:
    """Runs the idle-time and recovery probes on a timer.

    Returns:
      True when the chip is usable right now.
    """
    now = self._time_fn()
    if now - self._last_probe_at < IDLE_PROBE_SECONDS:
      return not self.quarantined
    return self._apply(self._run_probe())

  def _run_probe(self) -> ProbeResult:
    self._last_probe_at = self._time_fn()
    return self._probe()

  def _apply(self, result: ProbeResult) -> bool:
    """Feeds a probe result into the state machine."""
    if result.healthy:
      self._consecutive_clean_probes += 1
      self._consecutive_infra = 0
      if self.quarantined and self._may_recover():
        self._recover()
      return not self.quarantined
    self._consecutive_clean_probes = 0
    if not self.quarantined:
      self._quarantine(result)
    return False

  def _may_recover(self) -> bool:
    if self._consecutive_clean_probes < protocol.PROBES_TO_RECOVER:
      return False
    return self._time_fn() >= self.quarantined_until

  def _recover(self) -> None:
    self.quarantined = False
    self.reason = ""
    self.detector = ""
    self.quarantined_until = 0.0
    self._consecutive_infra = 0

  def _quarantine(self, result: ProbeResult) -> None:
    self.quarantined = True
    self.quarantine_count += 1
    self.detector = result.detector
    self.reason = result.detail or self.reason or "probe failed"
    self.quarantined_until = self._time_fn() + self._hold_seconds()
    self._consecutive_clean_probes = 0

  def _hold_seconds(self) -> float:
    """Returns how long this quarantine lasts, doubling on each repeat."""
    span = protocol.QUARANTINE_MIN_SECONDS * (
        2 ** (self.quarantine_count - 1)
    )
    return min(span, protocol.QUARANTINE_MAX_SECONDS)
