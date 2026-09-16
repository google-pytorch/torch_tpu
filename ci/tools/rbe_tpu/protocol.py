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

"""Object layout and message shapes shared by the agent and the worker.

Every object in the bucket has exactly one writer:

  fleet/<tpu>.json            the agent on that TPU
  bindings/<tpu>.json         the worker currently holding that TPU
  blobs/<sha256>              any worker, and the content decides the name
  mailbox/<tpu>/work/<id>     the worker bound to that TPU
  mailbox/<tpu>/running/<id>  the agent on that TPU
  mailbox/<tpu>/result/<id>   the agent on that TPU
  mailbox/<tpu>/out/<id>/...  the agent on that TPU

Single-writer is what keeps the hot path free of locking. The one place
two parties contend is `bindings/`, and that is settled by a generation
precondition rather than by agreement.
"""

from __future__ import annotations

import dataclasses
import json
import uuid

# How often the agent rewrites its fleet entry. This doubles as the
# liveness heartbeat, so it is much faster than the binding refresh.
AGENT_HEARTBEAT_SECONDS = 15.0

# A fleet entry older than this is treated as a dead agent. Twelve missed
# beats, giving generous headroom during heavy compilation or test runs.
FLEET_STALE_SECONDS = 180.0

# How often a worker extends its binding, and how long a binding survives
# without an extension. The binding is a slow-moving lock: it is taken
# once at boot, not once per action.
BINDING_REFRESH_SECONDS = 60.0
BINDING_TTL_SECONDS = 300.0

# The agent's idle poll for new work.
AGENT_POLL_SECONDS = 1.0

# The worker polls fast at first, because short tests are common, then
# backs off for the long tail.
WORKER_POLL_FAST_SECONDS = 1.0
WORKER_POLL_FAST_WINDOW_SECONDS = 30.0
WORKER_POLL_SLOW_SECONDS = 5.0

# How long an action waits for a binding before giving up. A TPU being
# down costs throughput; it should not cost a red test.
BIND_WAIT_TIMEOUT_SECONDS = 600.0

# How many different TPUs one action will try before failing. Bounded so
# a fleet-wide outage fails loudly instead of spinning.
MAX_DISPATCH_ATTEMPTS = 3

# Consecutive infrastructure-looking failures before the agent stops and
# probes the chip. One is a flake; two in a row is a pattern.
INFRA_FAILURES_BEFORE_PROBE = 2

# Quarantine lasts at least this long, doubling for each repeat so a
# chip that keeps breaking stops flapping in and out of the fleet.
QUARANTINE_MIN_SECONDS = 120.0
QUARANTINE_MAX_SECONDS = 3600.0

# Consecutive clean probes needed to leave quarantine.
PROBES_TO_RECOVER = 3

# How many times an agent will start the same item. An agent that keeps
# dying partway through one item would otherwise retry it forever.
MAX_ITEM_ATTEMPTS = 3

STATE_HEALTHY = "healthy"
STATE_QUARANTINED = "quarantined"


def fleet_path(tpu: str) -> str:
  return f"fleet/{tpu}.json"


def binding_path(tpu: str) -> str:
  return f"bindings/{tpu}.json"


def blob_path(digest: str) -> str:
  return f"blobs/{digest}"


def work_path(tpu: str, item_id: str) -> str:
  return f"mailbox/{tpu}/work/{item_id}.json"


def running_path(tpu: str, item_id: str) -> str:
  return f"mailbox/{tpu}/running/{item_id}.json"


def result_path(tpu: str, item_id: str) -> str:
  return f"mailbox/{tpu}/result/{item_id}.json"


def output_path(tpu: str, item_id: str, name: str) -> str:
  return f"mailbox/{tpu}/out/{item_id}/{name}"


def work_prefix(tpu: str) -> str:
  return f"mailbox/{tpu}/work/"


def running_prefix(tpu: str) -> str:
  return f"mailbox/{tpu}/running/"


FLEET_PREFIX = "fleet/"


def tpu_from_fleet_path(path: str) -> str:
  """Recovers the TPU name from a fleet object name."""
  return path[len(FLEET_PREFIX) : -len(".json")]


def new_item_id(now: float) -> str:
  """Builds a work item id that sorts by submission time.

  The agent is the only reader of its own mailbox and does not need the
  ordering, but it makes a directory listing readable when something has
  gone wrong.

  Args:
    now: Unix timestamp of submission.

  Returns:
    A unique, time-ordered identifier.
  """
  return f"{int(now * 1e6):019d}-{uuid.uuid4().hex[:12]}"


def _dumps(payload: dict) -> bytes:
  return json.dumps(payload, sort_keys=True).encode("utf-8")


@dataclasses.dataclass(frozen=True)
class FleetEntry:
  """What an agent publishes about itself and its chip."""

  tpu: str
  agent_id: str
  updated_at: float
  state: str = STATE_HEALTHY
  reason: str = ""
  detector: str = ""
  quarantine_count: int = 0
  quarantined_until: float = 0.0

  def is_fresh(self, now: float) -> bool:
    """Reports whether the agent has heartbeat recently enough."""
    return now - self.updated_at <= FLEET_STALE_SECONDS

  def is_available(self, now: float) -> bool:
    """Reports whether this TPU may be bound or dispatched to."""
    return self.is_fresh(now) and self.state == STATE_HEALTHY

  def encode(self) -> bytes:
    return _dumps(dataclasses.asdict(self))

  @classmethod
  def decode(cls, payload: dict) -> FleetEntry:
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in fields})


@dataclasses.dataclass(frozen=True)
class Binding:
  """A worker's claim on one TPU."""

  tpu: str
  worker_id: str
  bound_at: float
  expires_at: float

  def is_expired(self, now: float) -> bool:
    return now >= self.expires_at

  def encode(self) -> bytes:
    return _dumps(dataclasses.asdict(self))

  @classmethod
  def decode(cls, payload: dict) -> Binding:
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in fields})


@dataclasses.dataclass(frozen=True)
class WorkItem:
  """One test to run on one chip."""

  item_id: str
  tpu: str
  binding_generation: int
  argv: list[str]
  workdir: str
  timeout_seconds: float
  env: dict[str, str] = dataclasses.field(default_factory=dict)
  layers: list[str] = dataclasses.field(default_factory=list)
  submitted_at: float = 0.0

  def encode(self) -> bytes:
    return _dumps(dataclasses.asdict(self))

  @classmethod
  def decode(cls, payload: dict) -> WorkItem:
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in fields})


@dataclasses.dataclass(frozen=True)
class Result:
  """What the agent reports back when an item finishes.

  `infra_failure` is deliberately a field rather than a reserved exit
  code. A test binary may exit with any status it likes, so overloading
  one of them to mean "the chip broke" would eventually misfire.
  """

  item_id: str
  exit_code: int
  started_at: float
  finished_at: float
  infra_failure: bool = False
  infra_reason: str = ""
  outputs: list[str] = dataclasses.field(default_factory=list)

  def encode(self) -> bytes:
    return _dumps(dataclasses.asdict(self))

  @classmethod
  def decode(cls, payload: dict) -> Result:
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in fields})
