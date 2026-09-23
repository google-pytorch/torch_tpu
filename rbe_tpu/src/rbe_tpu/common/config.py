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

"""Shared entry-point plumbing for the agent and the worker."""

from __future__ import annotations

import os
import socket
import sys

from rbe_tpu.common import gcs

BUCKET_ENV = "TORCH_TPU_RELAY_BUCKET"
TPU_ENV = "TORCH_TPU_RELAY_TPU"
WORKER_ID_ENV = "TORCH_TPU_RELAY_WORKER_ID"
ROOT_ENV = "TORCH_TPU_RELAY_ROOT"
LAYER_DIRS_ENV = "TORCH_TPU_RELAY_LAYER_DIRS"
TIMEOUT_ENV = "TORCH_TPU_RELAY_TIMEOUT"


class ConfigError(Exception):
  """A required setting is missing or unusable."""


def require(env, name: str) -> str:
  """Reads a setting that has no sensible default.

  Args:
    env: Environment mapping.
    name: Variable to read.

  Returns:
    The value.

  Raises:
    ConfigError: when it is missing or empty.
  """
  value = env.get(name, "").strip()
  if not value:
    raise ConfigError(f"{name} must be set")
  return value


def bucket_client(env) -> gcs.HttpClient:
  """Builds a storage client for the configured relay bucket."""
  return gcs.HttpClient(require(env, BUCKET_ENV))


def worker_id(env) -> str:
  """Returns a stable-enough identity for one RBE worker.

  Only used for logging and for spotting a worker talking to itself in
  a trace. Correctness rests on binding generations, not on this.
  """
  configured = env.get(WORKER_ID_ENV, "").strip()
  if configured:
    return configured
  return f"{socket.gethostname()}-{os.getpid()}"


def log(message: str) -> None:
  """Writes a line to stderr, where journald and RBE both pick it up."""
  print(f"relay: {message}", file=sys.stderr, flush=True)
