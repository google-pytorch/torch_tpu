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

"""Runs the relay agent. One of these per TPU VM, under systemd.

Settings come from the environment:

  TORCH_TPU_RELAY_BUCKET   relay bucket name (required)
  TORCH_TPU_RELAY_TPU      this VM's TPU name (required)
  TORCH_TPU_RELAY_ROOT     scratch directory, default /tmp/torch_tpu_relay
  TORCH_TPU_RELAY_CHIP_PROBE  probe command, default is a device node check

The TPU name is required rather than guessed. A TPU VM's hostname is not
its TPU name, and getting it wrong would have an agent serve somebody
else's mailbox.
"""

from __future__ import annotations

import os
import sys

from ci.tools.rbe_tpu import agent as agent_lib
from ci.tools.rbe_tpu import config
from ci.tools.rbe_tpu import runner as runner_lib


def build(env, client=None) -> agent_lib.Agent:
  """Assembles an agent from the environment.

  Args:
    env: Environment mapping.
    client: Storage client, built from the environment when omitted.

  Returns:
    An agent ready to serve.
  """
  client = client or config.bucket_client(env)
  tpu = config.require(env, config.TPU_ENV)
  root = env.get(config.ROOT_ENV, "").strip() or runner_lib.DEFAULT_ROOT
  runner = runner_lib.LocalRunner(client, root)
  return agent_lib.Agent(client, tpu, runner)


def main(argv=None, env=None) -> int:
  """Serves this VM's mailbox until the process is stopped."""
  del argv
  env = os.environ if env is None else env
  try:
    agent = build(env)
  except config.ConfigError as error:
    config.log(str(error))
    return 2
  config.log("agent starting")
  if not agent.start():
    config.log("chip failed its first probe; staying out of the fleet")
  agent.run_forever()
  return 0


if __name__ == "__main__":
  sys.exit(main())
