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

"""OSS implementation of TorchTPU's multiprocessing wrapper."""

import multiprocessing
import sys
from typing import Any

from absl import app
from absl import flags


def handle_main(main, *args, **kwargs):
  """Handles main entrypoint for multiprocessing applications in OSS."""
  return app.run(main, *args, **kwargs)


def handle_test_main(main, *args, **kwargs):
  """Handles test main entrypoint for multiprocessing tests in OSS."""
  return main(*args, **kwargs)


def get_context(method=None) -> Any:
  """Returns a multiprocessing context."""
  return multiprocessing.get_context(method)


def parse_absl_flags() -> None:
  """Parses absl flags in spawned worker subprocesses if not already parsed.

  Required for OSS: Subprocesses spawned via torch.multiprocessing.spawn bypass
  the absl main entry point and start with unparsed flags. This prevents
  subprocesses from raising UnparsedFlagAccessError when downstream utilities
  read flag values (e.g., --test_mode in et.assert_raises_message).
  """
  if not flags.FLAGS.is_parsed():
    # Calling flags.FLAGS as a callable evaluates and marks the registry parsed.
    # We pass known_only=True so absl consumes flags declared in the binary
    # without raising UnrecognizedFlagError on runner-injected arguments (e.g.,
    # torchrun/pytest options present in sys.argv).
    flags.FLAGS(sys.argv, known_only=True)
