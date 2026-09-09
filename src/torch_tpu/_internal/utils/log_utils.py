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

"""Helper for setting up python logging with absl."""

import logging  # PYTHON_LOGGING_OK=Used to configure absl logging.
import sys

from absl import logging as absl_logging


# TODO(b/498756865): apply this to all torch_tpu python files.
def log_to_stderr() -> None:
  """Sets up a logger that directs all logs to stderr."""
  # Route absl logs to the standard python logging module
  absl_logging.use_python_logging()

  # Clear any existing handlers (like the default absl ones)
  for handler in list(logging.root.handlers):
    logging.root.removeHandler(handler)

  # Create a clean stderr stream handler
  formatter = logging.Formatter(
      "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  )
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setFormatter(formatter)

  # Attach to the root logger and set level
  logging.root.addHandler(stderr_handler)
  logging.root.setLevel(logging.INFO)


__all__ = [
    "log_to_stderr",
]
