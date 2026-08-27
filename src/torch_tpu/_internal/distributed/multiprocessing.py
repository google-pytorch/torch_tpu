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
from typing import Any
from absl import app


def handle_main(main, *args, **kwargs):
  """Handles main entrypoint for multiprocessing applications in OSS."""
  return app.run(main, *args, **kwargs)


def handle_test_main(main, *args, **kwargs):
  """Handles test main entrypoint for multiprocessing tests in OSS."""
  return main(*args, **kwargs)


def get_context(method=None) -> Any:
  """Returns a multiprocessing context."""
  return multiprocessing.get_context(method)
