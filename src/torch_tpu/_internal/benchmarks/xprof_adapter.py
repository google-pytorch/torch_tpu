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

"""OSS implementation of TorchTPU's XProf adapter."""

import contextlib


class TraceMe(contextlib.AbstractContextManager):
  """A no-op context manager for TraceMe for OSS."""

  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


class XprofSession(contextlib.AbstractContextManager):
  """A no-op context manager for XprofSession for OSS."""

  def __init__(self, *args, **kwargs):
    pass

  def start_session(self, *args, **kwargs):
    pass

  def end_session_and_get_session_id(self):
    pass

  def end_session_and_get_url(self):
    # MUST return a string to satisfy the f-string formatter
    return "http://dummy-url-for-oss"

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


class XprofAnalysisClient(contextlib.AbstractContextManager):
  """A no-op context manager for XprofAnalysisClient for OSS."""

  def __init__(self, *args, **kwargs):
    pass

  def get_profile_data(self, *args, **kwargs):
    return None, None

  def upload(self, *args, **kwargs):
    pass

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass
