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

"""Unit tests for the GCS HTTP client and token source."""

from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from ci.tools.relay_mailbox import gcs


class _FakeOpener:

  def __init__(self, payload: dict | None = None, error: Exception | None = None):
    self.payload = payload
    self.error = error
    self.calls = 0

  def open(self, _request, timeout=2):
    self.calls += 1
    if self.error:
      raise self.error
    data = json.dumps(self.payload or {}).encode("utf-8")
    return io.BytesIO(data)


class MetadataTokenSourceTest(
    unittest.TestCase  # UNITTEST_OK=CI utility test with no torch_tpu deps
):

  def setUp(self):
    super().setUp()
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.now = 1000.0

  def test_prefers_user_adc_when_present(self):
    adc_file = os.path.join(self.temp.name, "adc.json")
    with open(adc_file, "w", encoding="utf-8") as handle:
      handle.write("{}")

    opener = _FakeOpener(payload={"access_token": "metadata-token", "expires_in": 3600})
    run_fn = mock.Mock(
        return_value=subprocess.CompletedProcess([], 0, "user-adc-token\n", "")
    )
    source = gcs._MetadataTokenSource(
        opener,
        time_fn=lambda: self.now,
        run_fn=run_fn,
        adc_path=adc_file,
    )

    self.assertEqual(source.token(), "user-adc-token")
    self.assertEqual(opener.calls, 0)
    run_fn.assert_called_once()

  def test_falls_back_to_metadata_server_when_no_user_adc(self):
    missing_adc = os.path.join(self.temp.name, "missing_adc.json")
    opener = _FakeOpener(payload={"access_token": "metadata-token", "expires_in": 3600})
    run_fn = mock.Mock()
    source = gcs._MetadataTokenSource(
        opener,
        time_fn=lambda: self.now,
        run_fn=run_fn,
        adc_path=missing_adc,
    )

    self.assertEqual(source.token(), "metadata-token")
    self.assertEqual(opener.calls, 1)
    run_fn.assert_not_called()

  def test_falls_back_to_metadata_server_if_adc_command_fails(self):
    adc_file = os.path.join(self.temp.name, "adc.json")
    with open(adc_file, "w", encoding="utf-8") as handle:
      handle.write("{}")

    opener = _FakeOpener(payload={"access_token": "metadata-token", "expires_in": 3600})
    run_fn = mock.Mock(side_effect=subprocess.CalledProcessError(1, "gcloud"))
    source = gcs._MetadataTokenSource(
        opener,
        time_fn=lambda: self.now,
        run_fn=run_fn,
        adc_path=adc_file,
    )

    self.assertEqual(source.token(), "metadata-token")
    self.assertEqual(opener.calls, 1)

  def test_caches_token_until_expiry(self):
    missing_adc = os.path.join(self.temp.name, "missing_adc.json")
    opener = _FakeOpener(payload={"access_token": "cached-token", "expires_in": 3600})
    source = gcs._MetadataTokenSource(
        opener,
        time_fn=lambda: self.now,
        adc_path=missing_adc,
    )

    self.assertEqual(source.token(), "cached-token")
    self.now += 600.0
    self.assertEqual(source.token(), "cached-token")
    self.assertEqual(opener.calls, 1)

    self.now += 3600.0
    self.assertEqual(source.token(), "cached-token")
    self.assertEqual(opener.calls, 2)


if __name__ == "__main__":
  unittest.main()
