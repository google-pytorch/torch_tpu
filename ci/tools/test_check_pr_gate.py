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

"""Unit tests for check_pr_gate.py."""

import http.server
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from typing import Final
import unittest
from unittest import mock
import urllib.error

_CI_TOOLS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent
_GATE_SCRIPT: Final[pathlib.Path] = _CI_TOOLS_DIR / "check_pr_gate.py"

# Import check_pr_gate directly for function-level testing.
sys.path.insert(0, str(_CI_TOOLS_DIR))
import check_pr_gate  # pylint: disable=g-import-not-at-top


class CheckPrGateTest(
    unittest.TestCase  # UNITTEST_OK=avoid circular dependency on bazel-diff
):
  """Validates PR pre-flight gate decision logic and fallbacks."""

  def test_is_pr_closed_or_merged(self):
    """Verifies state and merged boolean classification."""
    self.assertTrue(check_pr_gate.is_pr_closed_or_merged({"state": "closed"}))
    self.assertTrue(check_pr_gate.is_pr_closed_or_merged({"state": "CLOSED"}))
    self.assertTrue(
        check_pr_gate.is_pr_closed_or_merged({"state": "open", "merged": True})
    )
    self.assertTrue(
        check_pr_gate.is_pr_closed_or_merged(
            {"state": "open", "merged_at": "2026-09-04T12:00:00Z"}
        )
    )
    self.assertFalse(
        check_pr_gate.is_pr_closed_or_merged(
            {"state": "open", "merged": False, "merged_at": None}
        )
    )
    self.assertFalse(check_pr_gate.is_pr_closed_or_merged({}))

  @mock.patch("urllib.request.urlopen")
  def test_non_pull_request_events_pass_gate(self, mock_urlopen):
    """Allows push, schedule, and workflow_dispatch events without API calls."""
    for event in ("push", "workflow_dispatch", "schedule"):
      result = check_pr_gate.should_run_presubmit(
          event_name=event,
          repo="google-pytorch/torch_tpu",
          pr_number="3550",
      )
      self.assertTrue(result)
    mock_urlopen.assert_not_called()

  @mock.patch("urllib.request.urlopen")
  def test_missing_pr_number_or_repo_fails_open(self, mock_urlopen):
    """Fails open when required PR metadata is absent."""
    self.assertTrue(
        check_pr_gate.should_run_presubmit(
            event_name="pull_request",
            repo="",
            pr_number="3550",
        )
    )
    self.assertTrue(
        check_pr_gate.should_run_presubmit(
            event_name="pull_request",
            repo="google-pytorch/torch_tpu",
            pr_number="",
        )
    )
    mock_urlopen.assert_not_called()

  @mock.patch("time.sleep")
  @mock.patch("urllib.request.urlopen")
  def test_pr_already_closed_skips_run(self, mock_urlopen, mock_sleep):
    """Skips presubmit immediately when PR is closed on initial lookup."""
    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = json.dumps({
        "state": "closed",
        "merged": False,
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    result = check_pr_gate.should_run_presubmit(
        event_name="pull_request",
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
        settle_seconds=6.0,
    )

    self.assertFalse(result)
    mock_sleep.assert_not_called()
    self.assertEqual(mock_urlopen.call_count, 1)

  @mock.patch("time.sleep")
  @mock.patch("urllib.request.urlopen")
  def test_pr_already_merged_skips_run(self, mock_urlopen, mock_sleep):
    """Skips presubmit immediately when PR is merged on initial lookup."""
    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = json.dumps({
        "state": "closed",
        "merged": True,
        "merged_at": "2026-09-04T12:00:00Z",
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    result = check_pr_gate.should_run_presubmit(
        event_name="pull_request",
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
        settle_seconds=6.0,
    )

    self.assertFalse(result)
    mock_sleep.assert_not_called()

  @mock.patch("time.sleep")
  @mock.patch("urllib.request.urlopen")
  def test_pr_closes_during_settling_window(self, mock_urlopen, mock_sleep):
    """Skips presubmit when an open PR merges during the settling window."""
    open_resp = mock.MagicMock()
    open_resp.read.return_value = json.dumps({
        "state": "open",
        "merged": False,
    }).encode("utf-8")
    open_resp.__enter__.return_value = open_resp

    closed_resp = mock.MagicMock()
    closed_resp.read.return_value = json.dumps({
        "state": "closed",
        "merged": True,
    }).encode("utf-8")
    closed_resp.__enter__.return_value = closed_resp

    mock_urlopen.side_effect = [open_resp, closed_resp]

    result = check_pr_gate.should_run_presubmit(
        event_name="pull_request",
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
        settle_seconds=5.0,
    )

    self.assertFalse(result)
    mock_sleep.assert_called_once_with(5.0)
    self.assertEqual(mock_urlopen.call_count, 2)

  @mock.patch("time.sleep")
  @mock.patch("urllib.request.urlopen")
  def test_pr_remains_open_after_settling_window(
      self, mock_urlopen, mock_sleep
  ):
    """Allows presubmit when an active review PR remains open after delay."""
    open_resp1 = mock.MagicMock()
    open_resp1.read.return_value = json.dumps({
        "state": "open",
        "merged": False,
    }).encode("utf-8")
    open_resp1.__enter__.return_value = open_resp1

    open_resp2 = mock.MagicMock()
    open_resp2.read.return_value = json.dumps({
        "state": "open",
        "merged": False,
    }).encode("utf-8")
    open_resp2.__enter__.return_value = open_resp2

    mock_urlopen.side_effect = [open_resp1, open_resp2]

    result = check_pr_gate.should_run_presubmit(
        event_name="pull_request",
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
        settle_seconds=3.0,
    )

    self.assertTrue(result)
    mock_sleep.assert_called_once_with(3.0)
    self.assertEqual(mock_urlopen.call_count, 2)

  @mock.patch("urllib.request.urlopen")
  def test_api_http_error_fails_open(self, mock_urlopen):
    """Fails open to allow presubmits when GitHub API returns HTTP error."""
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://api.github.com",
        code=500,
        msg="Internal Server Error",
        hdrs={},
        fp=io.BytesIO(b"error"),
    )

    result = check_pr_gate.should_run_presubmit(
        event_name="pull_request",
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
    )

    self.assertTrue(result)

  @mock.patch("urllib.request.urlopen")
  def test_api_network_error_fails_open(self, mock_urlopen):
    """Fails open when GitHub API request encounters a network error."""
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    result = check_pr_gate.should_run_presubmit(
        event_name="pull_request",
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
    )

    self.assertTrue(result)

  @mock.patch("urllib.request.urlopen")
  def test_malformed_json_fails_open(self, mock_urlopen):
    """Fails open when GitHub API response is not valid JSON."""
    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = b"not-json"
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    result = check_pr_gate.should_run_presubmit(
        event_name="pull_request",
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
    )

    self.assertTrue(result)

  @mock.patch("urllib.request.urlopen")
  def test_authorization_header_passed_when_token_provided(self, mock_urlopen):
    """Supplies bearer token header when authentication token is given."""
    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = json.dumps({"state": "closed"}).encode(
        "utf-8"
    )
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    check_pr_gate.fetch_pr_data(
        repo="google-pytorch/torch_tpu",
        pr_number="3550",
        token="ghp_test_token_123",
    )

    req = mock_urlopen.call_args[0][0]
    self.assertEqual(
        req.get_header("Authorization"), "Bearer ghp_test_token_123"
    )
    self.assertEqual(req.get_header("Accept"), "application/vnd.github+json")

  def test_record_output_appends_to_file(self):
    """Writes gate result to target GitHub Actions step output file."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
      temp_path = f.name
    try:
      check_pr_gate.record_output(temp_path, should_run=True)
      check_pr_gate.record_output(temp_path, should_run=False)
      with open(temp_path, "r", encoding="utf-8") as f:
        content = f.read()
      self.assertEqual(content, "should_run=true\nshould_run=false\n")
    finally:
      os.unlink(temp_path)

  def test_cli_integration_push_event(self):
    """Executes CLI directly for push events and verifies output."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
      output_file = f.name
    try:
      env = os.environ.copy()
      env["EVENT_NAME"] = "push"
      env["GITHUB_OUTPUT"] = output_file
      res = subprocess.run(
          [sys.executable, str(_GATE_SCRIPT)],
          env=env,
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertIn("Gate evaluation: should_run=true", res.stdout)
      with open(output_file, "r", encoding="utf-8") as f:
        self.assertEqual(f.read().strip(), "should_run=true")
    finally:
      os.unlink(output_file)

  def test_cli_integration_closed_pr(self):
    """Executes CLI against a mock server returning closed PR state."""

    class MockHandler(http.server.BaseHTTPRequestHandler):

      def do_GET(self):  # pylint: disable=invalid-name
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"state": "closed", "merged": True}).encode("utf-8")
        )

      def log_message(self, *args):
        pass

    server = http.server.HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
      output_file = f.name
    try:
      res = subprocess.run(
          [
              sys.executable,
              str(_GATE_SCRIPT),
              "--event-name=pull_request",
              "--repo=google-pytorch/torch_tpu",
              "--pr-number=3550",
              "--settle-seconds=0",
              f"--api-url=http://127.0.0.1:{port}",
              f"--output-file={output_file}",
          ],
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertIn("Gate evaluation: should_run=false", res.stdout)
      with open(output_file, "r", encoding="utf-8") as f:
        self.assertEqual(f.read().strip(), "should_run=false")
    finally:
      server.server_close()
      os.unlink(output_file)


if __name__ == "__main__":
  unittest.main()
