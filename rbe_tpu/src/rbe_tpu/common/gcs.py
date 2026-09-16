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

"""A small Cloud Storage client built on the JSON API and urllib.

The relay agent runs on TPU VMs and the relay worker runs inside an RBE
action. Neither is guaranteed to have gcloud, gsutil, or the Cloud Storage
Python library on it, so this talks to the REST API directly and gets its
credentials from the GCE metadata server.

The part that matters is generation preconditions. `put` with
`if_generation_match=0` creates an object only when it does not already
exist, which is the compare-and-swap the binding protocol is built on.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

_STORAGE_HOST = "https://storage.googleapis.com"
_METADATA_HOST = "http://metadata.google.internal"
_TOKEN_PATH = "/computeMetadata/v1/instance/service-accounts/default/token"

# Refresh the access token this many seconds before it actually expires.
_TOKEN_SKEW_SECONDS = 120

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0


class Error(Exception):
  """Base class for storage errors."""


class PreconditionFailed(Error):
  """A generation precondition did not hold.

  This is never retried. It is the expected outcome of losing a race, and
  callers are supposed to handle it.
  """


class TransientError(Error):
  """The request failed after exhausting retries."""


@dataclasses.dataclass(frozen=True)
class Blob:
  """An object's bytes together with the generation they came from."""

  name: str
  data: bytes
  generation: int

  def json(self) -> dict:
    """Parses the payload as a JSON object.

    Returns:
      The decoded mapping.

    Raises:
      ValueError: if the payload is not a JSON object.
    """
    value = json.loads(self.data.decode("utf-8"))
    if not isinstance(value, dict):
      raise ValueError(f"{self.name} is not a JSON object")
    return value


class Client:
  """Reads and writes objects in one bucket.

  Implementations must agree on generation semantics, because the binding
  protocol relies on them. See `fake_gcs.FakeClient` for the reference
  behaviour that tests assert against.
  """

  def get(self, name: str) -> Blob | None:
    raise NotImplementedError

  def put(
      self, name: str, data: bytes, *, if_generation_match: int | None = None
  ) -> Blob:
    raise NotImplementedError

  def delete(
      self, name: str, *, if_generation_match: int | None = None
  ) -> bool:
    raise NotImplementedError

  def list(self, prefix: str) -> list[str]:
    raise NotImplementedError


class _MetadataTokenSource:
  """Fetches and caches an access token from user ADC or GCE metadata."""

  def __init__(
      self,
      opener,
      time_fn=time.time,
      run_fn=subprocess.run,
      adc_path: str | None = None,
  ):
    self._opener = opener
    self._time_fn = time_fn
    self._run_fn = run_fn
    self._adc_path = adc_path
    self._token = ""
    self._expires_at = 0.0

  def _user_adc_exists(self) -> bool:
    path = (
        self._adc_path
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.path.expanduser(
            "~/.config/gcloud/application_default_credentials.json"
        )
    )
    return os.path.exists(path)

  def token(self) -> str:
    """Returns a valid access token, fetching a new one when needed."""
    if self._token and self._time_fn() < self._expires_at:
      return self._token

    if self._user_adc_exists():
      try:
        cmd = ["gcloud", "auth", "application-default", "print-access-token"]
        proc = self._run_fn(cmd, capture_output=True, text=True, check=True)
        self._token = proc.stdout.strip()
        self._expires_at = self._time_fn() + 1800.0
        return self._token
      except Exception:
        pass

    try:
      request = urllib.request.Request(
          _METADATA_HOST + _TOKEN_PATH,
          headers={"Metadata-Flavor": "Google"},
      )
      with self._opener.open(request, timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
      self._token = payload["access_token"]
      lifetime = float(payload.get("expires_in", 3600))
      self._expires_at = self._time_fn() + max(
          lifetime - _TOKEN_SKEW_SECONDS, 60.0
      )
      return self._token
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        KeyError,
        OSError,
    ):
      try:
        cmd = ["gcloud", "auth", "application-default", "print-access-token"]
        proc = self._run_fn(cmd, capture_output=True, text=True, check=True)
        self._token = proc.stdout.strip()
      except Exception:
        cmd = ["gcloud", "auth", "print-access-token"]
        proc = self._run_fn(cmd, capture_output=True, text=True, check=True)
        self._token = proc.stdout.strip()
      self._expires_at = self._time_fn() + 1800.0
      return self._token


class HttpClient(Client):
  """Talks to the real Cloud Storage JSON API."""

  def __init__(self, bucket: str, *, opener=None, sleep_fn=time.sleep):
    self._bucket = bucket
    self._opener = opener or urllib.request.build_opener()
    self._sleep_fn = sleep_fn
    self._tokens = _MetadataTokenSource(self._opener)

  def get(self, name: str) -> Blob | None:
    """Downloads an object, or returns None when it does not exist."""
    url = (
        f"{_STORAGE_HOST}/storage/v1/b/{self._bucket}"
        f"/o/{urllib.parse.quote(name, safe='')}?alt=media"
    )
    response = self._request("GET", url, allow_missing=True)
    if response is None:
      return None
    body, headers = response
    # A media download carries the generation in a header rather than the
    # body, which keeps this to a single round trip.
    generation = int(headers.get("x-goog-generation", "0"))
    return Blob(name=name, data=body, generation=generation)

  def put(
      self, name: str, data: bytes, *, if_generation_match: int | None = None
  ) -> Blob:
    """Uploads an object, honouring a generation precondition."""
    query = {"uploadType": "media", "name": name}
    if if_generation_match is not None:
      query["ifGenerationMatch"] = str(if_generation_match)
    url = (
        f"{_STORAGE_HOST}/upload/storage/v1/b/{self._bucket}/o"
        f"?{urllib.parse.urlencode(query)}"
    )
    body, _ = self._request(
        "POST", url, data=data, content_type="application/octet-stream"
    )
    metadata = json.loads(body.decode("utf-8"))
    return Blob(name=name, data=data, generation=int(metadata["generation"]))

  def delete(
      self, name: str, *, if_generation_match: int | None = None
  ) -> bool:
    """Removes an object. Returns False when it was already gone."""
    url = (
        f"{_STORAGE_HOST}/storage/v1/b/{self._bucket}"
        f"/o/{urllib.parse.quote(name, safe='')}"
    )
    if if_generation_match is not None:
      url += f"?ifGenerationMatch={if_generation_match}"
    return self._request("DELETE", url, allow_missing=True) is not None

  def list(self, prefix: str) -> list[str]:
    """Returns every object name under a prefix, in sorted order."""
    names: list[str] = []
    page_token = ""
    while True:
      query = {"prefix": prefix, "maxResults": "1000"}
      if page_token:
        query["pageToken"] = page_token
      url = (
          f"{_STORAGE_HOST}/storage/v1/b/{self._bucket}/o"
          f"?{urllib.parse.urlencode(query)}"
      )
      body, _ = self._request("GET", url)
      payload = json.loads(body.decode("utf-8"))
      names.extend(item["name"] for item in payload.get("items", []))
      page_token = payload.get("nextPageToken", "")
      if not page_token:
        return sorted(names)

  def _request(
      self, method, url, *, data=None, content_type=None, allow_missing=False
  ):
    """Issues one request, retrying transient failures.

    Args:
      method: HTTP verb.
      url: Fully built request URL.
      data: Optional request body.
      content_type: Optional Content-Type for the body.
      allow_missing: When true, a 404 returns None instead of raising.

    Returns:
      A (body, headers) pair, or None for a tolerated 404.

    Raises:
      PreconditionFailed: on 412, immediately and without retrying.
      TransientError: when retries are exhausted or the error is fatal.
    """
    last_error = None
    for attempt in range(_MAX_ATTEMPTS):
      try:
        return self._attempt(method, url, data, content_type)
      except urllib.error.HTTPError as error:
        if error.code == 412:
          raise PreconditionFailed(f"{method} {url} failed precondition")
        if error.code == 404 and allow_missing:
          return None
        if error.code not in _RETRYABLE_STATUS:
          raise TransientError(f"{method} failed: {error.code}") from error
        last_error = error
      except urllib.error.URLError as error:
        last_error = error
      self._sleep_fn(_backoff_seconds(attempt))
    raise TransientError(f"{method} {url} failed: {last_error}")

  def _attempt(self, method, url, data, content_type):
    headers = {"Authorization": f"Bearer {self._tokens.token()}"}
    if content_type:
      headers["Content-Type"] = content_type
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    with self._opener.open(request, timeout=120) as response:
      return response.read(), {
          k.lower(): v for k, v in response.headers.items()
      }


def _backoff_seconds(attempt: int) -> float:
  """Returns a jittered exponential backoff delay.

  Jitter matters here because up to 28 workers race for the same binding
  objects at boot and would otherwise retry in lockstep.

  Args:
    attempt: Zero-based attempt number.

  Returns:
    Seconds to wait before the next attempt.
  """
  ceiling = min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_CAP_SECONDS)
  return random.uniform(0.0, ceiling)
