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

"""Stateful fakes for the external boundaries of the TorchTPU tooling.

These replace call-recording mocks of `subprocess.run`, Google Cloud Storage and
GitHub in tool tests, so that tests can assert on the resulting state (files
written, objects uploaded, team membership) rather than on the exact calls that
were made. See http://go/tott/289.

* `FakeSubprocessRunner` is a drop-in replacement for `subprocess.run`. Commands
  are served by handlers registered for an argv prefix; handlers may have side
  effects, such as writing the files the real command would produce.
* `FakeGcs` is a bucket store backed by a local directory. It serves the
  `gcloud storage` commands through a `FakeSubprocessRunner`.
* `FakeGithub` is an in-memory stand-in for the PyGithub client, covering users,
  organizations and teams.

This module depends only on the Python standard library.
"""

# TODO(wan): revisit DECISION (b/568592948): the fakes live under
# oss/ci/tools/testing because OSS code cannot depend on google/ code, while
# google/ tests can import from oss/. Move them if a better shared test-only
# location emerges.

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import os
import pathlib
import subprocess
from typing import Any


@dataclasses.dataclass(frozen=True)
class FakeCall:
  """A command received by `FakeSubprocessRunner`.

  Attributes:
    args: The argv, with every element converted to `str`. A shell command
      string is stored as a one-element tuple.
    env: The `env` mapping passed to `subprocess.run`, if any.
    cwd: The `cwd` passed to `subprocess.run`, if any.
  """

  args: tuple[str, ...]
  env: Mapping[str, str] | None = None
  cwd: str | None = None


@dataclasses.dataclass(frozen=True)
class FakeResult:
  """What a fake command writes and returns.

  `stdout` and `stderr` are text; `FakeSubprocessRunner` encodes them when the
  caller did not ask for text mode.
  """

  returncode: int = 0
  stdout: str = ""
  stderr: str = ""


# A handler receives the call and returns its result. Returning `None` means
# success with no output.
Handler = Callable[[FakeCall], FakeResult | None]


class UnexpectedCommandError(AssertionError):
  """Raised when a command has no registered handler."""


class FakeSubprocessRunner:
  """A stateful stand-in for `subprocess.run`.

  Register a handler per command prefix with `on()`, then pass the runner
  wherever `subprocess.run` is expected (an injected `runner=` parameter, or a
  patch of the module's `subprocess.run`). The longest matching prefix wins,
  and a command without a handler raises `UnexpectedCommandError`, so tests
  fail loudly when the code under test runs something unexpected.

  The runner follows the `subprocess.run` contract for the arguments that
  affect the result: output is only returned when captured (`capture_output`
  or `stdout`/`stderr=subprocess.PIPE`), it is text only in text mode, and
  `check=True` raises `subprocess.CalledProcessError` on a non-zero exit code.

  Example:

    runner = fakes.FakeSubprocessRunner()
    runner.on(["git", "rev-parse"], stdout="abc123\\n")
    runner.on(["blaze", "test"], write_test_logs)  # A side-effecting handler.
  """

  def __init__(self) -> None:
    self._handlers: list[tuple[tuple[str, ...], Handler]] = []
    self.calls: list[FakeCall] = []

  def on(
      self,
      prefix: Sequence[str],
      handler: Handler | None = None,
      *,
      returncode: int = 0,
      stdout: str = "",
      stderr: str = "",
  ) -> None:
    """Serves commands starting with `prefix`.

    Args:
      prefix: The leading argv elements to match.
      handler: Called with the `FakeCall`; may have side effects. If omitted,
        every matching command returns the given canned result.
      returncode: Exit code of the canned result.
      stdout: Standard output of the canned result.
      stderr: Standard error of the canned result.
    """
    if handler is None:
      canned = FakeResult(returncode, stdout, stderr)
      handler = lambda call: canned
    self._handlers.append((tuple(str(arg) for arg in prefix), handler))

  def commands(self, *prefix: str) -> list[tuple[str, ...]]:
    """Returns the argv of every received command starting with `prefix`."""
    return [
        call.args
        for call in self.calls
        if call.args[: len(prefix)] == tuple(prefix)
    ]

  def _find_handler(self, args: tuple[str, ...]) -> Handler:
    best: tuple[tuple[str, ...], Handler] | None = None
    for prefix, handler in self._handlers:
      if args[: len(prefix)] == prefix and (
          best is None or len(prefix) > len(best[0])
      ):
        best = (prefix, handler)
    if best is None:
      raise UnexpectedCommandError(f"no fake handler for command: {args}")
    return best[1]

  def __call__(
      self,
      args: str | Sequence[Any],
      *,
      check: bool = False,
      capture_output: bool = False,
      stdout: Any = None,
      stderr: Any = None,
      text: bool | None = None,
      universal_newlines: bool | None = None,
      encoding: str | None = None,
      errors: str | None = None,
      env: Mapping[str, str] | None = None,
      cwd: str | os.PathLike[str] | None = None,
      **kwargs: Any,
  ) -> subprocess.CompletedProcess[Any]:
    del kwargs  # E.g. `timeout` or `input`; they do not affect the result.
    argv = (args,) if isinstance(args, str) else tuple(str(a) for a in args)
    call = FakeCall(
        args=argv,
        env=dict(env) if env is not None else None,
        cwd=os.fspath(cwd) if cwd is not None else None,
    )
    self.calls.append(call)
    result = self._find_handler(argv)(call) or FakeResult()

    text_mode = bool(text or universal_newlines or encoding or errors)

    def output(value: str, captured: bool) -> str | bytes | None:
      if not captured:
        return None
      return value if text_mode else value.encode(encoding or "utf-8")

    stdout_text = result.stdout
    if stderr == subprocess.STDOUT:
      stdout_text += result.stderr
    out = output(stdout_text, capture_output or stdout == subprocess.PIPE)
    err = output(result.stderr, capture_output or stderr == subprocess.PIPE)
    if check and result.returncode != 0:
      raise subprocess.CalledProcessError(
          result.returncode, list(argv), output=out, stderr=err
      )
    return subprocess.CompletedProcess(list(argv), result.returncode, out, err)


class FakeGcs:
  """A Google Cloud Storage stand-in backed by a local directory.

  Objects are stored as `<root>/<bucket>/<object name>`. Call `install()` to
  serve uploads of local files with `gcloud storage cp` from a
  `FakeSubprocessRunner`. Uploads into a bucket listed in
  `read_only_buckets` fail the way `gcloud` does on a permission error.
  """

  def __init__(
      self,
      root: str | os.PathLike[str],
      *,
      read_only_buckets: Iterable[str] = (),
  ) -> None:
    self._root = pathlib.Path(root)
    self._read_only_buckets = frozenset(read_only_buckets)

  def install(self, runner: FakeSubprocessRunner) -> None:
    """Makes `runner` serve `gcloud storage cp` uploads into this store."""
    runner.on(["gcloud", "storage", "cp"], self._cp)

  def _path(self, uri: str) -> pathlib.Path:
    if not uri.startswith("gs://"):
      raise ValueError(f"not a gs:// URI: {uri}")
    return self._root / uri.removeprefix("gs://")

  def write(self, uri: str, data: bytes) -> None:
    """Creates or replaces the object at `uri`."""
    path = self._path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

  def read(self, uri: str) -> bytes:
    """Returns the content of the object at `uri`."""
    return self._path(uri).read_bytes()

  def exists(self, uri: str) -> bool:
    return self._path(uri).is_file()

  def list_objects(self, prefix: str) -> list[str]:
    """Returns the sorted URIs of all objects under the `prefix` URI."""
    base = self._path(prefix)
    if base.is_file():
      return [prefix]
    if not base.is_dir():
      return []
    return sorted(
        "gs://" + path.relative_to(self._root).as_posix()
        for path in base.rglob("*")
        if path.is_file()
    )

  def _cp(self, call: FakeCall) -> FakeResult:
    *sources, dest = [arg for arg in call.args[3:] if not arg.startswith("-")]
    if not sources or not dest.startswith("gs://"):
      raise UnexpectedCommandError(f"only uploads are supported: {call.args}")
    bucket = dest.removeprefix("gs://").split("/", 1)[0]
    if bucket in self._read_only_buckets:
      return FakeResult(
          1,
          stderr=(
              "AccessDeniedException: 403 caller does not have"
              f" storage.objects.create access to bucket {bucket}\n"
          ),
      )
    for source in sources:
      source_path = pathlib.Path(source)
      target = dest + source_path.name if dest.endswith("/") else dest
      self.write(target, source_path.read_bytes())
    return FakeResult()


class FakeGithubError(LookupError):
  """Raised for an unknown user, organization or team (an HTTP 404)."""


@dataclasses.dataclass
class FakeGithubUser:
  """A `github.NamedUser.NamedUser` stand-in."""

  login: str
  name: str | None = None


class FakeGithubTeam:
  """A `github.Team.Team` stand-in that tracks membership state.

  `add_membership()` adds members of the organization to the team directly and
  invites everyone else, like GitHub does.
  """

  def __init__(self, organization: "FakeGithubOrganization", slug: str):
    self._organization = organization
    self.slug = slug
    self.member_logins: set[str] = set()
    self.invited_logins: set[str] = set()

  def get_members(self) -> list[FakeGithubUser]:
    github = self._organization.github
    return [github.get_user(login) for login in sorted(self.member_logins)]

  def invitations(self) -> list[FakeGithubUser]:
    github = self._organization.github
    return [github.get_user(login) for login in sorted(self.invited_logins)]

  def add_membership(
      self, member: FakeGithubUser, role: str = "member"
  ) -> None:
    if role not in ("member", "maintainer"):
      raise ValueError(f"invalid team role: {role}")
    if member.login in self._organization.member_logins:
      self.member_logins.add(member.login)
    else:
      self.invited_logins.add(member.login)


class FakeGithubOrganization:
  """A `github.Organization.Organization` stand-in."""

  def __init__(self, github: "FakeGithub", login: str):
    self.github = github
    self.login = login
    self.member_logins: set[str] = set()
    self._teams: dict[str, FakeGithubTeam] = {}

  def get_team_by_slug(self, slug: str) -> FakeGithubTeam:
    if slug not in self._teams:
      raise FakeGithubError(f"team not found: {self.login}/{slug}")
    return self._teams[slug]

  def add_team(self, slug: str) -> FakeGithubTeam:
    return self._teams.setdefault(slug, FakeGithubTeam(self, slug))


class FakeGithub:
  """An in-memory stand-in for the PyGithub client.

  Seed state with `add_user()`, `add_organization()` and
  `FakeGithubOrganization.add_team()`, then read the state back after the code
  under test ran, e.g. `team.member_logins`.
  """

  def __init__(self) -> None:
    self._users: dict[str, FakeGithubUser] = {}
    self._organizations: dict[str, FakeGithubOrganization] = {}

  def add_user(self, login: str, name: str | None = None) -> FakeGithubUser:
    user = FakeGithubUser(login, name)
    self._users[login] = user
    return user

  def add_organization(
      self, login: str, members: Iterable[str] = ()
  ) -> FakeGithubOrganization:
    """Creates an organization whose `members` are existing users."""
    organization = self._organizations.setdefault(
        login, FakeGithubOrganization(self, login)
    )
    for member in members:
      self.get_user(member)  # Members must exist.
      organization.member_logins.add(member)
    return organization

  def get_user(self, login: str) -> FakeGithubUser:
    if login not in self._users:
      raise FakeGithubError(f"user not found: {login}")
    return self._users[login]

  def get_organization(self, login: str) -> FakeGithubOrganization:
    if login not in self._organizations:
      raise FakeGithubError(f"organization not found: {login}")
    return self._organizations[login]
