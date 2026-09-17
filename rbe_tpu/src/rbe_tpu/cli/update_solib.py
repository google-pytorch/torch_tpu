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

"""Synchronizes C++ shared libraries across the Cloud TPU VM fleet in parallel."""

from __future__ import annotations

from collections.abc import Sequence
import concurrent.futures
import dataclasses
import subprocess
import sys
from typing import Final

DEFAULT_PROJECT: Final[str] = "rbe-tpu-oss"
DEFAULT_BUCKET: Final[str] = "torch-tpu-mailbox-eval"
DEFAULT_CONCURRENCY: Final[int] = 8

HELP_TEXT: Final[str] = """Usage: rbe_tpu/deploy/update_fleet_solib.sh [options]

Fetches inventory of TPU VMs in the project and syncs the solib base layer in parallel.

Options:
  --project PROJECT     GCP project containing TPU VMs (default: rbe-tpu-oss).
  --bucket BUCKET       GCS bucket containing base/solib.tar.gz (default: torch-tpu-mailbox-eval).
  --concurrency N       Max parallel SSH sync jobs (default: 8).
  --dry-run             List target VMs without executing sync.
  -h, --help            Show this help message."""


class CLIError(Exception):
  """Raised when invalid CLI arguments are given."""


@dataclasses.dataclass(frozen=True)
class Options:
  project: str = DEFAULT_PROJECT
  bucket: str = DEFAULT_BUCKET
  concurrency: int = DEFAULT_CONCURRENCY
  dry_run: bool = False
  show_help: bool = False


def log(msg: str) -> None:
  print(f"[solib-sync] {msg}", flush=True)


def parse_args(argv: Sequence[str]) -> Options:
  project = DEFAULT_PROJECT
  bucket = DEFAULT_BUCKET
  concurrency = DEFAULT_CONCURRENCY
  dry_run = False

  idx = 0
  n = len(argv)
  while idx < n:
    arg = argv[idx]
    if arg in ("-h", "--help"):
      return Options(show_help=True)
    if arg == "--project":
      if idx + 1 >= n:
        raise CLIError("--project requires an argument.")
      project = argv[idx + 1]
      idx += 2
    elif arg == "--bucket":
      if idx + 1 >= n:
        raise CLIError("--bucket requires an argument.")
      bucket = argv[idx + 1]
      idx += 2
    elif arg == "--concurrency":
      if idx + 1 >= n:
        raise CLIError("--concurrency requires an argument.")
      if not argv[idx + 1].isdigit():
        raise CLIError("--concurrency must be a positive integer.")
      concurrency = int(argv[idx + 1])
      idx += 2
    elif arg == "--dry-run":
      dry_run = True
      idx += 1
    elif arg.startswith("-"):
      raise CLIError(f"Unknown option: {arg}")
    else:
      raise CLIError(f"Unexpected positional argument: {arg}")
  return Options(
      project=project,
      bucket=bucket,
      concurrency=concurrency,
      dry_run=dry_run,
  )


def parse_vm_list(raw_output: str) -> list[tuple[str, str]]:
  vms: list[tuple[str, str]] = []
  for line in raw_output.splitlines():
    parts = line.strip().split()
    if len(parts) >= 2:
      vms.append((parts[0], parts[1]))
  return vms


def fetch_vm_inventory(project: str) -> list[tuple[str, str]]:
  log(f"Fetching TPU VM inventory from project {project}...")
  res = subprocess.run(
      [
          "gcloud",
          "compute",
          "tpus",
          "tpu-vm",
          "list",
          "--zone=-",
          f"--project={project}",
          "--format=value(name.basename(), name.segment(3))",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  if res.returncode != 0:
    print(
        f"ERROR [solib-sync]: Failed to list TPU VMs: {res.stderr}",
        file=sys.stderr,
    )
    return []
  return parse_vm_list(res.stdout)


def build_remote_command(bucket: str) -> str:
  return f"""TOKEN=$(python3 -c "import urllib.request, json; req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token', headers={{'Metadata-Flavor': 'Google'}}); print(json.loads(urllib.request.urlopen(req).read().decode())['access_token'])")
curl -s -f -H "Authorization: Bearer $TOKEN" "https://storage.googleapis.com/storage/v1/b/{bucket}/o/base%2Fsolib.tar.gz?alt=media" -o /tmp/solib.tar.gz
sudo tar -xzf /tmp/solib.tar.gz -C /tmp/torch_tpu_relay/base
rm -f /tmp/solib.tar.gz
sudo sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 || true"""


def update_one_vm(
    vm: str, zone: str, project: str, remote_cmd: str
) -> bool:
  log(f"Starting {vm} in {zone}...")
  res = subprocess.run(
      [
          "gcloud",
          "compute",
          "tpus",
          "tpu-vm",
          "ssh",
          vm,
          f"--zone={zone}",
          f"--project={project}",
          "--",
          remote_cmd,
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  if res.returncode != 0:
    print(f"[solib-sync] ERROR: failed for {vm} ({zone})", file=sys.stderr)
    return False
  log(f"Completed {vm} ({zone})")
  return True


def sync_fleet(
    vms: Sequence[tuple[str, str]],
    project: str,
    bucket: str,
    concurrency: int,
) -> bool:
  vm_count = len(vms)
  log(f"Syncing solib to {vm_count} VMs with concurrency {concurrency}...")
  remote_cmd = build_remote_command(bucket)
  all_ok = True
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=concurrency
  ) as executor:
    futures = {
        executor.submit(
            update_one_vm, vm, zone, project, remote_cmd
        ): (vm, zone)
        for vm, zone in vms
    }
    for fut in concurrent.futures.as_completed(futures):
      vm, zone = futures[fut]
      try:
        if not fut.result():
          all_ok = False
      except Exception as err:  # pylint: disable=broad-except
        print(
            f"[solib-sync] ERROR: Exception updating {vm} ({zone}): {err}",
            file=sys.stderr,
        )
        all_ok = False
  if all_ok:
    log("Solib fleet sync complete.")
  return all_ok


def main(argv: Sequence[str] | None = None) -> int:
  args = sys.argv[1:] if argv is None else argv
  try:
    opts = parse_args(args)
  except CLIError as err:
    print(f"ERROR [solib-sync]: {err}", file=sys.stderr)
    return 1

  if opts.show_help:
    print(HELP_TEXT)
    return 0

  vms = fetch_vm_inventory(opts.project)
  if not vms:
    print("ERROR [solib-sync]: No TPU VMs found.", file=sys.stderr)
    return 1

  if opts.dry_run:
    print(f"[DRY RUN] Found {len(vms)} TPU VM(s) in project {opts.project}:")
    for vm, zone in vms:
      print(f"  {vm} ({zone})")
    return 0

  ok = sync_fleet(vms, opts.project, opts.bucket, opts.concurrency)
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
