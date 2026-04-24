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

"""Local utility to create or update .lfs pointers or download missing .gz files in CitC."""

import base64
import hashlib
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request


def sync_lfs_environment(branch="main", token=None):
  """Creates or updates .gz.lfs files, or pulls down missing .gz binaries."""
  pkg_dir = pathlib.Path(__file__).parent.resolve()

  # 1. Scan for missing .gz files and download them from GitHub LFS
  lfs_files = list(pkg_dir.glob("*.gz.lfs"))
  print(
      f"Scanning {len(lfs_files)} pointer files for missing binaries in"
      f" {pkg_dir}..."
  )
  for lfs_file in lfs_files:
    gz_file = lfs_file.with_suffix("")
    if not gz_file.exists():
      print(
          f"Missing binary file: {gz_file.name}. Attempting to pull from GitHub"
          " LFS cache via API..."
      )

      lfs_text = lfs_file.read_text()
      oid = None
      size = None
      for line in lfs_text.splitlines():
        if line.startswith("oid sha256:"):
          oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
          size = int(line.split(" ", 1)[1].strip())

      downloaded = False
      if oid and size:
        urls = [
            "https://github.com/google-pytorch/torch_tpu.git/info/lfs/objects/batch",
            "https://github.com/google-pytorch/torch_tpu/info/lfs/objects/batch",
        ]
        headers = {
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        }

        t = token or os.environ.get("GITHUB_TOKEN")
        payload = {
            "operation": "download",
            "transfers": ["basic"],
            "ref": {"name": f"refs/heads/{branch}"},
            "objects": [{"oid": oid, "size": size}],
        }

        # Test Basic Auth using multiple fallback usernames
        usernames = ["git"]
        for url in urls:
          for u in usernames:
            if not u or not t:
              continue
            try:
              auth = base64.b64encode(f"{u}:{t}".encode("utf-8")).decode(
                  "utf-8"
              )
              headers["Authorization"] = f"Basic {auth}"
              req = urllib.request.Request(
                  url,
                  data=json.dumps(payload).encode("utf-8"),
                  headers=headers,
                  method="POST",
              )
              with urllib.request.urlopen(req) as res:
                response_data = json.loads(res.read().decode("utf-8"))
                obj = response_data.get("objects", [{}])[0]
                download_url = (
                    obj.get("actions", {}).get("download", {}).get("href")
                )
                if download_url:
                  urllib.request.urlretrieve(download_url, str(gz_file))
                  print(
                      f"✓ Successfully downloaded {gz_file.name} from GitHub"
                      " LFS cache!"
                  )
                  downloaded = True
                  break
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                ValueError,
                TypeError,
            ):
              pass
          if downloaded:
            break

      if not downloaded:
        print(
            f"Failed to download {gz_file.name} from GitHub LFS cache using"
            f" OID: {oid}"
        )

  # 2. Scan for new/updated .gz binaries and update the .lfs files
  gz_files = list(pkg_dir.glob("*.gz"))
  print(f"Scanning {len(gz_files)} .gz files to sync pointer contents...")
  updated = 0
  for gz_file in gz_files:
    lfs_file = gz_file.with_suffix(gz_file.suffix + ".lfs")
    with gz_file.open("rb") as f:
      data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    size = len(data)

    new_text = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha}\n"
        f"size {size}\n"
    )

    should_update = True
    if lfs_file.exists():
      if new_text.strip() == lfs_file.read_text().strip():
        should_update = False

    if should_update:
      print(f"✓ Regenerating pointer: {lfs_file.name} for {gz_file.name}")
      lfs_file.write_text(new_text)
      updated += 1

  print("Success: Git LFS files are fully synchronized!")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--fix":
    target_branch = "main"
    auth_token = None
    for arg in sys.argv:
      if arg.startswith("--branch="):
        target_branch = arg.split("=", 1)[1]
      elif arg.startswith("--token="):
        auth_token = arg.split("=", 1)[1]
    sync_lfs_environment(branch=target_branch, token=auth_token)
  else:
    print("Usage: python3 third_party/py/torch_tpu/tests/lfs_sync.py --fix")
