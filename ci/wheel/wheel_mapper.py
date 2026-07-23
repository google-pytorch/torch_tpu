# Copyright 2025 Google LLC
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

"""Remap pywrap binaries for Python Wheel."""

import argparse
import json
import os
import re
import shutil


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--manifest",
      required=True,
      action="append",
      help=(
          "Path to a .json mapping of pywrap binaries. May be repeated (one per"
          " pywrap_library) for the multi-version wheel; the per-version common"
          " libraries share the single libxla_base.so."
      ),
  )
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("binaries", nargs="+")
  args = parser.parse_args()

  binary_map = {os.path.basename(p): p for p in args.binaries}

  mapping = {}
  for manifest_path in args.manifest:
    with open(manifest_path, "r") as f:
      mapping.update(json.load(f))

  for original_path, dest_full_path in mapping.items():
    if not dest_full_path:
      continue

    basename = os.path.basename(original_path)
    if basename not in binary_map:
      continue

    src = binary_map[basename]
    if dest_full_path.startswith(original_path) and len(dest_full_path) > len(
        original_path
    ):
      # pywrap_binaries derives each destination as "<root><pkg>/<basename>",
      # where <root> comes from splitting the binary's path on its short_path.
      # For a binary built in an external repository (our generated glue
      # packages) the short_path spells the repo "../<repo>/" while the path
      # spells it "external/<repo>/", so the split never matches and the "root"
      # degenerates to the binary's entire path -- which is also this mapping's
      # key. The intended "<package>/<basename>" tail survives after it.
      clean_dest_path = dest_full_path[len(original_path) :].lstrip("/")
    else:
      clean_dest_path = re.sub(r"^.*?/bin/", "", dest_full_path)

    dest = os.path.join(args.out_dir, clean_dest_path)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
      os.remove(dest)

    try:
      os.link(src, dest)
    except OSError:
      shutil.copy2(src, dest)


if __name__ == "__main__":
  main()
