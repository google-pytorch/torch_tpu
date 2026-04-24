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

"""Lint test to verify that large .gz files and their Git LFS pointers stay in sync."""

import hashlib
from absl.testing import absltest
from etils import epath


class LfsSyncTest(absltest.TestCase):

  def test_lfs_pointers_are_in_sync(self):
    """Validates that every .gz file matches its pointer OID and file size in internal builds."""
    pkg_dir = epath.resource_path("torch_tpu") / "tests"
    gz_files = list(pkg_dir.glob("*.gz"))
    self.assertNotEmpty(gz_files, "No .gz files found in tests/ directory.")

    for gz_file in gz_files:
      lfs_file = epath.Path(str(gz_file) + ".lfs")

      # 1. Ensure the file exists
      self.assertTrue(
          lfs_file.exists(),
          f"\n[ERROR] Missing LFS pointer file for {gz_file.name}!\nTo fix this"
          " issue and generate/update the pointer file, please run locally:\n "
          " python3 third_party/py/torch_tpu/tests/lfs_sync.py --fix\n",
      )

      lfs_text = lfs_file.read_text()
      with gz_file.open("rb") as f:
        file_data = f.read()

      computed_sha = hashlib.sha256(file_data).hexdigest()
      computed_size = len(file_data)

      expected_oid = f"oid sha256:{computed_sha}"
      expected_size = f"size {computed_size}"

      # 2. Verify OID
      self.assertIn(
          expected_oid,
          lfs_text,
          f"\n[ERROR] File {gz_file.name} has test OID checksum mismatch"
          f" against {lfs_file.name}!\n"
          f"To fix this locally in CitC, please run:\n"
          f"  python3 third_party/py/torch_tpu/tests/lfs_sync.py --fix\n\n"
          f"Then fully commit and push the new payload to GitHub LFS from your"
          f" workstation:\n"
          f"  git add tests/{gz_file.name}\n"
          f'  git commit -m "Update golden file"\n'
          f"  git lfs push origin --all\n"
          f"  git push origin your-branch-name\n",
      )

      # 3. Verify Size
      self.assertIn(
          expected_size,
          lfs_text,
          f"\n[ERROR] File {gz_file.name} has byte size mismatch against"
          f" {lfs_file.name}!\nTo update the pointer locally in CitC, please"
          " run:\n  python3 third_party/py/torch_tpu/tests/lfs_sync.py"
          " --fix\n",
      )


if __name__ == "__main__":
  absltest.main()
