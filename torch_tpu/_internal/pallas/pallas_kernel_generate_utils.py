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

"""Utility functions for generating dynamic kernels exports from Pallas."""


def generate_embedded_file(
    header_path: str, implementation_path: str, data: list[tuple[str, bytes]]
):
  """Generates an embedded file with the given data.

  Args:
    header_path: The path to the header file to generate.
    implementation_path: The path to the implementation file to generate.
    data: A list of tuples containing the data to export. Each data tuple
      contains the variable name and the data itself as a byte array.

  An implementation file will be generated with the following declarations for
  each data tuple. The header will contain the declarations.
    const unsigned char <varname>_data[] = ...;
    const int <varname>_len = ...;
  """

  with open(header_path, "w") as hdr_f, open(
      implementation_path, "w"
  ) as impl_f:
    include_guard = (
        header_path.upper()
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        + "_"
    )
    hdr_f.write(f"#ifndef {include_guard}\n")
    hdr_f.write(f"#define {include_guard}\n")
    for d in data:
      hdr_f.write(f"extern const unsigned char {d[0]}_data[];\n")
      hdr_f.write(f"extern const unsigned int {d[0]}_len;\n")
      impl_f.write(
          f"extern const unsigned char {d[0]}_data[] ="
          f" {{{','.join(['0x%x' % c for c in d[1]])}}};\n"
      )
      impl_f.write(
          f"extern const unsigned int {d[0]}_len = sizeof({d[0]}_data);\n"
      )
    hdr_f.write("#endif\n\n")
