#!/usr/bin/env python3
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

"""Generate the notebooks in docs/notebooks/ from their Markdown sources.

Converts Markdown files that contain an "Open in Colab" snippet.

Example usage:
  # Convert all sources with "Open in Colab" snippet.
  $ tools/generate_doc_notebooks.py

  # Convert only docs/NAME.md if it has the snippet.
  $ tools/generate_doc_notebooks.py NAME

NAME can be a basename or a path such as docs/NAME.md. Files without the
"Open in Colab" snippet are skipped, so the pre-commit hook can pass every
changed Markdown file under docs/. Passing pyproject.toml converts all sources
with the snippet, since the notebooks record the version set there.
"""

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Matches the "Open in Colab" markdown link (case-insensitive).
# Pattern: [![Open In Colab]...](...)
_COLAB_SNIPPET = re.compile(r"\[!\[Open in Colab\]\([^)]*\)\]\([^)]*\)", re.IGNORECASE)

# Matches the start of a relative link target, up to any .md suffix or anchor.
# Targets with a scheme (https:, mailto:) or that start with # do not match.
_LINK_TARGET = r"\]\((?:\./)?([^)#:\s]+)"
_MARKDOWN_LINK = re.compile(_LINK_TARGET + r"\.md(#[^)\n]*)?\)")
_OTHER_LINK = re.compile(_LINK_TARGET + r"(#[^)\n]*)?\)")


def read_version() -> str:
  """Returns the torch_tpu version set in pyproject.toml."""
  with (REPO_ROOT / "pyproject.toml").open("rb") as f:
    return tomllib.load(f)["project"]["version"]


def docs_site_url(version: str) -> str:
  """Returns the Read the Docs URL that notebook links should point to.

  A released version has a vX.Y.Z git tag and its docs are served at
  /en/X.Y.Z. Unreleased versions link to /en/latest. Releases are detected
  from local tags, so fetch them first (git fetch --tags upstream).

  Args:
      version: The torch_tpu version.

  Returns:
      The URL of the docs site, without a trailing slash.
  """
  tag_check = subprocess.run(
    ["git", "rev-parse", "-q", "--verify", f"refs/tags/v{version}"],
    cwd=REPO_ROOT,
    capture_output=True,
    check=False,
  )
  docs_version = version if tag_check.returncode == 0 else "latest"
  return f"https://torchtpu.readthedocs.io/en/{docs_version}"


def rewrite_links(markdown: str, site_url: str) -> str:
  """Rewrites relative link targets to point to the docs site.

  Notebooks are opened outside of the docs site (for example, in Colab), where
  relative links do not resolve. mkdocs.yml sets use_directory_urls, so
  page.md#anchor is served at page/#anchor and index.md at the site root.
  Other relative targets, such as images, keep their path.

  Args:
      markdown: The Markdown text.
      site_url: The URL of the docs site, without a trailing slash.

  Returns:
      The Markdown text with its relative links rewritten.
  """
  markdown = _MARKDOWN_LINK.sub(rf"]({site_url}/\1/\2)", markdown)
  markdown = markdown.replace(f"]({site_url}/index/", f"]({site_url}/")
  return _OTHER_LINK.sub(rf"]({site_url}/\1\2)", markdown)


def strip_colab_snippet(markdown: str) -> str:
  """Removes the 'Open in Colab' HTML comment from Markdown."""
  return _COLAB_SNIPPET.sub("", markdown)


def has_colab_snippet(path: pathlib.Path) -> bool:
  """Returns True if the file contains an 'Open in Colab' snippet."""
  try:
    content = path.read_text(encoding="utf-8")
    return bool(_COLAB_SNIPPET.search(content))
  except (OSError, UnicodeDecodeError):
    return False


def select_sources(names: list[str]) -> list[str]:
  """Returns stems of Markdown files with 'Open in Colab' snippet."""
  if not names:
    docs_dir = REPO_ROOT / "docs"
    sources = [
      f.stem for f in docs_dir.glob("*.md")
      if f.is_file() and has_colab_snippet(f)
    ]
    return sorted(sources)

  if "pyproject.toml" in names:
    return select_sources([])

  stems = []
  for name in names:
    path = pathlib.Path(name)
    stem = path.name.removesuffix(".md")
    source_file = REPO_ROOT / "docs" / f"{stem}.md"
    if has_colab_snippet(source_file):
      stems.append(stem)
  return stems


def main() -> None:
  """Converts Markdown files with 'Open in Colab' snippet to notebooks."""
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "names",
    nargs="*",
    metavar="NAME",
    help="docs to convert (default: all docs with 'Open in Colab' snippet)",
  )
  args = parser.parse_args()

  if shutil.which("jupytext") is None:
    sys.exit(
      "error: jupytext not found. Install the docs dependencies with:\n"
      "  python -m pip install -r docs/requirements.txt"
    )

  version = read_version()
  site_url = docs_site_url(version)

  # The accelerator and colab entries make the "Open in Colab" links start a
  # TPU runtime. torch_tpu records the version that the notebook was generated
  # for.
  metadata = json.dumps({
    "accelerator": "TPU",
    "colab": {
      "gpuType": "V5E1",
      "machine_shape": "hm",
      "provenance": [],
    },
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "torch_tpu": {"version": version},
  })

  for name in select_sources(args.names):
    source = REPO_ROOT / "docs" / f"{name}.md"
    if not source.is_file():
      sys.exit(f"error: {source} does not exist")

    # Read source and strip the colab snippet before conversion.
    markdown = source.read_text(encoding="utf-8")
    markdown = strip_colab_snippet(markdown)
    markdown = rewrite_links(markdown, site_url)

    # --update reuses the cell ids of the existing notebook, so that
    # unchanged sources produce identical output.
    subprocess.run(
      [
        "jupytext",
        "--from",
        "md",
        "--to",
        "ipynb",
        "--update",
        "--update-metadata",
        metadata,
        "--output",
        f"docs/notebooks/{name}.ipynb",
        "-",
      ],
      cwd=REPO_ROOT,
      input=markdown,
      encoding="utf-8",
      check=True,
    )


if __name__ == "__main__":
  main()
