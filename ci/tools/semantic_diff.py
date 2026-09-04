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

"""Semantic and AST-based diff analyzer for CI presubmit triage.

Determines whether changes between BASE_SHA and HEAD_SHA in source files
(.py, .cc, .cpp, .h) are purely cosmetic/comments or involve genuine AST/code
changes.
"""

import argparse
import ast
import os
import pathlib
import subprocess
import sys
from typing import List, Tuple


def _strip_docstring_from_body(body: List[ast.stmt]) -> List[ast.stmt]:
  """Removes the docstring statement from a block of statements if present."""
  if not body:
    return body
  first = body[0]
  if isinstance(first, ast.Expr):
    val = first.value
    if isinstance(val, ast.Constant) and isinstance(val.value, str):
      return body[1:]
    elif (
        hasattr(ast, "Str")
        and isinstance(val, ast.Str)
        and isinstance(val.s, str)
    ):
      return body[1:]
  return body


class _DocstringStripper(ast.NodeTransformer):
  """AST transformer that removes module, class, and function docstrings."""

  def visit_Module(self, node: ast.Module) -> ast.Module:
    node.body = _strip_docstring_from_body(node.body)
    self.generic_visit(node)
    return node

  def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
    node.body = _strip_docstring_from_body(node.body)
    self.generic_visit(node)
    return node

  def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
    node.body = _strip_docstring_from_body(node.body)
    self.generic_visit(node)
    return node

  def visit_AsyncFunctionDef(
      self, node: ast.AsyncFunctionDef
  ) -> ast.AsyncFunctionDef:
    node.body = _strip_docstring_from_body(node.body)
    self.generic_visit(node)
    return node


def is_python_semantic_change(base_code: str, head_code: str) -> bool:
  """Compares Python code snippets ignoring comments, whitespace, and docstrings.

  Args:
    base_code: Source string from base commit.
    head_code: Source string from head commit.

  Returns:
    True if there is a functional/AST change, False if
    comments/docstrings/formatting only.
  """
  try:
    base_ast = ast.parse(base_code)
    head_ast = ast.parse(head_code)
  except SyntaxError:
    # Gracefully fail open on syntax error and treat as semantic change
    return True
  except Exception:
    return True

  base_clean = _DocstringStripper().visit(base_ast)
  head_clean = _DocstringStripper().visit(head_ast)

  dump_base = ast.dump(base_clean, include_attributes=False)
  dump_head = ast.dump(head_clean, include_attributes=False)

  return dump_base != dump_head


def strip_cpp_comments_and_normalize(code: str) -> str:
  """Strips C/C++ comments (// and /* ...

  */) while respecting strings, and collapses whitespace.
  """
  result = []
  i = 0
  n = len(code)

  STATE_NORMAL = 0
  STATE_SINGLE_COMMENT = 1
  STATE_MULTI_COMMENT = 2
  STATE_STRING = 3
  STATE_CHAR = 4

  state = STATE_NORMAL

  while i < n:
    ch = code[i]
    next_ch = code[i + 1] if i + 1 < n else ""

    if state == STATE_NORMAL:
      if ch == "/" and next_ch == "/":
        state = STATE_SINGLE_COMMENT
        i += 2
        result.append(" ")
        continue
      elif ch == "/" and next_ch == "*":
        state = STATE_MULTI_COMMENT
        i += 2
        result.append(" ")
        continue
      elif ch == '"':
        state = STATE_STRING
        result.append(ch)
        i += 1
        continue
      elif ch == "'":
        state = STATE_CHAR
        result.append(ch)
        i += 1
        continue
      else:
        result.append(ch)
        i += 1
        continue

    elif state == STATE_SINGLE_COMMENT:
      if ch == "\n":
        state = STATE_NORMAL
        result.append("\n")
        i += 1
      else:
        i += 1

    elif state == STATE_MULTI_COMMENT:
      if ch == "*" and next_ch == "/":
        state = STATE_NORMAL
        i += 2
        result.append(" ")
      else:
        i += 1

    elif state == STATE_STRING:
      result.append(ch)
      if ch == "\\" and i + 1 < n:
        result.append(code[i + 1])
        i += 2
      elif ch == '"':
        state = STATE_NORMAL
        i += 1
      else:
        i += 1

    elif state == STATE_CHAR:
      result.append(ch)
      if ch == "\\" and i + 1 < n:
        result.append(code[i + 1])
        i += 2
      elif ch == "'":
        state = STATE_NORMAL
        i += 1
      else:
        i += 1

  tokens = "".join(result).split()
  return " ".join(tokens)


def is_cpp_semantic_change(base_code: str, head_code: str) -> bool:
  """Compares C/C++ code snippets ignoring comments and whitespace.

  Args:
    base_code: Source string from base commit.
    head_code: Source string from head commit.

  Returns:
    True if there is a functional statement/token change, False if
    comment/formatting only.
  """
  norm_base = strip_cpp_comments_and_normalize(base_code)
  norm_head = strip_cpp_comments_and_normalize(head_code)
  return norm_base != norm_head


def get_git_file_content(sha: str, rel_path: str) -> Tuple[bool, str]:
  """Retrieves file content at a given Git SHA."""
  try:
    res = subprocess.run(
        ["git", "show", f"{sha}:{rel_path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return True, res.stdout
  except subprocess.CalledProcessError:
    return False, ""


def check_file_semantic_change(
    rel_path: str, base_sha: str, head_sha: str
) -> bool:
  """Analyzes a single file diff between base_sha and head_sha.

  Returns True if file has semantic changes (or unsupported/error), False
  otherwise.
  """
  ext = pathlib.Path(rel_path).suffix.lower()
  if ext not in (".py", ".cc", ".cpp", ".c", ".h", ".hpp", ".cxx"):
    # Unsupported file type -> treat as semantic change
    return True

  # Get base content
  ok_base, base_code = get_git_file_content(base_sha, rel_path)
  if not ok_base:
    # File added or base unreadable -> treat as semantic change
    return True

  # Get head content (either from head_sha or disk if head_sha is HEAD/WORKING/empty)
  if head_sha and head_sha not in ("HEAD", "WORKING"):
    ok_head, head_code = get_git_file_content(head_sha, rel_path)
    if not ok_head:
      return True
  else:
    # Read from local filesystem if accessible
    if os.path.exists(rel_path):
      try:
        with open(rel_path, "r", encoding="utf-8") as f:
          head_code = f.read()
      except Exception:
        return True
    else:
      ok_head, head_code = get_git_file_content("HEAD", rel_path)
      if not ok_head:
        return True

  if ext == ".py":
    return is_python_semantic_change(base_code, head_code)
  else:
    return is_cpp_semantic_change(base_code, head_code)


def main():
  parser = argparse.ArgumentParser(
      description="Analyze semantic code changes between Git commits."
  )
  parser.add_argument(
      "--base-sha", default="", help="Base commit SHA for comparison"
  )
  parser.add_argument(
      "--head-sha", default="HEAD", help="Head commit SHA for comparison"
  )
  parser.add_argument(
      "--files",
      nargs="*",
      default=[],
      help="List of changed file paths to evaluate",
  )
  parser.add_argument(
      "--base-content", default=None, help="Direct base code string for testing"
  )
  parser.add_argument(
      "--head-content", default=None, help="Direct head code string for testing"
  )
  parser.add_argument(
      "--file-type",
      default="py",
      help="File extension type when passing direct content (e.g. py, cc)",
  )

  args = parser.parse_args()

  # Direct string testing mode
  if args.base_content is not None and args.head_content is not None:
    ext = (
        args.file_type
        if args.file_type.startswith(".")
        else f".{args.file_type}"
    )
    if ext == ".py":
      has_change = is_python_semantic_change(
          args.base_content, args.head_content
      )
    else:
      has_change = is_cpp_semantic_change(args.base_content, args.head_content)

    if has_change:
      print("has_semantic_changes=true")
      sys.exit(1)
    else:
      print("has_semantic_changes=false")
      sys.exit(0)

  if not args.files:
    # If no files specified, evaluate git diff --name-only between base and head
    if args.base_sha:
      try:
        res = subprocess.run(
            ["git", "diff", "--name-only", args.base_sha, args.head_sha],
            capture_output=True,
            text=True,
            check=True,
        )
        args.files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
      except Exception:
        print("has_semantic_changes=true")
        sys.exit(1)

  if not args.files:
    print("has_semantic_changes=false")
    sys.exit(0)

  has_any_semantic_change = False
  for f in args.files:
    if check_file_semantic_change(f, args.base_sha, args.head_sha):
      has_any_semantic_change = True
      break

  if has_any_semantic_change:
    print("has_semantic_changes=true")
    sys.exit(1)
  else:
    print("has_semantic_changes=false")
    sys.exit(0)


if __name__ == "__main__":
  main()
