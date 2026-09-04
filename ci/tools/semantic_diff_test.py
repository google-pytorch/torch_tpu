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

"""Unit tests for AST and semantic diff analyzer tool (semantic_diff.py)."""

import importlib
import unittest

try:
  semantic_diff = importlib.import_module("ci.tools.semantic_diff")
except ImportError:
  semantic_diff = importlib.import_module("semantic_diff")


class SemanticDiffTest(  #
    unittest.TestCase  # UNITTEST_OK=CI tool unit test without tensor RNG
):
  """Validates Python AST and C++ comment/formatting diff analyzer logic."""

  def test_python_comments_or_docstrings_only_returns_false(self):
    """Python changes with only comments, docstrings, or whitespace return False."""
    base_code = '''"""Module docstring base."""
# Base comment
def foo(x):
    """Function docstring base."""
    return x + 1
'''
    head_code = '''"""Module docstring updated."""
# Updated comment
def foo(x):
    """Function docstring updated."""
    return x + 1
'''
    self.assertFalse(
        semantic_diff.is_python_semantic_change(base_code, head_code)
    )

  def test_python_logic_change_returns_true(self):
    """Python changes with variable, logic, or statement edits return True."""
    base_code = "x = 1\n"
    head_code = "x = 2\n"
    self.assertTrue(
        semantic_diff.is_python_semantic_change(base_code, head_code)
    )

  def test_python_syntax_error_fallback_returns_true(self):
    """Python files with syntax errors gracefully fallback to True."""
    base_code = "def foo(:"
    head_code = "def foo(x): return x"
    self.assertTrue(
        semantic_diff.is_python_semantic_change(base_code, head_code)
    )

  def test_cpp_comments_or_formatting_only_returns_false(self):
    """C++ changes with only // or /* ... */ comment edits return False."""
    base_code = """// Single line comment
/* Multi-line
 * comment
 */
int add(int a, int b) {
    return a + b;
}
"""
    head_code = """// Modified comment
/* Updated multi-line
 * comment
 */
int add(int a, int b) {
    return a + b;
}
"""
    self.assertFalse(semantic_diff.is_cpp_semantic_change(base_code, head_code))

  def test_cpp_statement_or_parameter_change_returns_true(self):
    """C++ changes with statement or parameter edits return True."""
    base_code = "int add(int a, int b) { return a + b; }"
    head_code = "int add(int a, int b, int c) { return a + b + c; }"
    self.assertTrue(semantic_diff.is_cpp_semantic_change(base_code, head_code))

  def test_cpp_comments_inside_string_literals_preserved(self):
    """Comment markers inside C++ string literals are treated as code."""
    base_code = 'std::string s = "http://google.com";'
    head_code = 'std::string s = "https://google.com";'
    self.assertTrue(semantic_diff.is_cpp_semantic_change(base_code, head_code))


if __name__ == "__main__":
  unittest.main()
