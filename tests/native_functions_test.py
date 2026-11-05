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

"""Parses ATen native function definitions and checks they are registered."""

from collections.abc import Mapping, Set
import os
import re
from typing import Final
from absl import app
from absl import flags
from absl.testing import absltest
import torch
from tests import native_functions_data as data
from tests import seed_test_utils

try:
  from rules_python.python.runfiles import runfiles  # pylint: disable=g-import-not-at-top

  _HAS_RESOURCES = True
except ImportError:
  resources = None
  _HAS_RESOURCES = False

_REGISTRATION_DECLARATIONS_PATH: Final[str] = (
    'third_party/py/torch/aten/src/ATen/RegistrationDeclarations.h'
)

_PRINT_REGISTRATIONS = flags.DEFINE_bool(
    'print_registrations',
    False,
    'Print the registered ops and their signatures.',
)


def get_registered_ops() -> Set[str]:
  """Returns a set of registered ops from the torch dispatcher."""
  return frozenset(
      w.removeprefix('aten::')
      for w in torch._C._dispatch_get_registrations_for_dispatch_key(
          'PrivateUse1'
      )
      if w.startswith('aten::')
  )


LINE_PATTERN = re.compile(
    r"""
    ;\s//\s*                            # find the start of the comment
    .*\"schema\":\s*\"(aten::[^\"]+)\"  # find the op schema
    .*\"dispatch\":\s*\"([^\"]+)\"      # find the dispatch value
    .*\"default\":\s*\"([^\"]+)\"       # find the default value
    """,
    re.VERBOSE,
)


def get_op_schemas(
    filepath: str, include_derived_ops: bool = False
) -> Mapping[str, str]:
  """Returns a dict of ops that require registration to their schema.

  We use third_party/py/torch/aten/src/ATen/RegistrationDeclarations.h and
  https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html#register-kernels-for-the-new-backend
  to determine which ops require registration.

  Args:
    filepath: Path to RegistrationDeclarations.h.

  Returns:
    A dict of ops that require registration to their schema.
  """
  ops = {}
  with open(filepath, 'r') as f:
    matched_lines, total_lines = 0, 0
    for line in f:
      total_lines += 1
      match = LINE_PATTERN.search(line)
      if not match:
        continue
      matched_lines += 1
      op_schema = match.group(1)
      op_name = op_schema.split('(', 1)[0].replace('aten::', '')
      dispatch = match.group(2)
      default = match.group(3)
      if dispatch not in ['True', 'False']:
        raise ValueError(f'Invalid dispatch: {dispatch} for op {op_name}')
      if default not in ['True', 'False']:
        raise ValueError(f'Invalid default: {default} for op {op_name}')
      needs_registration = (
          dispatch == 'True'
          and default == 'False'
          and '_test_' not in op_name
          and 'cudnn' not in op_name
          and op_name not in ['_foobar']
      )
      if needs_registration or include_derived_ops:
        ops[op_name] = op_schema

  print(
      f'RegistrationDeclarations.h has {total_lines} lines,'
      f' {matched_lines} matched an op declaration, and {len(ops)} matched'
      ' an op requiring registration.'
  )
  return ops


class NativeFunctionsTest(seed_test_utils.RepeatableTest):
  """Check op registrations are consistent.

  This test maintains the invariant
    ops_requiring_registration + REGISTRATION_OVERRIDES = registered_ops
    + UNREGISTERED_OPS,
  where + represents disjoint set union, by parsing RegistrationDeclarations.h
  and querying torch for registered ops, to produce ops_requiring_registration
  and registered_ops respectively. We also check that UNREGISTERED_OPS is a
  subset of ops_requiring_registration, to prevent adding garbage to both lists.
  """

  def __init__(self, *args):
    super().__init__(*args)

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.registered_ops = get_registered_ops()
    registration_declarations_path = None
    if _HAS_RESOURCES:
      try:
        registration_declarations_path = runfiles.Create().Rlocation(
            _REGISTRATION_DECLARATIONS_PATH
        )
      except Exception:  # pylint: disable=broad-except
        pass

    if (
        not registration_declarations_path
        or not os.path.exists(registration_declarations_path)
    ):
      raise absltest.SkipTest(
          f'{_REGISTRATION_DECLARATIONS_PATH} not found. Skipping tests.'
      )

    cls.ops_schemas = get_op_schemas(registration_declarations_path)
    cls.ops_requiring_registration = frozenset(cls.ops_schemas.keys())
    # This is the union of REGISTRATION_OVERRIDES and
    # REGISTRATION_OVERRIDES_REGEX matched against registered_ops.
    expanded_overrides = set(data.REGISTRATION_OVERRIDES)
    for pattern_str in data.REGISTRATION_OVERRIDES_REGEX:
      pattern = re.compile(pattern_str)
      matches = {op for op in cls.registered_ops if pattern.fullmatch(op)}
      if not matches:
        raise ValueError(
            f"REGISTRATION_OVERRIDES_REGEX pattern '{pattern_str}' "
            'matched no registered ops.'
        )
      expanded_overrides.update(matches)
    cls.registration_overrides = frozenset(expanded_overrides)
    cls.unregistered_ops = (
        data.UNREGISTERED_OPS & cls.ops_requiring_registration
    )

  def test_all_ops_requiring_registration_union_maybe_unneeded_registered_ops_equals_registered_ops_union_missing_cc_ops(
      self,
  ):
    """Tests that all ops requiring registration union maybe unneeded registered ops equals registered ops union missing cc ops."""
    left = self.ops_requiring_registration | self.registration_overrides
    right = self.registered_ops | self.unregistered_ops
    try:
      self.assertEqual(
          left,
          right,
          '\nops_requiring_registration + REGISTRATION_OVERRIDES !='
          ' registered_ops + UNREGISTERED_OPS',
      )
    except AssertionError as e:
      left_minus_right = left - right
      right_minus_left = right - left
      if left_minus_right:
        e.add_note(
            'Elements of self.ops_requiring_registration |'
            ' self.registration_overrides but not'
            f' self.registered_ops | data.UNREGISTERED_OPS: {left_minus_right}'
        )
      if right_minus_left:
        e.add_note(
            ' self.registered_ops | data.UNREGISTERED_OPS but not'
            ' self.ops_requiring_registration |'
            f' self.registration_overrides: {right_minus_left}',
        )
      raise e

  def test_all_ops_requiring_registration_disjoint_from_maybe_unneeded_registered_ops(
      self,
  ):
    """Tests that all ops requiring registration are not in maybe unneeded registered ops."""
    self.assertEmpty(
        self.ops_requiring_registration & self.registration_overrides,
        '\nThese ops are both required to be registered and in'
        ' REGISTRATION_OVERRIDES. Please remove from REGISTRATION_OVERRIDES.',
    )

  def test_registered_ops_disjoint_from_missing_cc_ops(self):
    """Tests that registered ops are not in missing cc ops."""
    self.assertEmpty(
        self.registered_ops & data.UNREGISTERED_OPS,
        '\nThese ops are both registered and in UNREGISTERED_OPS. Please remove'
        ' from UNREGISTERED_OPS.',
    )

  def test_missing_cc_ops_subset_of_ops_requiring_registration(self):
    """Tests that missing cc ops are a subset of ops requiring registration."""
    if os.environ.get('TEST_WORKSPACE') != 'google3':
      self.skipTest('Only enforced in Google3')
    self.assertContainsSubset(
        data.UNREGISTERED_OPS,
        self.ops_requiring_registration,
        '\nThese ops are defined in UNREGISTERED_OPS but are not required to be'
        ' registered, please remove them from UNREGISTERED_OPS.',
    )

  def test_maybe_unneeded_registered_ops_subset_of_registered_ops(self):
    """Tests that REGISTRATION_OVERRIDES are a subset of registered ops."""
    self.assertContainsSubset(
        self.registration_overrides,
        self.registered_ops,
        '\nThese ops are defined in REGISTRATION_OVERRIDES but are not'
        ' registered, please remove them from REGISTRATION_OVERRIDES.',
    )


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if _PRINT_REGISTRATIONS.value:
    registered_ops = get_registered_ops()
    registration_declarations_path = None
    if _HAS_RESOURCES:
      try:
        registration_declarations_path = runfiles.Create().Rlocation(
            _REGISTRATION_DECLARATIONS_PATH
        )
      except Exception:  # pylint: disable=broad-except
        pass

    if (
        not registration_declarations_path
        or not os.path.exists(registration_declarations_path)
    ):
      print(
          f'{_REGISTRATION_DECLARATIONS_PATH} not found. Cannot print'
          ' registrations.'
      )
      return

    ops_schemas = get_op_schemas(
        registration_declarations_path, include_derived_ops=True
    )
    print('\n\nCurrently registered ops:')
    print(
        '===================================================================='
    )
    for op_name in sorted(registered_ops):
      schema = ops_schemas.get(
          op_name,
          f'Schema not found in RegistrationDeclarations.h for op={op_name}',
      )
      print(f'  {schema}')
  else:
    registration_declarations_path = None
    if _HAS_RESOURCES:
      try:
        registration_declarations_path = runfiles.Create().Rlocation(
            _REGISTRATION_DECLARATIONS_PATH
        )
      except Exception:  # pylint: disable=broad-except
        pass

    if not registration_declarations_path or not os.path.exists(
        registration_declarations_path
    ):
      print(f'{_REGISTRATION_DECLARATIONS_PATH} not found. Skipping tests.')
      return

    absltest.main()


if __name__ == '__main__':
  app.run(main)
