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

"""Categories of the warnings TorchTPU emits.

Like PyTorch, TorchTPU emits its warnings, from both Python and C++, with
Python's built-in warning categories: `DeprecationWarning` for deprecated APIs
and `UserWarning` for everything else. To tell TorchTPU warnings apart, every
message starts with a prefix naming its category in the following hierarchy:

  TorchTpu
  ├── Performance
  │   ├── UnoptimizedBuild
  │   └── TcMalloc
  ├── Correctness
  └── ApiStage
      ├── InternalApi
      ├── ExperimentalApi
      └── DeprecatedApi

Prefix format
-------------

The prefix is `<`, the path of the category with every segment followed by
`/`, then `> `. For example:

  <TorchTpu/Performance/TcMalloc/> tcmalloc is not loaded; ...

Filtering
---------

Python matches the message of a warning filter against the *start* of the
warning message, so with this format:

  * `<TorchTpu/Performance/` matches `Performance` and all its subcategories;
  * `<TorchTpu/Performance/>` matches `Performance` only;
  * `<TorchTpu/` matches all TorchTPU warnings.

That is, end the filter with `/` to select a category and its subcategories,
or with `/>` to select just that category. Don't drop the trailing `/`:
`<TorchTpu/Performance` would also match a future sibling category such as
`PerformanceFoo`.

The same filter text works in all the usual places:

  # In code.
  import warnings
  warnings.filterwarnings("ignore", message="<TorchTpu/Performance/")
  warnings.filterwarnings("error", message="<TorchTpu/")

  # From the environment or the command line. This needs no import, so it also
  # applies to warnings emitted while `import torch_tpu` is still running, such
  # as the tcmalloc warning. Quote the filter, since `<` and `>` are special to
  # the shell.
  export PYTHONWARNINGS="ignore:<TorchTpu/Performance/"
  python -W "ignore:<TorchTpu/Performance/TcMalloc/>" ...
  pytest -W "error:<TorchTpu/" ...

  # In pytest configuration.
  [pytest]
  filterwarnings =
      ignore:<TorchTpu/Performance/

`PYTHONWARNINGS` takes a comma-separated list, so several filters can be
combined:

  PYTHONWARNINGS="ignore:<TorchTpu/Performance/,error:<TorchTpu/ApiStage/"

Design rationale
----------------

The prefix format is chosen so that a filter is the same plain string
everywhere, with no quoting or escaping beyond the shell:

  * No `:`, `,`, or whitespace inside the path or a filter, since `-W` and
    `PYTHONWARNINGS` split filters on `:` and `,` and strip whitespace from
    each field.
  * No regex metacharacters (e.g. `.`, `[`, `]`), since
    `warnings.filterwarnings` and pytest's `filterwarnings` ini option and mark
    treat the message as a regex, while `-W` and `PYTHONWARNINGS` treat it as
    literal text. With `<`, `/`, and `>`, both readings agree.
  * A `/` after every segment, including the last, so that a filter for a
    category can't match a sibling whose name it is a prefix of, and so that the
    most common filter, a category and its subcategories, is a single filter
    that keeps working when subcategories are added.
  * The closing `>` marks the end of the path, so that a single category can be
    selected exactly.

The categories and their prefixes are defined in C++ (csrc/common/warnings.h)
and exposed here as module-level constants.
"""

from typing import Final

from torch_tpu._internal import warnings_impl

WarningCategory = warnings_impl.WarningCategory

# go/keep-sorted start
API_STAGE: Final[WarningCategory] = WarningCategory.API_STAGE
CORRECTNESS: Final[WarningCategory] = WarningCategory.CORRECTNESS
DEPRECATED_API: Final[WarningCategory] = WarningCategory.DEPRECATED_API
EXPERIMENTAL_API: Final[WarningCategory] = WarningCategory.EXPERIMENTAL_API
INTERNAL_API: Final[WarningCategory] = WarningCategory.INTERNAL_API
PERFORMANCE: Final[WarningCategory] = WarningCategory.PERFORMANCE
TCMALLOC: Final[WarningCategory] = WarningCategory.TCMALLOC
TORCH_TPU: Final[WarningCategory] = WarningCategory.TORCH_TPU
UNOPTIMIZED_BUILD: Final[WarningCategory] = WarningCategory.UNOPTIMIZED_BUILD
# go/keep-sorted end


def get_prefix(category: WarningCategory) -> str:
  """Returns the prefix of the messages of warnings in `category`."""
  return warnings_impl.get_warning_prefix(category)


def get_builtin_category(category: WarningCategory) -> type[Warning]:
  """Returns the built-in Python warning category to emit `category` with."""
  if warnings_impl.is_deprecation_warning(category):
    return DeprecationWarning
  return UserWarning


def format_message(category: WarningCategory, message: str) -> str:
  """Returns `message` prefixed with the prefix of `category`."""
  return get_prefix(category) + message
