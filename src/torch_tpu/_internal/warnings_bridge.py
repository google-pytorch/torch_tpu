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

"""Bridge to translate C++ TorchTPU warnings into TorchTpuWarning subclasses."""

import re
from typing import Optional, Type
import warnings

from torch_tpu._internal import exceptions

_TT_WARN_TOKEN_RE = re.compile(r"^\x1e\[TT_WARN:(\w+)\]\x1e")


def _decode_warning(
    category: Type[Warning], message_str: str
) -> tuple[Optional[Type[exceptions.TorchTpuWarning]], str]:
  """Decodes a C++ TorchTPU warning message.

  Args:
    category: The Python warning class of the emitted warning.
    message_str: The raw warning string message.

  Returns:
    A tuple of (target_category, clean_message). If the message is a TorchTPU
    C++ warning, target_category is the resolved TorchTpuWarning subclass and
    clean_message is stripped of the delimiter token. Otherwise, returns
    (None, message_str).
  """
  # If it's already a TorchTpuWarning subclass, it was emitted directly in Python
  # (e.g. via warnings.warn(..., exceptions.TorchTpuWarning)) and does not need
  # C++ token remapping.
  if issubclass(category, exceptions.TorchTpuWarning):
    return None, message_str

  token_match = _TT_WARN_TOKEN_RE.match(message_str)
  if token_match:
    class_name = token_match.group(1)
    target_category = getattr(
        exceptions, class_name, exceptions.TorchTpuWarning
    )
    clean_message = _TT_WARN_TOKEN_RE.sub("", message_str)
    return target_category, clean_message

  return None, message_str


_INSTALLED = False
_ORIG_SHOWWARNMSG = None


def install_warning_bridge() -> None:
  """Installs the TorchTPU C++ warning translation hook into Python warnings.

  Refer to third_party/py/torch_tpu/csrc/common/warnings.h for details on how
  the delimiter token is injected and translated.
  """
  global _INSTALLED, _ORIG_SHOWWARNMSG
  if _INSTALLED:
    return

  _ORIG_SHOWWARNMSG = getattr(warnings, "_showwarnmsg", None)

  def _tpu_showwarnmsg(msg: warnings.WarningMessage) -> None:
    """Intercepts and translates TorchTPU C++ warning messages before display."""
    text = str(msg.message)
    new_category, clean_text = _decode_warning(msg.category, text)
    if new_category is not None:
      warnings.warn_explicit(
          clean_text,
          new_category,
          msg.filename,
          msg.lineno,
          source=msg.source,
      )
      return

    if _ORIG_SHOWWARNMSG is not None:
      _ORIG_SHOWWARNMSG(msg)

  setattr(warnings, "_showwarnmsg", _tpu_showwarnmsg)
  _INSTALLED = True
