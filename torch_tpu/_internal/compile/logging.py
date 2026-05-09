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

"""Logging utilities for TorchTPU compilation."""

import os
from absl import logging

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

_LOG_COMPILE_ENV = "TORCH_TPU_INTERNAL_LOG_COMPILE"


def log_compile_enabled() -> bool:
  log_compile_env = os.environ.get(_LOG_COMPILE_ENV, "0")
  return log_compile_env.lower() in ("1", "true", "yes")


def debug(msg, *args, **kwargs):
  if log_compile_enabled():
    info(msg, *args, **kwargs)
  else:
    logging.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
  logging.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
  logging.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
  logging.error(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
  logging.exception(msg, *args, **kwargs)


def vlog_is_on(level: int) -> bool:
  if level == logging.DEBUG and log_compile_enabled():
    return True
  return logging.vlog_is_on(level)
