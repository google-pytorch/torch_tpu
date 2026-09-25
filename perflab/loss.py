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

from absl import logging
import torch


def real_loss(model, input_args, input_kwargs) -> torch.Tensor:
  """Default compute_loss for training step functions.

  Run the model forward pass, produce a scalar loss to call backward() on.

  Note: This doesn't do the naive fallback (`out.loss if hasattr(...) else
  out.sum()`). The fallback measures a different backward graph when a model
  has a fused loss path that wasn't triggered (e.g. it only returns `.loss` when
  passed `labels=`), so it would miss the loss/softmax/cross_entropy backward
  and measure the wrong thing.

  Raises TypeError if it cannot find loss value in the model output.
  """
  out = model(*input_args, **input_kwargs)

  if hasattr(out, "loss") and out.loss is not None:
    loss_method = "out.loss"
    loss = out.loss
  elif isinstance(out, dict) and "loss" in out:
    loss_method = "out['loss']"
    loss = out["loss"]
  elif isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
    loss_method = "out[0]"
    loss = out[0]
  elif torch.is_tensor(out):
    loss_method = "out tensor directly"
    loss = out
  else:
    raise TypeError(
        f"real_loss found no loss in {type(out).__name__}. Pass"
        " step_kwargs={'compute_loss': ...}, or use grad_probe for pure"
        " perf."
    )

  if not torch.is_tensor(loss):
    raise TypeError(f"real_loss: loss is {type(loss).__name__}, not a Tensor")

  model_name = type(model).__name__
  logging.log_first_n(
      logging.INFO,
      "real_loss: out is extracted from %s for model %s",
      1,
      loss_method,
      model_name,
  )
  if loss.ndim == 0:
    return loss
  else:
    logging.log_first_n(
        logging.INFO,
        "real_loss: Taking mean of loss with shape %s for %s",
        1,
        loss.shape,
        model_name,
    )
    return loss.mean()
