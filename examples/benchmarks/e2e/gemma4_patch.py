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

"""Patches and custom forward pass for Gemma4 models.

This module provides a custom `gemma4_custom_forward` function and helper
functions like `create_causal_mask_mapping` to adapt Gemma4 models
for specific use cases, potentially including multimodal inputs and
efficient attention mechanisms.
"""

from collections.abc import Callable, Mapping
import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig
from transformers.masking_utils import create_causal_mask
from transformers.masking_utils import create_masks_for_generate
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import Gemma4ModelOutputWithPast


def token_type_ids_mask_function(
    token_type_ids: torch.Tensor | None,
    image_group_ids: torch.Tensor | None,
) -> (
    Callable[
        [int, int, int | torch.Tensor, int | torch.Tensor], bool | torch.Tensor
    ]
    | None
):
  """Creates a mask function based on token types and image group IDs.

  This function generates a callable that determines whether attention should be
  allowed between a query token at `q_idx` and a key/value token at `kv_idx`.
  The masking logic ensures that attention is only permitted between tokens
  belonging to the same image group, as defined by `image_group_ids`.

  Args:
    token_type_ids: Optional tensor indicating the type of each token. If None,
      the function returns None.
    image_group_ids: Tensor containing group IDs for image tokens. Must not be
      None if `token_type_ids` is provided. Tokens with the same group ID belong
      to the same image.

  Returns:
    A callable `inner_mask(batch_idx, head_idx, q_idx, kv_idx)` which returns
    True if attention is allowed, False otherwise, or None if token_type_ids
    is None. The `inner_mask` checks if the query and key/value tokens are
    within the same image group and are valid tokens.
  """
  if token_type_ids is None:
    return None

  def inner_mask(
      batch_idx: int,
      _head_idx: int,
      q_idx: int | torch.Tensor,
      kv_idx: int | torch.Tensor,
  ) -> bool | torch.Tensor:
    assert (
        image_group_ids is not None
    ), "image_group_ids must not be None when token_type_ids is provided"
    seq_length = image_group_ids.shape[-1]
    if isinstance(q_idx, int):
      q_idx_clamped = min(q_idx, seq_length - 1)
    else:
      q_idx_clamped = q_idx.clamp(max=seq_length - 1)

    if isinstance(kv_idx, int):
      kv_idx_clamped = min(kv_idx, seq_length - 1)
    else:
      kv_idx_clamped = kv_idx.clamp(max=seq_length - 1)
    q_group = image_group_ids[batch_idx, q_idx_clamped]
    kv_group = image_group_ids[batch_idx, kv_idx_clamped]
    q_group = torch.where(q_idx < seq_length, q_group, -1)
    kv_group = torch.where(kv_idx < seq_length, kv_group, -1)
    return (q_group == kv_group) & (q_group >= 0)

  return inner_mask


def create_causal_mask_mapping(
    *,
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
    _is_first_iteration: bool | None = None,
    **_kwargs,
) -> Mapping:
  """Creates causal mask mapping for Gemma 4 models.

  Args:
    config: The model configuration.
    inputs_embeds: The input embeddings.
    attention_mask: The attention mask.
    past_key_values: The past key/values cache.
    position_ids: The position IDs.
    mm_token_type_ids: Optional multimodal token type IDs.
    is_first_iteration: Optional flag for first iteration.
    **kwargs: Additional keyword arguments.

  Returns:
    A mapping containing the generated masks.
  """
  mask_kwargs = {
      "config": config.get_text_config(),
      "inputs_embeds": inputs_embeds,
      "attention_mask": attention_mask,
      "past_key_values": past_key_values,
      "position_ids": position_ids,
  }
  if mm_token_type_ids is not None:
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    vision_group_ids = torch.where(is_vision, vision_group_ids, -1)
    or_mask_func = token_type_ids_mask_function(
        mm_token_type_ids.to(inputs_embeds.device), vision_group_ids
    )
    sliding_mask_kwargs = mask_kwargs | {"or_mask_function": or_mask_func}
  else:
    sliding_mask_kwargs = mask_kwargs

  return {
      "full_attention": create_causal_mask(**mask_kwargs),
      "sliding_attention": create_sliding_window_causal_mask(
          **sliding_mask_kwargs
      ),
  }


def gemma4_custom_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    input_features: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    mm_token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    image_position_ids: torch.LongTensor | None = None,
    video_position_ids: torch.LongTensor | None = None,
    **kwargs,
):
  """Custom forward pass for Gemma 4 model to avoid dynamic shapes.

  This function replaces the standard forward pass to avoid operations that
  cause dynamic shapes, such as boolean indexing. It uses masked operations
  instead.

  Args:
      self: The model instance.
      input_ids: The input token IDs.
      pixel_values: Optional image pixel values.
      pixel_values_videos: Optional video pixel values.
      input_features: Optional audio features.
      attention_mask: Optional attention mask.
      input_features_mask: Optional audio features mask.
      position_ids: Optional position IDs.
      past_key_values: Optional cache for generation.
      mm_token_type_ids: Optional multimodal token type IDs.
      inputs_embeds: Optional input embeddings.
      use_cache: Whether to use cache.
      image_position_ids: Optional image position IDs.
      video_position_ids: Optional video position IDs.
      **kwargs: Additional keyword arguments.

  Returns:
      Gemma4ModelOutputWithPast object.
  """
  image_features = None
  if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError(
        "You must specify exactly one of input_ids or inputs_embeds"
    )

  image_mask, video_mask, audio_mask = self.get_placeholder_mask(
      input_ids, inputs_embeds
  )
  multimodal_mask = image_mask | video_mask | audio_mask

  if inputs_embeds is None:
    assert (
        input_ids is not None
    ), "input_ids must not be None when inputs_embeds is None"
    # Use torch.where to avoid in-place operations, which can cause issues
    # with tracing or graph compilation.
    llm_input_ids = torch.where(
        multimodal_mask, self.config.text_config.pad_token_id, input_ids.clone()
    )
    initial_inputs_embeds = self.get_input_embeddings()(llm_input_ids)
  else:
    llm_input_ids = None
    initial_inputs_embeds = inputs_embeds

  if self.config.get_text_config().hidden_size_per_layer_input:
    pad_embedding = self.language_model.embed_tokens.weight[
        self.config.text_config.pad_token_id, :
    ]
    llm_inputs_embeds = torch.where(
        multimodal_mask[..., None],
        pad_embedding.view(1, 1, -1),
        initial_inputs_embeds,
    )
    per_layer_inputs = self.language_model.get_per_layer_inputs(
        llm_input_ids, llm_inputs_embeds
    )
  else:
    per_layer_inputs = None

  if pixel_values is not None:
    image_features = self.get_image_features(
        pixel_values, image_position_ids, return_dict=True
    ).pooler_output
    image_features = image_features.to(
        initial_inputs_embeds.device, initial_inputs_embeds.dtype
    )

    expanded_image_mask = (
        image_mask.unsqueeze(-1)
        .expand_as(initial_inputs_embeds)
        .to(initial_inputs_embeds.device)
    )

    assert (
        initial_inputs_embeds[expanded_image_mask].numel()
        == image_features.numel()
    ), (
        f"Image features ({image_features.numel()}) and image tokens"
        f" ({initial_inputs_embeds[expanded_image_mask].numel()}) do not match"
    )

    inputs_embeds_1 = initial_inputs_embeds.masked_scatter(
        expanded_image_mask.to(initial_inputs_embeds.device),
        image_features.to(initial_inputs_embeds.device),
    )
  else:
    inputs_embeds_1 = initial_inputs_embeds

  if pixel_values_videos is not None:
    video_features = self.get_video_features(
        pixel_values_videos, video_position_ids, return_dict=True
    ).pooler_output
    video_features = video_features.to(
        inputs_embeds_1.device, inputs_embeds_1.dtype
    )

    expanded_video_mask = (
        video_mask.unsqueeze(-1)
        .expand_as(inputs_embeds_1)
        .to(inputs_embeds_1.device)
    )
    assert (
        inputs_embeds_1[expanded_video_mask].numel() == video_features.numel()
    ), (
        f"Video features ({video_features.numel()}) and video tokens"
        f" ({inputs_embeds_1[expanded_video_mask].numel()}) do not match"
    )

    inputs_embeds_2 = inputs_embeds_1.masked_scatter(
        expanded_video_mask.to(inputs_embeds_1.device),
        video_features.to(inputs_embeds_1.device),
    )
  else:
    inputs_embeds_2 = inputs_embeds_1

  audio_features = None
  if input_features is not None and input_features_mask is not None:
    audio_output = self.get_audio_features(
        input_features, input_features_mask, return_dict=True
    )
    audio_features = audio_output.pooler_output
    audio_mask_from_encoder = audio_output.attention_mask

    audio_features = audio_features[audio_mask_from_encoder]

    expanded_audio_mask = (
        audio_mask.unsqueeze(-1)
        .expand_as(inputs_embeds_2)
        .to(inputs_embeds_2.device)
    )
    assert (
        inputs_embeds_2[expanded_audio_mask].numel() == audio_features.numel()
    ), (
        f"Audio features ({audio_features.numel()}) and audio tokens"
        f" ({inputs_embeds_2[expanded_audio_mask].numel()}) do not match"
    )

    inputs_embeds_final = inputs_embeds_2.masked_scatter(
        expanded_audio_mask.to(inputs_embeds_2.device),
        audio_features.to(inputs_embeds_2.device),
    )
  else:
    inputs_embeds_final = inputs_embeds_2

  if position_ids is None:
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    position_ids_final = (
        torch.arange(
            inputs_embeds_final.shape[1], device=inputs_embeds_final.device
        )
        + past_seen_tokens
    ).unsqueeze(0)
  else:
    position_ids_final = position_ids

  if (
      not isinstance(attention_mask, dict)
      and self.config.get_text_config().use_bidirectional_attention == "vision"
  ):
    causal_mask_mapping_final = create_causal_mask_mapping(
        config=self.config,
        inputs_embeds=inputs_embeds_final,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids_final,
        mm_token_type_ids=mm_token_type_ids,
    )
  elif not isinstance(attention_mask, dict):
    causal_mask_mapping_final = create_masks_for_generate(
        self.config,
        inputs_embeds_final,
        attention_mask,
        past_key_values,
        position_ids_final,
    )
  else:
    causal_mask_mapping_final = attention_mask

  kwargs["return_dict"] = True
  outputs = self.language_model(
      per_layer_inputs=per_layer_inputs,
      attention_mask=causal_mask_mapping_final,
      position_ids=position_ids_final,
      past_key_values=past_key_values,
      inputs_embeds=inputs_embeds_final,
      use_cache=use_cache,
      **kwargs,
  )

  return Gemma4ModelOutputWithPast(
      last_hidden_state=outputs.last_hidden_state,
      past_key_values=outputs.past_key_values,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
      image_hidden_states=image_features if pixel_values is not None else None,
      audio_hidden_states=audio_features
      if input_features is not None
      else None,
      shared_kv_states=outputs.shared_kv_states,
  )
