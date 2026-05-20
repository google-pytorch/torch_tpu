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

"""Provides a unified registry for instantiating PyTorch models from various sources.

This module defines a generic interface for model providers (`BaseProvider`) and
implementations for popular model libraries:
    - Torchvision
    - TIMM (PyTorch Image Models)
    - Hugging Face Transformers

It exposes a `ModuleRegistry` that aggregates these providers, allowing users
to list available models and retrieve `ModuleSpec` objects containing the
instantiated model and factory functions for generating compatible sample
inputs.
"""

# pylint: disable=g-import-not-at-top

import abc
from collections.abc import Callable, Sequence
from importlib import resources
from importlib.resources import abc as resources_abc
import inspect
import pathlib
from typing import Any, Iterator

from absl import flags
from absl import logging
from etils import epath
import torch

try:
  import diffusers
  from diffusers.models.auto_model import AutoModel

  _HAS_DIFFUSERS = True
except ImportError:
  _HAS_DIFFUSERS = False

try:
  import timm

  _HAS_TIMM = True
except ImportError:
  _HAS_TIMM = False

try:
  import torchvision

  _HAS_TORCHVISION = True
except ImportError:
  _HAS_TORCHVISION = False

try:
  import transformers

  _HAS_TRANSFORMERS = True
except ImportError:
  _HAS_TRANSFORMERS = False

_MAX_SEQ_LEN_HEURISTIC_CAP = 100_000

_WEIGHTS_BASE_PATH = flags.DEFINE_string(
    "weights_base_path",
# "$DATA_ROOT/$WEIGHTS_SUBDIR"
    "",
    "Default base location of model configs and weights.",
)


class ModuleSpec:
  """A specification container for a PyTorch model and its inputs.

  Attributes:
    module_factory: A callable that returns an instantiated `torch.nn.Module`.
    sample_inputs_factory: A callable that generates compatible input tensors.
      It accepts optional `shape` (Sequence) and `device` (str) arguments and
      returns a tuple containing `(args, kwargs)` for the model's forward pass.
    preprocessor_factory: A callable that generates a preprocessor for models
      that need it. None by default.
    config: Optional configuration object associated with the model (e.g., a
      Transformers `AutoConfig`).
  """

  def __init__(
      self,
      module_factory: Callable[[], torch.nn.Module],
      sample_inputs_factory: Callable[
          [Sequence[int] | None, str | None],
          tuple[tuple[Any, ...], dict[str, Any]],
      ],
      preprocessor_factory: Callable[[], Any] | None = None,
      config: Any | None = None,
  ):
    self.module_factory = module_factory
    self.sample_inputs_factory = sample_inputs_factory
    self.preprocessor_factory = preprocessor_factory
    self.config = config


class BaseProvider(abc.ABC):
  """Abstract base class for model source providers."""

  def __init__(self, base_path: str | None = None, subdir: str | None = None):
    if base_path is None:
      base_path = _WEIGHTS_BASE_PATH.value

    # If base_path is empty, assume no local cache is available
    self.has_cache_dir = bool(base_path)

    if self.has_cache_dir:
      self._base_path = epath.Path(base_path)
      if subdir:
        self._base_path = self._base_path / subdir
      if not self._base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {self._base_path}")
    else:
      self._base_path = None

  @abc.abstractmethod
  def list_modules(self) -> list[str]:
    """Lists the names of all models available from this provider.

    Returns:
      A list of model name strings.
    """
    pass

  @abc.abstractmethod
  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Retrieves the specification for a specific model.

    Args:
      name: The name of the model to retrieve.
      load_weights: If True, loads pre-trained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable to modify the model configuration. The
        config object is specific to the provider.
      **kwargs: Additional provider-specific arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    pass


class TorchvisionProvider(BaseProvider):
  """Provider for standard Torchvision models."""

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="torchvision")

  def list_modules(self) -> list[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/torchvision/{model_name}/weights.pt

    Returns:
      A list of model name strings.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    return [path.name for path in self._base_path.iterdir() if path.is_dir()]

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    if modify_config_hook is not None:
      raise ValueError("modify_config_hook is not supported for torchvision.")
    if load_weights:
      raise NotImplementedError(
          "Loading pretrained weights not yet implemented."
      )
    default_shape = (1, 3, 224, 224)
    return ModuleSpec(
        lambda: torchvision.models.get_model(name),
        lambda shape=None, device="cpu": (
            (
                torch.randn(
                    shape if shape is not None else default_shape, device=device
                ),
            ),
            {},
        ),
    )


class TimmProvider(BaseProvider):
  """Provider for TIMM (PyTorch Image Models)."""

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="timm")

  def list_modules(self) -> list[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/timm/{model_name}/weights.pth

    Returns:
      A list of model name strings.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    return [path.name for path in self._base_path.iterdir() if path.is_dir()]

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a TIMM model.

    Attempts to determine the default input shape from the model's pretrained
    configuration. Defaults to (3, 224, 224) if configuration is unavailable.

    Args:
      name: Name (str) of the timm model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable that accepts and returns a
        timm.models.PretrainedCfg object to modify the model configuration.
      **kwargs: Additional keyword arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    if load_weights and not self.has_cache_dir:
      raise ValueError(
          f"load_weights cannot be set to True for {name} when no cache"
          " directory is available."
      )

    try:
      config = timm.models.get_pretrained_cfg(name)
    except RuntimeError:
      config = None
      logging.warning(
          "Couldn't find config for %s.",
          name,
      )

    if config and modify_config_hook is not None:
      config = modify_config_hook(config)

    def _module_factory():
      if load_weights and self._base_path:
        local_checkpoint = self._base_path / name / f"{name}.pth"

        if local_checkpoint.exists():
          model = timm.create_model(name, pretrained=False)
          #  local checkpoint can't be loaded by timm.create_model directly.
          with local_checkpoint.open("rb") as f:
            state_dict = torch.load(f)
          model.load_state_dict(state_dict)
          return model
        else:
          raise ValueError(
              f"Cannot load weights for {name} because checkpoint is missing"
              f" at {local_checkpoint}."
          )

      return timm.create_model(name, pretrained=False)

    def _input_factory(shape=None, device="cpu"):
      input_size = config.input_size if config else (3, 224, 224)
      final_shape = shape if shape else (1, *input_size)
      return ((torch.randn(final_shape, device=device),), {})

    def _preprocessor_factory():
      if config:
        data_config = timm.data.resolve_data_config(
            {}, pretrained_cfg=config.to_dict()
        )
        transform = timm.data.create_transform(**data_config)
        return lambda img: transform(img).unsqueeze(0)
      return None

    return ModuleSpec(
        _module_factory, _input_factory, _preprocessor_factory, config
    )


def _get_max_seq_len(
    config: transformers.AutoConfig, default: int = 512
) -> int:
  """Heuristically determines the maximum sequence length from a config.

  Checks multiple attributes (e.g., `max_position_embeddings`, `n_positions`)
  and applies a cap to avoid unreasonably large allocations (e.g., for models
  using relative embeddings or integer-limit defaults).

  Args:
    config: The Transformers configuration object.
    default: The fallback sequence length if no attribute is found.

  Returns:
    The determined maximum sequence length.
  """
  for attr in [
      "max_position_embeddings",
      "n_positions",
      "seq_length",
      "max_seq_len",
  ]:
    if hasattr(config, attr):
      value = getattr(config, attr)
      # Some configs have this set to huge numbers (e.g. integer limit)
      # or None. We cap it to an arbitrary value.
      if value is not None and value < _MAX_SEQ_LEN_HEURISTIC_CAP:
        return value

    # Check tokenizer-specific max length if available in config
    # (Sometimes 'model_max_length' is injected into config).
    if hasattr(config, "model_max_length"):
      value = config.model_max_length
      if value is not None and value < _MAX_SEQ_LEN_HEURISTIC_CAP:
        return value

  # Fallback for relative position models (T5, etc.) or missing data.
  return default


def _walk_package_resources(
    traversable: resources_abc.Traversable,
) -> Iterator[resources_abc.Traversable]:
  """Recursively yields Traversable objects for all files in a directory/package."""
  for path in traversable.iterdir():
    if path.is_file():
      yield path
    elif path.is_dir():
      # Recursively call the function for subdirectories
      yield from _walk_package_resources(path)


class TransformersProvider(BaseProvider):
  """Provider for Hugging Face Transformers models.

  Note:
    This provider relies on local resource files for model configurations
    rather than downloading directly from the Hugging Face Hub.
  """

  _FILES = resources.files("examples").joinpath(
      "huggingface_transformers/model_configs"
  )

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="transformers")

  def list_modules(self) -> list[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/transformers/{owner}/{model}/files

    Returns:
      A list of model name strings formatted as '{owner}/{model}'.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    modules = []
    for owner_path in self._base_path.iterdir():
      if owner_path.is_dir():
        for model_path in owner_path.iterdir():
          if model_path.is_dir():
            modules.append(f"{owner_path.name}/{model_path.name}")
    return modules

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a Transformer model.

    Automatically detects the architecture from the config and prepares
    integer-based input tensors (token IDs).

    Args:
      name: Name (str) of the hf transformer model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable that accepts and returns a
        transformers.PretrainedConfig object to modify the model configuration.
      **kwargs: Additional keyword arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    # Load the config first
    config = None

    if self.has_cache_dir:
      model_dir_or_repo_id = self._base_path / name
      try:
        if model_dir_or_repo_id.exists():
          config = transformers.AutoConfig.from_pretrained(
              str(model_dir_or_repo_id)
          )
      except Exception as exc:  # pylint: disable=broad-except
        logging.warning(
            "Failed to access %s in cache, falling back to local resources."
            " Error: %s",
            model_dir_or_repo_id,
            exc,
        )

    if config is None:  # Fallback to local resources
      # Pretrained weights are not available in local resources.
      if load_weights:
        raise ValueError(
            f"load_weights cannot be set to True for {name} when falling back"
            " to local configuration resources."
        )
      try:
        with resources.as_file(
            self._FILES.joinpath(str(pathlib.Path(name) / "config.json"))
        ) as f:
          config = transformers.AutoConfig.from_pretrained(str(f))
      except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(
            f"Model config for '{name}' is missing in local resources"
            f" ({self._FILES.joinpath(str(pathlib.Path(name) / 'config.json'))})"
            " and could not be loaded from cache."
        ) from exc

    if modify_config_hook is not None:
      if load_weights:
        raise NotImplementedError(
            "modify_config_hook is not supported when load_weights is True."
        )
      config = modify_config_hook(config)

    if load_weights:
      model_fn = lambda: transformers.AutoModelForCausalLM.from_pretrained(
          str(model_dir_or_repo_id), **kwargs
      )
      preprocessor_fn = lambda: (
          transformers.AutoTokenizer.from_pretrained(str(model_dir_or_repo_id))
      )
    else:
      architectures = getattr(config, "architectures", [])
      model_cls = transformers.AutoModel

      if architectures:
        try:
          model_cls = getattr(transformers, architectures[0])
        except AttributeError:
          logging.warning(
              "Could not find architecture %s, falling back to AutoModel.",
              architectures[0],
          )

      model_fn = lambda: model_cls(config)
      preprocessor_fn = None

    safe_seq_len = min(_get_max_seq_len(config), 512)

    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None and hasattr(config, "text_config"):
      vocab_size = getattr(config.text_config, "vocab_size", None)
    if vocab_size is None:
      vocab_size = 32000

    def _input_fn(
        shape=None, device="cpu"
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      actual_shape = shape if shape is not None else (1, safe_seq_len)
      return (
          (),
          {
              "input_ids": torch.randint(
                  0, vocab_size, actual_shape, device=device
              ),
              "attention_mask": torch.ones(
                  actual_shape, device=device, dtype=torch.long
              ),
          },
      )

    return ModuleSpec(model_fn, _input_fn, preprocessor_fn, config)


class DiffusersProvider(BaseProvider):
  """Provider for Hugging Face Diffusers models.

  Note:
    This provider relies on local resource files for model configurations
    rather than downloading directly from the Hugging Face Hub.
  """

  _FILES = resources.files("examples").joinpath(
      "huggingface_diffusers/model_configs"
  )

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="diffusers")

  def list_modules(self) -> list[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/diffusers/{owner}/{model}/files

    Returns:
      A list of model name strings formatted as '{owner}/{model}'.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    modules = []
    for owner_path in self._base_path.iterdir():
      if owner_path.is_dir():
        for model_path in owner_path.iterdir():
          if model_path.is_dir():
            modules.append(f"{owner_path.name}/{model_path.name}")
    return modules

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a Diffuser model.

    Args:
      name: Name (str) of the hf diffuser model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable that accepts and returns a config
        dictionary to modify the model configuration for diffusers models.
      **kwargs: Additional keyword arguments. Supported arguments: - subfolder:
        Subfolder of the model to load.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    subfolder = kwargs.get("subfolder")
    d_type = torch.bfloat16

    raw_config = None

    if self.has_cache_dir:
      model_path = self._base_path / name
      try:
        if model_path.exists():
          raw_config = AutoModel.load_config(
              str(model_path), subfolder=subfolder
          )
      except Exception as exc:  # pylint: disable=broad-except
        logging.warning(
            "Failed to access %s in cache, falling back to local resources."
            " Error: %s",
            model_path,
            exc,
        )

    if raw_config is None:  # Fallback to local resources
      if load_weights:
        # Pretrained weights are not available in local resources.
        raise ValueError(
            f"load_weights cannot be set to True for {name} when falling back"
            " to local configuration resources."
        )
      model_dir = pathlib.Path(name)
      if subfolder:
        model_dir = model_dir / subfolder
      try:
        with resources.as_file(
            self._FILES.joinpath(str(model_dir / "config.json"))
        ) as f:
          raw_config = AutoModel.load_config(str(f.parent))
      except Exception as exc:  # pylint: disable=broad-exception-caught
        raise ValueError(
            f"Model config for '{name}' is missing in local resources "
            f"({self._FILES.joinpath(str(model_dir / 'config.json'))}) "
            "and could not be loaded from cache."
        ) from exc

    if modify_config_hook is not None:
      if load_weights:
        raise NotImplementedError(
            "modify_config_hook is not supported when load_weights is True."
        )
      raw_config = modify_config_hook(raw_config)
    config_dict = dict(raw_config)

    def _module_factory():
      if load_weights:
        return AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=d_type,
            subfolder=subfolder,
        )
      else:
        class_name = config_dict.get("_class_name")
        if not class_name:
          raise ValueError(f"Config for {name} is missing '_class_name'.")
        model_cls = getattr(diffusers, class_name)
        return model_cls.from_config(config_dict).to(d_type)

    def _input_factory(shape=None, device="cpu"):
      batch_size = shape[0] if shape else 1
      seq_len = shape[1] if shape and len(shape) > 1 else 77
      cfg = config_dict

      latent_channels = cfg.get("in_channels", 4)

      # Use latent dimensions passed in shape if available
      # (e.g. for video models)
      if shape and len(shape) > 2:
        latent_dims = shape[2:]
      else:
        # Fall back to sample_size from config for 2D models
        latent_size = cfg.get("sample_size", 64)
        if isinstance(latent_size, int):
          latent_dims = (latent_size, latent_size)
        else:
          latent_dims = tuple(latent_size)

        # Check for 3D model if we only have 2D defaults.
        # If patch_size has 3 dimensions, it's likely a 3D/video model and
        # expects a "frames" dimension.
        patch_size = cfg.get("patch_size")
        if (
            len(latent_dims) == 2
            and isinstance(patch_size, (list, tuple))
            and len(patch_size) == 3
        ):
          latent_dims = (2,) + latent_dims  # Default 2 frames

      noisy_latents = torch.randn(
          (batch_size, latent_channels, *latent_dims),
          dtype=d_type,
          device=device,
      )

      # standard num_train_timesteps for diffusion models is typically 1000
      timesteps = torch.randint(
          0, 1000, (batch_size,), device=device, dtype=torch.long
      )

      # Fallback to text_dim if cross_attention_dim is not specified in config
      # (e.g. for Wan2.2)
      cross_attention_dim = cfg.get("cross_attention_dim")
      if cross_attention_dim is None:
        cross_attention_dim = cfg.get("text_dim", 2048)

      # Encoder hidden states - These would be per token text embeddings
      # returned by CLIP-ViT/L text encoder
      dummy_encoder_hidden_states = torch.randn(
          (batch_size, seq_len, cross_attention_dim),
          dtype=d_type,
          device=device,
      )

      # Dynamically get the model class to inspect its signature
      class_name = cfg.get("_class_name")
      if not class_name:
        raise ValueError(
            f"Config for {name} is missing '_class_name'. Cannot generate"
            " inputs."
        )
      model_cls = getattr(diffusers, class_name)
      forward_params = inspect.signature(model_cls.forward).parameters

      # Dynamically determine the primary input key
      if "sample" in forward_params:
        primary_input_key = "sample"
      elif "hidden_states" in forward_params:
        primary_input_key = "hidden_states"
      else:
        # Fallback to the first non-'self' argument of the forward method.
        non_self_params = [p for p in forward_params.keys() if p != "self"]
        if non_self_params:
          primary_input_key = non_self_params[0]
        else:
          raise ValueError(
              f"Could not determine primary input key for {name}. No suitable"
              " parameter found in forward signature."
          )

      kwargs = {
          primary_input_key: noisy_latents,
          "timestep": timesteps,
          "encoder_hidden_states": dummy_encoder_hidden_states,
      }

      # Create dummy additional conditioning inputs which would be expected in
      # an SDXL pipeline.
      if cfg.get("addition_embed_type") == "text_time":
        proj_dim = cfg.get("projection_class_embeddings_input_dim", 2816)
        time_dim = cfg.get("addition_time_embed_dim", 256)
        text_embeds_dim = proj_dim - (6 * time_dim)

        # Pooled text embeddings - A single embedding vector per batch
        # representing the entire text sequence returned by OpenCLIP-ViT/G
        dummy_pooled_embeds = torch.randn(
            (batch_size, text_embeds_dim), dtype=d_type, device=device
        )
        # time_ids - Tensor of shape [batch_size, 6] providing crop information
        # to the unet (nothing to do with timesteps).
        # Represents: [orig_height, orig_width, crops_coords_top,
        # crops_coords_left, target_height, target_width]
        # Here, we initialize random values.
        dummy_time_ids = torch.randn(
            (batch_size, 6), dtype=d_type, device=device
        )

        kwargs["added_cond_kwargs"] = {
            "text_embeds": dummy_pooled_embeds,
            "time_ids": dummy_time_ids,
        }

      if "encoder_hidden_states" not in forward_params:
        kwargs.pop("encoder_hidden_states", None)
      if "added_cond_kwargs" not in forward_params:
        kwargs.pop("added_cond_kwargs", None)
      if "timestep" not in forward_params:
        kwargs.pop("timestep", None)

      return (), kwargs

    return ModuleSpec(_module_factory, _input_factory, config=config_dict)


class ModuleRegistry:
  """Central registry for managing multiple model providers."""

  def __init__(self, base_path: str | None = None):
    """Initializes the registry.

    Providers construct specific paths relative to this base_path.
    - For Timm and Torchvision library, models are structured as:
    base_path/{provider}/{model}/
    - For other libraries, models are structured as:
    base_path/{provider}/{owner}/{model}/

    Args:
      base_path: The base directory for model weights. Defaults to the value of
        `_WEIGHTS_BASE_PATH`.
    """
    if base_path is None:
      base_path = _WEIGHTS_BASE_PATH.value

    self._providers: dict[str, BaseProvider] = {}
    if _HAS_TORCHVISION:
      self._providers["torchvision"] = TorchvisionProvider(base_path=base_path)
    if _HAS_TIMM:
      self._providers["timm"] = TimmProvider(base_path=base_path)
    if _HAS_TRANSFORMERS:
      self._providers["transformers"] = TransformersProvider(
          base_path=base_path
      )
    if _HAS_DIFFUSERS:
      self._providers["diffusers"] = DiffusersProvider(base_path=base_path)

  def list_all_sources(self) -> list[str]:
    """Returns a list of registered provider keys (e.g., 'torchvision')."""
    return list(self._providers.keys())

  def list_all_modules(self) -> list[str]:
    """Returns a unified list of all available modules across all providers.

    Returns:
      A list of strings formatted as '{source}/{model_name}'.
    """
    modules = []
    for key, p in self._providers.items():
      for m in p.list_modules():
        modules.append(f"{key}/{m}")
    return modules

  def list_modules(self, source: str) -> list[str]:
    """Lists available models for a specific source.

    Args:
      source: The provider key (e.g., 'timm', 'transformers').

    Returns:
      A list of model names.

    Raises:
      ValueError: If the source is not found in the registry.
    """
    if source not in self._providers:
      raise ValueError(f"Source '{source}' not supported.")
    return self._providers[source].list_modules()

  def get_module_spec(
      self,
      source: str,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Instantiates and returns the ModuleSpec for a specific model.

    Args:
      source: The provider key.
      name: The name of the model within that provider.
      load_weights: Whether to load pre-trained weights.
      modify_config_hook: A callable to modify the model configuration. The
        callable accepts and returns a config object specific to the model
        library (e.g., transformers.PretrainedConfig for transformers).
      **kwargs: Additional provider-specific arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.

    Raises:
      ValueError: If the source is not found in the registry.
    """
    if source not in self._providers:
      raise ValueError(f"Source '{source}' not supported.")

    provider = self._providers[source]
    return provider.get_module_spec(
        name,
        load_weights=load_weights,
        modify_config_hook=modify_config_hook,
        **kwargs,
    )
