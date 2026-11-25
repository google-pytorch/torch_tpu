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

"""Model tests with dynamic shapes through torch.compile()."""

import copy
import enum
import re

from absl.testing import absltest
import torch
from torch._dynamo import testing as dynamo_testing
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils
import transformers


class _RecompileCheck(enum.Enum):
  NONE = "none"
  ZERO_STEADY_STATE = "zero_steady_state"


class ModelTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    torch.compiler.reset()
    self.device = torch.accelerator.current_accelerator()

  def _assert_decode_matches_eager(
      self,
      model: torch.nn.Module,
      prompt: torch.Tensor,
      recompile_check: _RecompileCheck = _RecompileCheck.NONE,
  ) -> None:
    def decode(fn, input_ids, counter=None):
      tokens = []
      logits = []
      step_compiles = []
      with torch.inference_mode():
        out = fn(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits.append(out.logits.cpu())
        for _ in range(4):
          tok = torch.argmax(out.logits[:, -1:, :], dim=-1)
          tokens.append(tok.cpu())
          before = counter.frame_count if counter is not None else 0
          out = fn(
              input_ids=tok,
              past_key_values=past,
              use_cache=True,
              logits_to_keep=1,
          )
          if counter is not None:
            step_compiles.append(counter.frame_count - before)
          past = out.past_key_values
          logits.append(out.logits.cpu())
      return tokens, logits, past.get_seq_length(), step_compiles

    eager_tokens, eager_logits, eager_seq_len, _ = decode(model, prompt)

    torch._dynamo.reset()
    counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled = torch.compile(
        model.to(self.device), backend=counter, dynamic=None
    )
    compiled_tokens, compiled_logits, compiled_seq_len, step_compiles = decode(
        compiled, prompt.to(self.device), counter=counter
    )

    self.assertEqual(compiled_seq_len, 20)
    self.assertEqual(compiled_seq_len, eager_seq_len)
    # bfloat16 logits differ by up to ~0.025 between eager and compiled on TPU
    # v6e.
    utils.assert_close(compiled_logits, eager_logits, rtol=3e-2, atol=3e-2)
    utils.assert_close(compiled_tokens, eager_tokens)
    if recompile_check == _RecompileCheck.ZERO_STEADY_STATE:
      # Decode step 0 compiles static, step 1 compiles dynamic, and steady-state
      # steps 2..3 must reuse the dynamic executable without recompiling.
      self.assertEqual(step_compiles, [1, 1, 0, 0])

  def test_gemma4_decode_dynamic(self):
    cfg = transformers.Gemma4TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        global_head_dim=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        sliding_window=64,
        use_cache=True,
        dtype="bfloat16",
        layer_types=["sliding_attention", "full_attention"],
    )
    model = transformers.Gemma4ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    prompt = torch.randint(0, 64, (1, 16))
    self._assert_decode_matches_eager(model, prompt)

  @absltest.skip("TODO(b/582377587): Enable with strided slice lowering.")
  def test_gptj_interleaved_rope_forward_and_backward_dynamic(self):
    cfg = transformers.GPTJConfig(
        vocab_size=64,
        n_embd=64,
        n_layer=1,
        n_head=4,
        rotary_dim=8,
        n_positions=64,
        use_cache=False,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model_cpu = transformers.GPTJForCausalLM(cfg).train()
    model_tpu = copy.deepcopy(model_cpu).to(self.device).train()

    torch._dynamo.reset()
    counter = dynamo_testing.CompileCounterWithBackend("tpu")
    compiled_fn = torch.compile(
        lambda ids: model_tpu(input_ids=ids, use_cache=False).logits,
        backend=counter,
        dynamic=None,
    )

    tpu_params = dict(model_tpu.named_parameters())
    for seq_len in (10, 11, 12, 15):
      input_ids = torch.randint(0, 64, (1, seq_len), dtype=torch.long)

      model_cpu.zero_grad(set_to_none=True)
      cpu_logits = model_cpu(input_ids=input_ids, use_cache=False).logits
      cpu_logits.mean().backward()

      model_tpu.zero_grad(set_to_none=True)
      tpu_ids = input_ids.to(self.device)
      torch._dynamo.mark_dynamic(tpu_ids, 1, min=6, max=32)
      tpu_logits = compiled_fn(tpu_ids)
      tpu_logits.mean().backward()

      utils.assert_close(
          tpu_logits.detach().cpu(),
          cpu_logits.detach(),
          rtol=3e-2,
          atol=3e-2,
          preamble=f"logits mismatch at seq_len={seq_len}",
      )
      for name, p_cpu in model_cpu.named_parameters():
        if p_cpu.grad is not None:
          p_tpu = tpu_params[name]
          self.assertIsNotNone(
              p_tpu.grad, f"missing grad for {name} at seq_len={seq_len}"
          )
          utils.assert_close(
              p_tpu.grad.cpu(),
              p_cpu.grad,
              rtol=5e-2,
              atol=5e-2,
              preamble=f"grad mismatch for {name} at seq_len={seq_len}",
          )
    self.assertEqual(counter.frame_count, 1)

  @absltest.skip("TODO(b/582377587): Enable with general view decomposition.")
  def test_gemma3_sliding_window_decode_dynamic(self):
    cfg = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sliding_window=4,
        use_cache=True,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = transformers.Gemma3ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    prompt = torch.randint(0, 64, (1, 16))
    self._assert_decode_matches_eager(
        model,
        prompt,
        recompile_check=_RecompileCheck.ZERO_STEADY_STATE,
    )

  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 54),
      "Requires libtpu >= 0.0.54 for 64-bit and 128-bit PadRealToStatic.",
  )
  def test_gemma4_dynamic_int64_prompt_and_mask_pad_real_to_static(self):
    cfg = transformers.Gemma4TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        global_head_dim=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        sliding_window=64,
        use_cache=True,
        dtype="bfloat16",
        layer_types=["sliding_attention", "full_attention"],
    )
    model = (
        transformers.Gemma4ForCausalLM(cfg)
        .to(device=self.device, dtype=torch.bfloat16)
        .eval()
    )
    cases = []
    for seq_len in (12, 24, 19):
      prompt = torch.randint(
          0, 64, (1, seq_len), dtype=torch.int64, device=self.device
      )
      base_mask = torch.ones((1, seq_len), dtype=torch.int64)
      base_mask[:, :2] = 0
      masks = [
          torch.cat(
              [base_mask, torch.ones((1, step), dtype=torch.int64)], dim=-1
          ).to(self.device)
          for step in range(4)
      ]
      cases.append((prompt, masks))

    def prefill_and_decode(fn, input_ids, step_masks):
      tokens = []
      logits = []
      with torch.inference_mode():
        out = fn(
            input_ids=input_ids,
            attention_mask=step_masks[0],
            use_cache=True,
            logits_to_keep=1,
        )
        past = out.past_key_values
        logits.append(out.logits.cpu())
        for step_mask in step_masks[1:]:
          tok = torch.argmax(out.logits[:, -1:, :], dim=-1)
          tokens.append(tok.cpu())
          out = fn(
              input_ids=tok,
              attention_mask=step_mask,
              past_key_values=past,
              use_cache=True,
              logits_to_keep=1,
          )
          past = out.past_key_values
          logits.append(out.logits.cpu())
      return tokens, logits, past.get_seq_length()

    eager_refs = [
        prefill_and_decode(model, prompt, masks) for prompt, masks in cases
    ]

    debugs = []
    compiled = torch.compile(
        model,
        backend="tpu",
        dynamic=None,
        options={"debug_callback": debugs.append},
    )

    compiled_outs = []
    unseen_compilations = None
    for idx, (prompt, masks) in enumerate(cases):
      before = torch.tpu._get_cache_stats().num_compilations
      res = prefill_and_decode(compiled, prompt, masks)
      after = torch.tpu._get_cache_stats().num_compilations
      if idx == 2:
        unseen_compilations = after - before
      compiled_outs.append(res)

    self.assertEqual(unseen_compilations, 0)
    for (
        (prompt, _),
        (got_toks, got_logits, got_len),
        (
            ref_toks,
            ref_logits,
            ref_len,
        ),
    ) in zip(cases, compiled_outs, eager_refs, strict=True):
      self.assertEqual(got_len, prompt.shape[1] + 3)
      self.assertEqual(got_len, ref_len)
      utils.assert_close(got_logits, ref_logits, rtol=3e-2, atol=3e-2)
      utils.assert_close(got_toks, ref_toks)

    mlir_texts = [t for d in debugs for t in d.stablehlo_forward_text]
    self.assertTrue(
        any(
            len(re.findall(r"@PadRealToStatic\b.*-> tensor<1x\d+xi64>", text))
            >= 2
            for text in mlir_texts
        )
    )
    self.assertTrue(
        any(
            re.search(r"@PadRealToStatic\b.*-> tensor<1x\d+xi64>", text)
            and re.search(r"@PadRealToStatic\b.*-> tensor<1x[^>]*xbf16>", text)
            for text in mlir_texts
        )
    )


if __name__ == "__main__":
  absltest.main()
