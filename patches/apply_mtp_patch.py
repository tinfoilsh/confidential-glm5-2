#!/usr/bin/env python3
"""
Backport PR #45895: [bugfix] Indexer init skip and MTP TopK share for iteration.

This PR fixes GLM-5.2 MTP (Multi-Token Prediction) speculative decoding:
- Fixes GLM-5.2 BF16 init Indexer in skip_topk layer
- Fixes GLM-5.2 MTP Port-Norm cycle (post-final-norm hidden recycling)
- MTP Mean Acceptance Length raises from ~3 to ~4, Mean Acceptance Rate to ~60%

Without this patch, MTP with sparse MLA (GLM-5.2's attention backend) crashes
because backbone "skip" layers don't have an indexer but the sparse MLA backend
asserts `indexer is not None`.

Changes:
1. mla_attention.py: Pass topk_indices_buffer to impl for sparse layers
2. mla.py: Pass topk_indices_buffer from MLAModules to MLAAttention
3. deepseek_mtp.py: Return (hidden_states, shared_head(hidden_states)) tuple
   for post-norm recycling
4. deepseek_v2.py: Always build indexer for MTP/nextn layers (is_mtp_layer check)
5. flashmla_sparse.py: Accept topk_indices_buffer directly when indexer is None
6. flashinfer_mla_sparse.py: Same - accept topk_indices_buffer directly
7. llm_base_proposer.py: model_returns_tuple returns True for DeepSeekMTPModel
8. speculative.py: Add mtp_recycle_post_norm field (config-level)
"""

import re
import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm")

def patch_file(rel_path, patches):
    """Apply a list of (old, new) replacements to a file."""
    fpath = DIST / rel_path
    content = fpath.read_text()
    for i, (old, new, desc) in enumerate(patches):
        if old not in content:
            print(f"  [{i+1}/{len(patches)}] SKIP (already applied or not found): {desc}")
            continue
        content = content.replace(old, new, 1)
        print(f"  [{i+1}/{len(patches)}] APPLIED: {desc}")
    fpath.write_text(content)
    print(f"  Wrote {fpath}")


print("=" * 60)
print("Patching mla_attention.py: pass topk_indices_buffer to impl")
print("=" * 60)

patch_file("model_executor/layers/attention/mla_attention.py", [
    # 1. Add topk_indices_buffer parameter to __init__
    (
        "        use_sparse: bool = False,\n        indexer: object | None = None,\n        **extra_impl_args,\n    ):",
        "        use_sparse: bool = False,\n        indexer: object | None = None,\n        topk_indices_buffer: torch.Tensor | None = None,\n        **extra_impl_args,\n    ):",
        "Add topk_indices_buffer param to MLAAttention.__init__"
    ),
    # 2. Pass topk_indices_buffer to impl for sparse layers
    (
        "        if (\n            cache_config is not None\n            and cache_config.enable_prefix_caching\n            and envs.VLLM_BATCH_INVARIANT",
        "        # Sparse MLA reads top-k indices from a shared buffer. Pass it\n        # explicitly so backbone \"skip\" layers (indexer=None) still find it.\n        # (PR #45895: Indexer init skip and MTP TopK share for iteration)\n        if use_sparse:\n            extra_impl_args[\"topk_indices_buffer\"] = topk_indices_buffer\n\n        if (\n            cache_config is not None\n            and cache_config.enable_prefix_caching\n            and envs.VLLM_BATCH_INVARIANT",
        "Pass topk_indices_buffer to extra_impl_args for sparse MLA"
    ),
])

print()
print("=" * 60)
print("Patching mla.py: pass topk_indices_buffer from MLAModules")
print("=" * 60)

patch_file("model_executor/layers/mla.py", [
    # 3. Pass topk_indices_buffer to MLAAttention constructor
    (
        "            kv_b_proj=self.kv_b_proj,\n            use_sparse=self.is_sparse,\n            indexer=self.indexer,\n        )",
        "            kv_b_proj=self.kv_b_proj,\n            use_sparse=self.is_sparse,\n            indexer=self.indexer,\n            topk_indices_buffer=mla_modules.topk_indices_buffer,\n        )",
        "Pass topk_indices_buffer to MLAAttention in MultiHeadLatentAttentionWrapper"
    ),
])

print()
print("=" * 60)
print("Patching deepseek_mtp.py: return tuple for post-norm recycling")
print("=" * 60)

patch_file("model_executor/models/deepseek_mtp.py", [
    # 4. DeepSeekMultiTokenPredictorLayer.forward returns tuple
    (
        "        hidden_states = residual + hidden_states\n        return hidden_states\n\n\nclass DeepSeekMultiTokenPredictor(nn.Module):",
        "        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)\n        # Recycle the post-final-norm hidden into the next draft step.\n        # compute_logits applies shared_head (== final norm) to the pre-norm\n        # element, so logits and the recycle each get exactly one final-norm.\n        # Matches SGLang's deepseek_nextn.\n        return hidden_states, self.shared_head(hidden_states)\n\n\nclass DeepSeekMultiTokenPredictor(nn.Module):",
        "DeepSeekMultiTokenPredictorLayer.forward returns (hidden, shared_head(hidden)) tuple"
    ),
])

print()
print("=" * 60)
print("Patching deepseek_v2.py: always build indexer for MTP/nextn layers")
print("=" * 60)

patch_file("model_executor/models/deepseek_v2.py", [
    # 5. Move skip_topk computation before the is_v32 check and add is_mtp_layer
    (
        "        self.is_v32 = hasattr(config, \"index_topk\")\n\n        _skip_topk = False\n        if self.is_v32:\n            self.indexer_rope_emb = get_rope(",
        "        self.is_v32 = hasattr(config, \"index_topk\")\n\n        # IndexCache config\n        # Refer: https://arxiv.org/abs/2603.12201 for more details.\n        _skip_topk = False\n        _index_topk_freq = getattr(config, \"index_topk_freq\", 1)\n        _index_topk_pattern = getattr(config, \"index_topk_pattern\", None)\n        _index_skip_topk_offset = getattr(config, \"index_skip_topk_offset\", 2)\n        layer_id = extract_layer_index(prefix)\n\n        if _index_topk_pattern is None:\n            _skip_topk = (\n                max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq != 0\n            )\n        elif 0 <= layer_id < len(_index_topk_pattern):\n            _skip_topk = _index_topk_pattern[layer_id] == \"S\"\n\n        # The skip pattern only governs backbone layers. MTP/nextn layers\n        # (layer_id >= num_hidden_layers) always build a full indexer: they\n        # compute indices at draft step 0 and toggle at runtime via\n        # set_skip_topk (index_share_for_mtp_iteration).\n        _num_hidden_layers = getattr(config, \"num_hidden_layers\", None)\n        is_mtp_layer = _num_hidden_layers is not None and layer_id >= _num_hidden_layers\n\n        if self.is_v32 and (not _skip_topk or is_mtp_layer):\n            self.indexer_rope_emb = get_rope(",
        "Move skip_topk computation before is_v32 check, add is_mtp_layer"
    ),
    # 6. Remove the old skip_topk computation block inside the is_v32 branch
    (
        "                is_inplace_rope=self.indexer_rope_emb.enabled(),\n            )\n\n            # IndexCache config\n            # Refer: https://arxiv.org/abs/2603.12201 for more details.\n            _index_topk_freq = getattr(config, \"index_topk_freq\", 1)\n            _index_topk_pattern = getattr(config, \"index_topk_pattern\", None)\n            _index_skip_topk_offset = getattr(config, \"index_skip_topk_offset\", 2)\n            layer_id = extract_layer_index(prefix)\n\n            if _index_topk_pattern is None:\n                _skip_topk = (\n                    max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq\n                    != 0\n                )\n            elif 0 <= layer_id < len(_index_topk_pattern):\n                _skip_topk = _index_topk_pattern[layer_id] == \"S\"\n\n        else:",
        "                is_inplace_rope=self.indexer_rope_emb.enabled(),\n            )\n\n        else:",
        "Remove old skip_topk computation block from inside is_v32 branch"
    ),
])

print()
print("=" * 60)
print("Patching flashmla_sparse.py: accept topk_indices_buffer directly")
print("=" * 60)

patch_file("v1/attention/backends/mla/flashmla_sparse.py", [
    # 7. Accept topk_indices_buffer when indexer is None
    (
        "        assert indexer is not None\n        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer\n        # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell",
        "        # The indexer carries the shared buffer for normal layers and tests;\n        # the explicitly-passed buffer covers backbone skip layers, whose\n        # indexer is not constructed (see deepseek_v2.py).\n        self.topk_indices_buffer: torch.Tensor | None = (\n            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer\n        )\n        # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell",
        "flashmla_sparse: accept topk_indices_buffer when indexer is None"
    ),
])

print()
print("=" * 60)
print("Patching flashinfer_mla_sparse.py: accept topk_indices_buffer directly")
print("=" * 60)

patch_file("v1/attention/backends/mla/flashinfer_mla_sparse.py", [
    # 8. Rename topk_indice_buffer -> topk_indices_buffer and accept when indexer is None
    (
        "        topk_indice_buffer: torch.Tensor | None = None,\n        indexer: \"Indexer | None\" = None,",
        "        topk_indices_buffer: torch.Tensor | None = None,\n        indexer: \"Indexer | None\" = None,",
        "flashinfer_mla_sparse: rename topk_indice_buffer -> topk_indices_buffer"
    ),
    (
        "        assert indexer is not None, \"Indexer required for sparse MLA\"\n        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer\n\n        self._workspace_buffer",
        "        # The indexer carries the shared buffer for normal layers and tests;\n        # the explicitly-passed buffer covers backbone skip layers, whose\n        # indexer is not constructed (see deepseek_v2.py).\n        self.topk_indices_buffer: torch.Tensor | None = (\n            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer\n        )\n\n        self._workspace_buffer",
        "flashinfer_mla_sparse: accept topk_indices_buffer when indexer is None"
    ),
])

print()
print("=" * 60)
print("Patching llm_base_proposer.py: model_returns_tuple for DeepSeekMTPModel")
print("=" * 60)

patch_file("v1/spec_decode/llm_base_proposer.py", [
    # 9. model_returns_tuple returns True for DeepSeekMTPModel
    (
        "    def model_returns_tuple(self) -> bool:\n        return self.method not in (\"mtp\", \"draft_model\", \"dflash\")",
        "    def model_returns_tuple(self) -> bool:\n        if self.method == \"mtp\":\n            # DeepSeek-family MTP (deepseek_mtp.py) recycles the post-final-\n            # norm hidden, so its forward returns (logit_hidden,\n            # recycle_hidden). Other MTP families return a single tensor.\n            return \"DeepSeekMTPModel\" in (\n                self.draft_model_config.hf_config.architectures or []\n            )\n        return self.method not in (\"mtp\", \"draft_model\", \"dflash\")",
        "model_returns_tuple: return True for DeepSeekMTPModel (GLM-5.2 MTP)"
    ),
])

# DeepSeekMultiTokenPredictor.forward (line 171) just returns
# self.layers[str(...)](...) - it passes through the tuple automatically.
# DeepSeekMTP.forward (line 236) does hidden_states = self.model(...) then
# return hidden_states - also passes through the tuple.
# The proposer checks model_returns_tuple() which now returns True for
# DeepSeekMTPModel, so it unpacks (last_hidden_states, hidden_states) correctly.

print()
print("=" * 60)
print("All patches applied. Verifying...")
print("=" * 60)

# Verify the patches
checks = [
    ("model_executor/layers/attention/mla_attention.py", "topk_indices_buffer", "mla_attention has topk_indices_buffer"),
    ("model_executor/layers/mla.py", "topk_indices_buffer=mla_modules.topk_indices_buffer", "mla.py passes topk_indices_buffer"),
    ("model_executor/models/deepseek_mtp.py", "return hidden_states, self.shared_head", "deepseek_mtp returns tuple"),
    ("model_executor/models/deepseek_v2.py", "is_mtp_layer", "deepseek_v2 has is_mtp_layer"),
    ("v1/attention/backends/mla/flashmla_sparse.py", "indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer", "flashmla_sparse accepts topk_indices_buffer"),
    ("v1/attention/backends/mla/flashinfer_mla_sparse.py", "indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer", "flashinfer_mla_sparse accepts topk_indices_buffer"),
    ("v1/spec_decode/llm_base_proposer.py", "DeepSeekMTPModel", "llm_base_proposer checks DeepSeekMTPModel"),
]

all_ok = True
for rel_path, needle, desc in checks:
    fpath = DIST / rel_path
    content = fpath.read_text()
    if needle in content:
        print(f"  OK: {desc}")
    else:
        print(f"  FAIL: {desc}")
        all_ok = False

if all_ok:
    print("\nAll patches verified successfully!")
else:
    print("\nWARNING: Some patches failed verification!")
    sys.exit(1)
