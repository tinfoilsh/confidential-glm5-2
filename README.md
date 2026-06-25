# confidential-glm5-2

Patched vLLM v0.23.0 image for GLM-5.2 FP8 on 8xH200.

## Image

- Base: `vllm/vllm-openai:v0.23.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches:
  - `patches/0001-glm52-sparse-mla-dcp-fp8.patch` — sparse MLA DCP + fp8 KV-cache
    (backport of vllm-project/vllm#45426)
  - `patches/apply_mtp_patch.py` — MTP speculative decoding fix
    (backport of vllm-project/vllm#45895)
- Production config: `--tensor-parallel-size 8`,
  `--decode-context-parallel-size 8`, `--kv-cache-dtype fp8`,
  `--dcp-sparse-indexer-mode union`, `--speculative-config '{method:mtp,num_speculative_tokens:5}'`
- RunAI streamer config: `RUNAI_STREAMER_CONCURRENCY=8`,
  `RUNAI_STREAMER_MEMORY_LIMIT=4294967296`

## MTP Speculative Decoding

GLM-5.2 has built-in Multi-Token Prediction (MTP) with 5 draft tokens.
Unlike Eagle, MTP uses the model's own MTP layer — no separate draft model needed.

The MTP patch (PR #45895) fixes two issues:
1. **Indexer init skip**: Backbone "skip" layers (without an indexer) crash
   because sparse MLA backends assert `indexer is not None`. The fix passes
   `topk_indices_buffer` explicitly so skip layers can share the buffer.
2. **Post-final-norm hidden recycling**: MTP draft steps need the post-final-norm
   hidden state. The fix makes `DeepSeekMultiTokenPredictorLayer.forward` return
   `(hidden_states, shared_head(hidden_states))` and `model_returns_tuple()`
   return True for DeepSeekMTPModel.

Without this patch, MTP acceptance is ~3 tokens. With it, acceptance rises to ~4.

## H200 Compatibility

GLM-5.2 uses sparse MLA (FlashMLA_SPARSE / FlashInfer_MLA_SPARSE), not TRITON_MLA.
No Blackwell-specific features are used. The sparse MLA DCP patch and MTP patch
work on both H200 and B200/B300.

## Build

```bash
docker build --network host -t confidential-glm5-2 .
```
