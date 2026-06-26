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
  `--dcp-sparse-indexer-mode union`, `--attention-backend FLASHMLA_SPARSE`,
  `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`
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

**Attention Backend:** `--attention-backend FLASHMLA_SPARSE` is required when
using DCP with sparse MLA. The default `FLASHINFER_MLA_SPARSE` backend does not
return softmax LSE during decode, which DCP requires. The DCP patch adds
`can_return_lse_for_decode: bool = True` to `FlashMLASparseImpl` but not to
`FlashInferMLASparseImpl`.

**Triton Cache:** The `/root` tmpfs must be at least 20g with `exec` enabled
(`--tmpfs /root:size=20g,exec`). Docker's default tmpfs mount uses `noexec`,
which causes `ImportError: failed to map segment from shared object` when
Triton tries to load compiled CUDA kernels from `/root/.triton/cache/`.

## Test Results (B300, 8x B200, DCP=8, fp8, MTP=5)

End-to-end testing confirmed MTP speculative decoding works correctly with
DCP=8 and fp8 KV cache using `FLASHMLA_SPARSE` attention backend.

**MTP Acceptance Metrics:**
- Draft tokens: 455 across 91 drafts (5 tokens/draft)
- Accepted tokens: 296 (65% overall acceptance rate)
- Per-position acceptance: pos0=80, pos1=66, pos2=56, pos3=51, pos4=43
  (decreasing acceptance at deeper positions is expected)

**Correctness Tests:**
- Basic chat (non-streaming): PASS - model returns correct answers with reasoning
- Streaming chat: PASS - tokens stream correctly with reasoning + content
- Tool calling (non-streaming): PASS - model calls get_weather with location
- Tool calling (streaming): PASS - tool call streams correctly with finish_reason: tool_calls
- Structured JSON output: PASS - model returns valid JSON with guided_json

**Server Stats:**
- Model loading: 90.34 GiB, ~880s
- KV cache: 142.77 GiB, 22.408M tokens, 56.99x max concurrency
- CUDA graph capture: 302s, 3.87 GiB
- Engine init (profile + create kv cache + warmup): 1188s

## Build

```bash
docker build --network host -t confidential-glm5-2 .
```
