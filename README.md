# confidential-glm5-2

Patched vLLM v0.24.0 image for GLM-5.2 FP8 on 8xH200.

## Image

- Base: `vllm/vllm-openai:v0.24.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches:
  - `patches/0001-glm52-sparse-mla-dcp-fp8.patch` — sparse MLA DCP + fp8 KV-cache
    (backport of [vllm#45426](https://github.com/vllm-project/vllm/pull/45426),
    closed upstream in favor of the still-open
    [vllm#46514](https://github.com/vllm-project/vllm/pull/46514); retire the
    patch when that lands in a release we consume)
  - `patches/0002-reasoning-boundary-bitmask-44297.patch` — constrain the
    reasoning-end bitmask and trim grammar advance at the reasoning→content
    boundary under MTP spec decode (backport of
    [vllm#44297](https://github.com/vllm-project/vllm/pull/44297), merged
    2026-07-04 after v0.24.0 was cut; retire when the base image includes #44297)
  - `patches/0003-reasoning-boundary-advance-44993.patch` — advance the grammar
    FSM across the reasoning→content boundary for json_object/regex/choice under
    MTP spec decode, fixing malformed JSON output and strict-tool-call 500s on
    GLM-5.2 (backport of [vllm#44993](https://github.com/vllm-project/vllm/pull/44993),
    open/approved as of 2026-07-15; stacked on #44297; retire when the base
    image includes #44993)
- Production config: `--tensor-parallel-size 8`,
  `--decode-context-parallel-size 8`, `--kv-cache-dtype fp8`,
  `--dcp-sparse-indexer-mode union`, `--attention-backend FLASHMLA_SPARSE`,
  `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`
- RunAI streamer config: `RUNAI_STREAMER_CONCURRENCY=8`,
  `RUNAI_STREAMER_MEMORY_LIMIT=4294967296`

## Notes

- `--attention-backend FLASHMLA_SPARSE` is required for DCP with sparse MLA.
  The default `FLASHINFER_MLA_SPARSE` does not return softmax LSE during decode.
- `/root` tmpfs must be at least 20g with `exec` enabled for Triton compilation.

## Build

```bash
docker build --network host -t confidential-glm5-2 .
```
