# confidential-glm5-2

Patched vLLM v0.23.0 image for GLM-5.2 FP8 on 8xH200.

## Image

- Base: `vllm/vllm-openai:v0.23.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches:
  - `patches/0001-glm52-sparse-mla-dcp-fp8.patch` — sparse MLA DCP + fp8 KV-cache
    (backport of [vllm#45426](https://github.com/vllm-project/vllm/pull/45426))
  - `patches/0002-glm52-mtp-speculative.patch` — MTP speculative decoding fix
    (backport of [vllm#45895](https://github.com/vllm-project/vllm/pull/45895))
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
