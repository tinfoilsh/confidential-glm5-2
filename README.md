# confidential-glm5-2

Patched vLLM v0.25.0 image for GLM-5.2 FP8 on 8xH200.

## Image

- Base: `vllm/vllm-openai:v0.25.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches:
  - `patches/0001-glm52-sparse-mla-dcp-fp8.patch` — sparse MLA DCP + fp8
    KV-cache for the FlashMLA sparse backend, with
    `--dcp-sparse-indexer-mode union|exact`
    (backport of [vllm#45426](https://github.com/vllm-project/vllm/pull/45426),
    closed upstream in favor of the still-open
    [vllm#46514](https://github.com/vllm-project/vllm/pull/46514); note #46514
    is exact-merge only — retiring this patch means giving up union mode)
  - `patches/0002-fp8-ds-mla-kv-spec.patch` — sets `kv_quant_mode` on the MLA
    KV-cache spec so `fp8_ds_mla` boots
    (cherry-pick of the open
    [vllm#48379](https://github.com/vllm-project/vllm/pull/48379); retire when
    it lands in a release we consume)
- Production config: `--tensor-parallel-size 8`,
  `--decode-context-parallel-size 8`, `--kv-cache-dtype fp8`,
  `--dcp-sparse-indexer-mode union`, `--attention-backend FLASHMLA_SPARSE`,
  `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`
- RunAI streamer config: `RUNAI_STREAMER_CONCURRENCY=8`,
  `RUNAI_STREAMER_MEMORY_LIMIT=4294967296`

## Notes

- `--attention-backend FLASHMLA_SPARSE` is required: the upstream DCP sparse
  backend (`FLASHINFER_MLA_SPARSE`) is SM100-only and cannot run on H200
  (SM90); patch 0001 additionally rejects it under DCP because the patched
  indexer merge uses FlashMLA-sparse local-index semantics.
- `/root` tmpfs must be at least 20g with `exec` enabled for Triton compilation.

## Build

```bash
docker build --network host -t confidential-glm5-2 .
```
