# confidential-glm5-2

Patched vLLM v0.27.0 image for GLM-5.2 FP8 on 8xB200.

## Image

- Base: `vllm/vllm-openai:v0.27.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches:
  - `patches/0001-uva-device-mirror.patch` — replace UVA zero-copy host
    buffers with device mirrors + staged copies; GPU reads of host-mapped
    memory are unsafe under confidential computing
- Production config: `--tensor-parallel-size 8`, `--kv-cache-dtype fp8`,
  `--attention-backend FLASHMLA_SPARSE`,
  `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`
- RunAI streamer config: `RUNAI_STREAMER_CONCURRENCY=8`,
  `RUNAI_STREAMER_MEMORY_LIMIT=4294967296`

## Notes

- v0.27.0 retires two patches the v0.26.0 build carried: FlashMLA-sparse
  fp8 KV-cache support is upstream (including SM100), and
  [vllm#44993](https://github.com/vllm-project/vllm/pull/44993) (grammar
  advance across the reasoning boundary under MTP spec decode) shipped in
  the release.
- Decode context parallelism is off on B200: per-GPU HBM fits the full
  393K-token fp8 KV cache without sharding, so the DCP sparse-indexer
  patch set ([vllm#45426](https://github.com/vllm-project/vllm/pull/45426)
  backport) is no longer applied. An H200 deployment needs it back —
  use a v0.0.19-lineage release for Hopper hosts.
- `/root` tmpfs must be at least 20g with `exec` enabled for Triton
  compilation.

## Build

```bash
docker build --network host -t confidential-glm5-2 .
```
