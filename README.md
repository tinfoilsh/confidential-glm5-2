# confidential-glm5-2

Patched vLLM v0.23.0 image for GLM-5.2 FP8 on 8xH200.

## Image

- Base: `vllm/vllm-openai:v0.23.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches: sparse MLA DCP + fp8 KV-cache support for GLM-5.x
  (`patches/0001-glm52-sparse-mla-dcp-fp8.patch`, backport of
  vllm-project/vllm#45426)
- Production config: `--tensor-parallel-size 8`,
  `--decode-context-parallel-size 8`, `--kv-cache-dtype fp8`,
  `--dcp-sparse-indexer-mode union`
- RunAI streamer config: `RUNAI_STREAMER_CONCURRENCY=8`,
  `RUNAI_STREAMER_MEMORY_LIMIT=4294967296`
- MTP was tested with DCP8 and fp8 KV-cache, but it regressed throughput
  across the production-shaped benchmark and is not enabled.

## Build

```bash
docker build --network host -t confidential-glm5-2 .
```
