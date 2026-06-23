# confidential-glm5-2

Patched vLLM v0.23.0 image for GLM-5.2 FP8 on 8xH200.

## Image

- Base: `vllm/vllm-openai:v0.23.0-ubuntu2404` (digest-pinned in `Dockerfile`)
- Patches: sparse MLA DCP + fp8 KV-cache support for GLM-5.x
  (`patches/0001-glm52-sparse-mla-dcp-fp8.patch`, backport of
  vllm-project/vllm#45426)
- Target canaries: `--decode-context-parallel-size 4` and `8` with
  `--kv-cache-dtype fp8`

## Build

```bash
docker build --network host -t confidential-glm5-2 .
```
