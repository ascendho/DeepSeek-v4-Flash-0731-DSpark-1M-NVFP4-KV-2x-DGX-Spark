# Repository Guidelines

## Project Structure & Module Organization

This repository is a deployment and benchmarking recipe for DeepSeek V4 Flash / DSpark on 2x DGX Spark. The main operational documentation is in `README.md`, with focused notes in top-level `*-UPDATE*.md`, `DEFAULT-CONFIG.md`, and `DSPARK-SHARED-EXPERT-FIX.md`. Runtime overlay source lives under `recipe/overlay/vllm/...`; Docker build stages live in `recipe/nvfp4/` and `recipe/official-main/`. Deployment entrypoints and diagnostics are top-level `*.sh` scripts plus helper tools in `scripts/`. `sparkrun/` contains the reproducible YAML recipe and speed-test script. Benchmark records belong in `benchmarks/`; patch files belong in `patches/`.

## Build, Test, and Development Commands

- `./scripts/verify-overlay-sources.sh`: checks that Dockerfile `COPY` sources exist in `recipe/overlay`.
- `./build-dspark-vllm-runtime.sh`: builds the local Stage C runtime image; requires Docker and the expected DGX Spark base image.
- `./validate-dspark-config.sh`: renders and sanity-checks `.env.dspark` plus `docker-compose.dspark.yml`.
- `./start-deepseek-v4-flash-dspark.sh` / `./stop-deepseek-v4-flash-dspark.sh`: manage the two-node runtime.
- `./smoke-deepseek-v4-flash-dspark.sh`: sends concurrent OpenAI-compatible requests to the running server.
- `DSPARK_BASE_URL=http://host:8888/v1 python3 scripts/agent_sanity_bench.py`: runs the Python concurrency and output-quality gate.

## Coding Style & Naming Conventions

Shell scripts use Bash with `set -euo pipefail`; keep environment overrides explicit, for example `ENV_FILE=...` or `WORKER_BUILD=0`. Python scripts are Python 3, standard-library first, four-space indentation, `snake_case` names, and concise module docstrings. Preserve upstream vLLM path structure under `recipe/overlay/vllm/` so patches and Docker `COPY` rules remain traceable. Name benchmark notes with dates and hardware/config context, e.g. `benchmarks/20260702-...md`.

## Testing Guidelines

There is no package-level unit-test harness. Validate changes with the smallest relevant gate first: overlay source check for Dockerfile edits, config validation for `.env` or compose changes, smoke tests for serving changes, and `agent_sanity_bench.py` for concurrency/output regressions. For runtime patches, document the exact model, image, context length, concurrency, and token/s measurement method.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive subjects, often scoped (`docs: ...`, `loop_detector: ...`, `Campaign 2026-08-20: ...`). Keep commits focused and include measured evidence when changing runtime behavior. Pull requests should explain the affected configuration, list verification commands, link related issues, and attach benchmark tables or logs when performance, correctness, or deployment behavior changes.

## Security & Configuration Tips

Do not commit `.env.dspark`, credentials, hostnames, or private model-cache paths. Use `.env.dspark.example` for documented defaults. Treat GHCR publishing and self-hosted runner details in `.github/workflows/publish-ghcr.yml` as operationally sensitive; update them only with a clear deployment reason.
