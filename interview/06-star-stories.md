# STAR 故事库

这份文档把项目里的关键经历整理成 STAR（Situation / Task / Action / Result）结构，方便面试时按需展开。每个故事都标注了可引用的仓库材料，回答时尽量带上硬件、配置和口径。

## 故事 A：双机不是一个服务输出两半文字

- **Situation**：单台 DGX Spark 是 128GB unified memory，DeepSeek V4 Flash 是大 MoE，装不下完整部署，需要跨两台机器推理。
- **Task**：让两台机器协同跑出**一个**模型服务，并对长上下文和并发保持稳定。
- **Action**：
  - 用 vLLM `--tensor-parallel-size 2`、`--nnodes 2`、`--node-rank` 组成一个分布式作业，rank0 在 head 提供 API，rank1 在 worker 以 `HEADLESS=1` 参与计算。
  - 文件层让两台机器都能读取完整 checkpoint；GPU 层由每个 rank 加载自己的 TP shard。
  - row-parallel 层用 NCCL all-reduce 合并 partial hidden，vocab-parallel logits 用 all-gather 做全局选择。
- **Result**：对外是一个 OpenAI-compatible 服务，合并的是 tensor/logits 候选，不是两段文本拼接。

可引用：`02-distributed-inference.md`。

## 故事 B：1M context 不是简单调大参数

- **Situation**：模型 `config.json` 用 YaRN，`original_max_position_embeddings=65536 × factor 16 = 1,048,576`，1M 是校准上限。
- **Task**：让服务真正支撑 1M 单请求，同时保留正常并发。
- **Action**：
  - `--max-model-len 1048576` 放开单请求上限。
  - 通过 Stage A/B/C 接入 `nvfp4_ds_mla`，为 DeepSeek V4 sparse MLA 降低 KV cache 显存占用。
  - 用 PagedAttention（`block-size 256`）按活跃 token 动态分配 KV，配合 chunked prefill 和 prefix caching 管理 prefill 压力。
- **Result**：在 2x DGX Spark 上把 1M context 跑成可运行配置；并发边界是 `sum(live tokens) <= KV pool`，不是 `max_num_seqs × 1M`。

可引用：`03-long-context-and-kv-cache.md`、`DEFAULT-CONFIG.md`。

## 故事 C：并发 bug 体现对 vLLM 调度语义的理解

- **Situation**：`--max-num-seqs > 1` 后，并发请求会触发异常或拿到错误的 draft context。
- **Task**：让 DSpark proposer 在 continuous batching 和 chunked prefill 下仍然正确。
- **Action**：
  - 引入 request-stable 的 DSpark main-KV slot，不把可复用的 batch row 当作请求的稳定身份。
  - 用 `query_start_loc` 走 ragged context 路径，正确处理每个 request 不等的 query row 数。
  - 把 prefill 与 speculative decode 的边界划清。
- **Result**：并发路径从偶发崩溃变成可验证运行，支撑后续 c6 并发 benchmark。

可引用：`04-speculative-decoding.md`、`docs/PATCHES.md`。

## 故事 D：shared expert loader 修复提升 acceptance

- **Situation**：切到官方 0731 checkpoint 后，输出质量没变，但 decode 吞吐掉了一半。
- **Task**：定位是权重、配置还是 loader 问题。
- **Action**：
  - 发现 vLLM 的 DSpark draft loader 静默丢弃 12 个 tensor：三个 draft stage 的 always-on shared expert `gate_up_proj`（`w1`/`w3`）没被映射，只命中 `logger.debug`。
  - 补上 loader mapping（Patch 4）。
- **Result**：target 仍负责验证所以输出正确，但 draft acceptance 从 **25.7% 回到 60.2%**，mean decode 从 **32.7 提升到 55.4 tok/s**；step 时间不变，全部是 acceptance 恢复。

可引用：`04-speculative-decoding.md`、`DSPARK-SHARED-EXPERT-FIX.md`、`patches/0004-*`。

## 故事 E：冷启动并发 garble（Patch 3 + CUDA graph 覆盖）

- **Situation**：冷启动、并发下，新会话的第一个 prompt 会吐出 tool-call 碎片 / XML junk，之后同会话又恢复干净，不像权重或输出质量问题。
- **Task**：定位这是采样问题还是 spec decode 问题，并给出稳定修复与代价量化。
- **Action**：
  - 判断根因是 DSpark 冷启动下 draft/target 分布不匹配：spec-config 没带 `draft_sample_method`（退化成 greedy draft），同时 `--max-cudagraph-capture-size` 没覆盖并发 batch，首批落到未捕获的 CUDA graph。
  - spec-config 显式补上 `draft_sample_method: probabilistic`，并把 `--max-cudagraph-capture-size` 设为覆盖 `max_num_seqs × (k+1)`（c6/k5 即 36），让并发首批命中已捕获图。
  - 移除有 spec-decode crash 风险的 `repetition_penalty`。
  - 复核备注：`draft_sample_method` 在当前 DSpark 树上其实是 no-op（drafter 始终 argmax），因此 garble 的确定性修复来自 **Patch 3** 与 CUDA graph 覆盖，需诚实说明。
- **Result**：修复后并发冷启动首批零 garble；速度几乎无损（61.8 → 61.3 tok/s，统计上一致）。

可引用：`README.md` 的 Garble fix 章节、`docs/PATCHES.md`、`AGENT_GARBLE_FIX.md`。

## 故事 F：可复现验证纪律（证明不是只跑通 demo）

- **Situation**：spec decode 的 tok/s 高度依赖 prompt 类型和冷热状态，单点“能跑通”不能作为证据。
- **Task**：建立能复现、有口径的验证方法。
- **Action**：
  - 用 smoke、`agent_sanity_bench.py`、`capture_runtime.sh` 分层验证；测速固定 `stream:false` 和 `usage.completion_tokens`。
  - 区分冷启动与热态：同一 prompt 冷启 58.5 tok/s，热态 83.3 tok/s，差值约 30%。
  - 做真实流量 soak：c4 下 40 分钟、553 请求、212,974 tokens，零退化输出、零错误。
- **Result**：能说清每个数字的硬件、配置、prompt 类型和统计口径，而不是只背一个峰值。

可引用：`benchmarks/`、`scripts/agent_sanity_bench.py`、`README.md` benchmark 章节。
