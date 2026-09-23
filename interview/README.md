# 面试资料索引

这个文件夹把 DeepSeek V4 Flash / DSpark on 2x DGX Spark 这个仓库，整理成求职面试可直接使用的技术叙事和问答材料。

原则：讲清系统边界、工程难点、定位思路和可复现证据；**不把 DeepSeek、vLLM、DGX Spark 的上游能力说成自己的发明**。

## 文件地图

| 文件 | 作用 | 对应简历亮点 |
| --- | --- | --- |
| `01-system-overview.md` | 系统定位与边界、目录角色、参数与 patch 概览 | 项目总述 |
| `02-distributed-inference.md` | TP=2 双机推理：rank、权重可见性、NCCL 合并、通信延迟优化 | 亮点 1、2 |
| `03-long-context-and-kv-cache.md` | 1M 上下文：YaRN、`nvfp4_ds_mla`、PagedAttention、并发边界 | 亮点 3 |
| `04-speculative-decoding.md` | DSpark 推测解码：draft/target、acceptance、相关 patch | 亮点 4 |
| `05-interview-playbook.md` | 30 秒/2 分钟稿、亮点地图、不要这样讲 | 表达层 |
| `06-star-stories.md` | STAR 故事库（可展开讲的项目经历） | 表达层 |
| `07-qa-and-numbers.md` | 追问题库 + 数字证据口径表 | 表达层 |

## 建议使用方式

1. **先读 05**，确定 30 秒自我介绍和两分钟深挖模板，建立主线。
2. **再读 02/03/04**，把四个亮点的技术细节补齐；每篇都能独立回答一类追问。
3. **用 06 练故事**，按 STAR 复述，重点讲清 Situation/Task/Action/Result。
4. **用 07 备数字**，所有数字都带硬件、配置、prompt 类型、冷热状态和统计口径。
5. **读 01 收边界**，确保不会把上游能力说成自己的成果。

## 30 秒项目介绍

> 这个项目是在 2 台 DGX Spark 上部署和验证 DeepSeek V4 Flash / DSpark 的分布式推理 recipe。它把模型以 TP=2 跑在 vLLM 上，rank0 在 head 节点对外提供 OpenAI-compatible API，rank1 在 worker 节点 headless 参与推理；双机数据面通过 NCCL over RoCE/InfiniBand 做 tensor collective。项目还支持默认 1M context，并围绕 DSpark speculative decoding、NVFP4-MLA KV Cache (`nvfp4_ds_mla`)、并发稳定性、冷启动 garble 和真实 agent benchmark 做了系统化修复和验证。

收束句：

> 它不是简单拉起一个模型，而是把双机推理、长上下文、spec decode、KV cache、网络配置和稳定性验证串成了一套可复现工程 recipe。

## 两分钟技术深挖模板

1. 系统入口：head 提供 OpenAI-compatible API，worker 由脚本和 compose 拉起。
2. 分布式执行：TP=2，rank0/rank1 共同 forward，NCCL/RoCE 做 all-reduce/all-gather。
3. 权重和显存：文件层完整 checkpoint 可读，GPU 层主要大权重按 TP shard 加载。
4. 长上下文：`max_model_len=1048576` 对齐 YaRN ceiling，KV cache 靠 `nvfp4_ds_mla`、PagedAttention、chunked prefill 和 prefix caching 管理。
5. DSpark 加速：draft model 生成候选，target model 验证，acceptance 决定吞吐。
6. 修复与验证：处理并发、ragged context、冷启动 garble、shared expert loader，并用 benchmark 证明。

面试收尾句：

> 这个项目最能体现的是：我不仅能把大模型服务跑起来，还能解释双机推理的数据流，定位 vLLM 调度和 KV cache 相关问题，用 benchmark 证明修复有效，并把整套部署过程沉淀成别人可以复现的工程 recipe。
