# 追问题库与数字证据

这份文档分两部分：前半是面试常见追问的回答要点，后半是所有可引用数字的**配置与口径**。原则是：说数字必须带上下文，否则一个孤立的 tok/s 没有意义。

## 1. 追问题库

### 权重是在两台机器都下载吗？

默认是的，或者至少两台机器都要能读到完整 checkpoint。每个 rank 启动时从标准 checkpoint 加载自己的 TP shard。不要说“head 下载前半，worker 下载后半”；应说“文件层完整可见，GPU 层按 TP 分片”。

### 两台机器都计算，那结果是不是要合并？

要合并，但不是合并自然语言。row-parallel 层会 all-reduce partial hidden；vocab-parallel logits 会 gather 候选或 logits 片段再做全局选择。最终 head 返回同一个 token 序列。

### rank0 和 rank1 的 rank 是什么？

rank 是分布式进程组里的进程编号，不是主从关系。本项目 `world_size=2`：head 是 `NODE_RANK=0`，worker 是 `NODE_RANK=1 HEADLESS=1`。rank0 多了 API server；两者都参与 TP forward。

### 项目如何降低双机通信延迟？

它主要避免走慢路径：显式设置 `NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`、`VLLM_HOST_IP`、`WORKER_VLLM_HOST_IP`，让数据面走 NCCL over RoCE/IB；使用 `network_mode: host` 和 `/dev/infiniband` 暴露 RDMA 设备；双 HCA 场景用 `NCCL_IB_MERGE_NICS=1`。worker-first 解决启动 rendezvous 竞态，不是直接降低 token latency。

> 若被追问带宽：仓库早期 nccl-tests 记录单 HCA ~98 Gb/s、双 HCA 合并 ~161 Gb/s（+64%）；但后续 2026-08-20 campaign 的 dual-HCA 实验（E6）显示，对**本推理负载**该互连不是瓶颈（decode/32K/128K prefill 均在噪声内）。稳妥说法是“避免走 socket 慢路径”，不承诺带宽数字直接换成吞吐。

### 为什么很多方案用 2 台 DGX Spark？

DeepSeek V4 Flash 是大 MoE，active params 低不等于只加载 active 权重。单台 Spark 有 128GB unified memory，双 Spark 通过 TP=2 分摊权重、KV 和 workspace，并用 ConnectX-7/RoCE/NCCL 承担跨节点 collective。它不是透明的 256GB unified memory。

### 1M 上下文和并发是怎么同时支持的？

`max_model_len` 和 `max_num_seqs` 都是上限而不是预留；KV pool 按 `sum(live tokens)` 动态分配。短/中上下文的 agent 请求可以多个共享同一个 pool；多条满 1M 长文会排队或触发 preemption。不要把 `max_num_seqs=6` 说成“6 条都能满 1M”。

### 为什么默认 1M 而不是 1.5M？

1M 是 YaRN 校准出的 ceiling（`65536 × 16`）。1.5M 是历史压力实验，超过校准范围，输出质量不作承诺；更长上下文还会挤压 KV pool。`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 只是让 vLLM 接受更长上限，不代表质量。

### `k` 为什么是 5，不是模型卡建议的 7？

`dspark_block_size = 5`，draft 每 pass 只发 5 列。在本 recipe 的镜像上，`k=7` 会先被 `hf_config_override` 的整除 guard 拒绝，强改 guard 后又在生成时因 `7 vs 5` 的形状不匹配崩溃。准确规则是 `k <= 5` 或 5 的倍数。更深 draft 也不是缺的速度：位置 4-5 在难内容上接受率很低。

### DSpark 和“外接一个小 draft 模型”有什么区别？

本项目不是外挂 draft 模型，而是用 checkpoint **内置**的 DSpark draft head：target 层 hidden 投影进 DSpark main KV，再由 Markov head + confidence head 产出候选。target model 仍然负责验证，所以 draft 猜错通常损失速度，不直接污染输出。

### 怎么证明输出是对的，而不只是快？

target model 会验证每个候选 token；修复 shared expert loader 提升的是 acceptance，不改变输出正确性。此外用 smoke、`agent_sanity_bench.py` 的 `bad_outputs`、以及对 churn 下确定输出的字节一致性来验证。

### 为什么不用 fp8 KV 一劳永逸？

`fp8_ds_mla` 与 `nvfp4_ds_mla` 对 acceptance 没有区别，选择取决于 KV pool 大小；本项目用 NVFP4 是为了在 2x DGX Spark 上腾出 1M 上下文的显存余量。

### 你觉得最难的部分是什么？

不是拉起模型，而是把“看起来能用”变成“可解释、可复现、可定位”：区分 step 慢（NCCL/MoE/KV/CUDA graph）和 acceptance 低（draft 权重/loader/上下文对齐），并把每个性能数字绑定到硬件、配置、prompt 类型和冷热状态。

## 2. 数字证据表

除非特别说明，均为 2x DGX Spark（GB10）、TP=2、`k=5`、`nvfp4_ds_mla`、`max_model_len=1048576`、`max_num_seqs=6`、temp 0、warm 状态；口径为服务端 `completion_tokens / wall time`。

### Decode（按内容类型，2026-07-29）

| prompt | tokens | tok/s |
| --- | ---: | ---: |
| count to 300 | 600 | 84.3 |
| 12x12 multiplication table | 900 | 77.9 |
| 60-object JSON array | 800 | 77.0 |
| binary search tree (code) | 600 | 64.2 |
| 200-word narrative | 251 | 34.6 |
| **peak / mean** | | **84.3 / 67.6** |

要点：34→84 的差距来自 DSpark acceptance 随内容变化，单点 tok/s 必须带 prompt 类型。

### 并发（同 prompt，各 400 tokens，aggregate）

| concurrency | aggregate tok/s | per-stream tok/s |
| ---: | ---: | ---: |
| 1 | 61.0 | 61.0 |
| 2 | 91.7 | 46.9 |
| 4 | 151.1 | 38.7 |
| 6 | 197.3 | 33.6 |

### Prefill（TTFT 方法，1 output token）

| prompt depth | prompt tokens | tok/s |
| ---: | ---: | ---: |
| 8K | 6,234 | 1,513 |
| 32K | 24,900 | 2,284 |
| 100K | 77,790 | 2,639 |

### 真实混合 agent 流量（`benchmarks/soak.py`）

| 口径 | benchmark (BST, temp 0) | realistic mixed traffic |
| --- | ---: | ---: |
| per-stream tok/s @ c4 | 38.7 | 22.3 |
| aggregate tok/s @ c4 | 151.1 | 88.6 |

规模：c4，40 分钟，553 请求，212,974 generated tokens，零退化输出、零错误。**容量规划用 ~88 aggregate / ~22 per-stream @ c4**，而不是 151。

### Shared expert loader 修复（0731，TP=2，k=5，nvfp4 KV，1M）

| | accept | tok/step | steps/s | mean tok/s |
| --- | ---: | ---: | ---: | ---: |
| stock loader | 25.7% | 2.28 | 14.4 | 32.7 |
| **Patch 4** | **60.2%** | **4.01** | 13.8 | **55.4** |

patched 在 peak-finder prompt 上达 **78.4 tok/s @ 98.9% acceptance**（5.95/6 每步）；冷启动后同 prompt 为 56.8 tok/s（冷启惩罚，非配置差异）。

### 冷启动与热态

| 状态 | count300 tok/s |
| --- | ---: |
| 刚启动（已 capture CUDA graph + 3 次短 warmup） | 58.5 |
| 若干长生成后 | 83.3 |

约 30% 冷启惩罚，boot log 不显示；空闲约 30 分钟后也会衰减（60.4 → 83.5）。

### 其他可引用 checkpoint

| 场景 | 配置 | 数值 |
| --- | --- | --- |
| 1M / 6 微基准 | `max_model_len=1048576`, `max_num_seqs=6` | KV pool 1,901,239 tokens，c6 ~182 tok/s aggregate |
| 200K / 16 并发 | `max_model_len=200000`, `max_num_seqs=16` | 静态 c16 315.1 tok/s，staggered c16 205.0 tok/s |
| C12 1.5M | `max_model_len=1500000`, `max_num_seqs=12` | KV pool 3,225,280 tokens，c12 230.10 tok/s |
| 诚实分类 (2026-07-04) | temp 0, 5 prompts | Math 60.1 / JSON 54.0 / Code 53.8 / Comm 42.0 / Narrative 33.7，mixed 48.7 |

## 3. 引用数字的纪律

- 说清硬件、配置、prompt 类型、冷/热状态和统计口径。
- spec decode 下不要用 streaming chunk 数当 tok/s；chunk 数更接近 decode step。
- KV pool 是每次启动的实测值（曾观察到 1.39M–1.53M 的 11% 波动），不是固定属性。
- 区分 benchmark（单一 prompt）和容量规划（混合流量），前者用于对比配置，后者用于报数。
