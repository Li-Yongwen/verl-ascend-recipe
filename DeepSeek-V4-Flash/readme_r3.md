# R3 (Router Replay) 增量合入说明

R3（Router Replay，路由重放）用于保证 MoE 模型 RL 训练与 rollout 阶段的专家路由一致性：rollout 阶段记录 router top-k 决策，训练阶段按记录重放，消除训练/推理路由不一致带来的偏差。

本方案将 `verl/routing_replay/output` 下的 R3 补丁以**增量补丁**形式合入本 recipe：新增 `patch/r3/` 目录存放 6 个 R3 增量补丁，与基线补丁 `patch/` 完全隔离，安装时通过开关选择是否启用。

## 1. 与基线补丁的差异对比（不多合、不少合）

先对比原始 R3 patch（`routing_replay/output`，基于开发环境 HEAD 的全量 diff）与 recipe 已有基线补丁，逐组件确认差异点，再计算纯 R3 增量：

| 组件 | 原始 R3 patch | 对比结论 | 合入处理（`patch/r3/`） |
| :--- | :--- | :--- | :--- |
| vllm | `vllm.patch`（1 个全新文件） | recipe 无 vllm 基线补丁，无重合 | 原样收录 |
| Megatron-LM | `megatron-lm.patch` | 基线部分与 recipe `megatron.patch` 完全一致，剔除后仅剩 R3 逻辑 | 纯增量（2 文件） |
| vllm-ascend | `vllm-ascend.patch` | 基线部分与 recipe `vllm-ascend.patch` 完全一致，剔除后仅剩 R3 逻辑 | 纯增量（6 文件） |
| MindSpeed-LLM | `mindspeed-llm.patch`（含整文件 dump） | recipe 基线无补丁，通过 `cp pretrain_deepseek4.py mindspeed_llm` 提供基础文件 | 整文件改为基于 cp 结果的修改 hunk（6 文件），避免覆盖基线 |
| verl | `verl.patch` | recipe 基线已含 router replay 基础框架，且基线比 R3 开发基线新（`VLLM_SLEEP_LEVEL`、A5 IPC 检测、fp8 import 容错等非 R3 演进） | 剔除重合部分并重定基到 recipe 新基线（7 文件） |
| mbridge | `mbridge.patch` | recipe 基线比 R3 开发基线新（含 `llm_bridge.py`/`deepseek_v3.py` 兼容性回退） | 重定基到 recipe 新基线（8 文件） |

验证方式：对每个组件按「干净仓库 → 基线 commit → apply recipe 基线补丁 → apply R3 增量补丁」重放，与 `routing_replay` 开发态工作树逐文件 diff。结果：vllm / vllm-ascend / megatron-lm / mindspeed-llm 全等；verl / mbridge 仅存在 recipe 基线演进的非 R3 差异（上述基线更新点，属预期），R3 功能文件全部一致，无少合多合。

## 2. 安装

R3 默认关闭，安装方式与基线完全相同：

```bash
bash install.sh            # 不含 R3
bash install.sh --r3       # 启用 R3（等价于 INSTALL_R3=1 bash install.sh）
```

`--r3` / `INSTALL_R3=1` 时 `install.sh` 的变化：

1. 第 7 步前将 `vllm-ascend`、`MindSpeed-LLM` 固定到 R3 补丁的生成基线（其余仓库已由 tag / commit 固定），保证补丁可干净应用：

   | 组件 | 基线版本 |
   | :--- | :--- |
   | vllm | v0.23.0（tag，不变） |
   | vllm-ascend | **4e5f393af**（releases/v0.23.0 分支） |
   | mbridge | v0.15.1（tag，不变） |
   | verl | 809f2d8f（不变） |
   | Megatron-LM | core_v0.12.1（tag，不变） |
   | MindSpeed-LLM | **3d86279d**（master） |

2. 在第 7 步基线补丁之后追加第 8 步，依次应用 `patch/r3/` 下 6 个增量补丁（顺序：megatron-lm → vllm → mbridge → vllm-ascend → verl → mindspeed-llm）。

R3 增量补丁依赖基线补丁先行应用，请勿单独使用。回退 R3：重新执行不带 `--r3` 的 `install.sh`，或进入各仓库 `git checkout .` 撤销。

## 3. R3 补丁内容

| 补丁 | 文件 | 说明 |
| :--- | :--- | :--- |
| `patch/r3/verl.patch` | `verl/utils/megatron/router_replay_patch.py`<br>`verl/utils/megatron/router_replay_utils.py`<br>`verl/utils/vllm/npu_vllm_patch.py`<br>`verl/workers/engine/megatron/transformer_impl.py`<br>`verl/workers/engine/mindspeed/transformer_impl.py`<br>`verl/workers/engine_workers.py`<br>`verl/workers/rollout/vllm_rollout/vllm_rollout.py` | Router replay 核心逻辑：记录 / 重放动作管理、`set_router_replay_data`（rmpad / SP 对齐）、训练引擎 hook 接入 |
| `patch/r3/vllm.patch` | `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | 新增：rollout 阶段捕获 routed experts top-k |
| `patch/r3/vllm-ascend.patch` | `vllm_ascend/core/single_type_kv_cache_manager.py`<br>`vllm_ascend/models/deepseek_v4.py`<br>`vllm_ascend/ops/rope_dsv4.py`<br>`vllm_ascend/patch/worker/__init__.py`<br>`vllm_ascend/worker/block_table.py`<br>`vllm_ascend/worker/model_runner_v1.py` | 压缩 KV cache 下 `cdiv` 对齐修复、block span / slot mapping 重建、routed experts 输出导出 |
| `patch/r3/mbridge.patch` | `mbridge/__init__.py`<br>`mbridge/core/bridge.py`<br>`mbridge/core/llm_bridge.py`<br>`mbridge/models/__init__.py`<br>`mbridge/models/deepseek_v3.py`<br>`mbridge/models/deepseek_v4.py`<br>`mbridge/models/ext/deepseek_v3/dequant_fp8_safetensor_io.py`<br>`pyproject.toml` | 新增 `DeepseekV4Bridge`、GEMM 专家布局（含 `llm_bridge.py` / `deepseek_v3.py` 兼容性调整） |
| `patch/r3/megatron-lm.patch` | `megatron/core/transformer/moe/moe_utils.py`<br>`megatron/core/transformer/transformer_config.py` | `compute_topk` 接入 router replay record / replay 动作 |
| `patch/r3/mindspeed-llm.patch` | `mindspeed_llm/core/transformer/moe/moe_utils.py`<br>`mindspeed_llm/core/transformer/moe/router.py`<br>`mindspeed_llm/core/transformer/transformer_block.py`<br>`mindspeed_llm/features_manager/pipeline_parallel/num_layer_list.py`<br>`mindspeed_llm/pretrain_deepseek4.py`<br>`pretrain_deepseek4.py` | MindSpeed-LLM 侧路由重放集成；同时修改上游根目录 `pretrain_deepseek4.py` 与 `mindspeed_llm/pretrain_deepseek4.py`（cp 副本），重复执行 `cp` 也不会丢失 R3 改动 |

## 4. 使用

安装启用 R3 后，通过训练配置开启（详见 verl `EngineRouterReplayConfig`）：

```bash
# 本 recipe 使用 mindspeed 后端，注意配置节点是 actor.mindspeed 而非 actor.megatron
actor_rollout_ref.actor.mindspeed.router_replay.mode=R3   # disabled（默认）/ R2 / R3
```

配置读取按 strategy 定位（`getattr(actor, strategy).router_replay`），megatron 后端则对应 `actor.megatron.router_replay.mode`。

其余启动流程与基线一致，参见 [readme.md](readme.md)。

## 5. 变更清单

- 新增 `patch/r3/`：6 个 R3 增量补丁（均为 LF 行尾）
- 修改 `install.sh`：新增 `--r3` / `INSTALL_R3` 开关、R3 模式下固定 `vllm-ascend` / `MindSpeed-LLM` commit、第 8 步应用 R3 补丁
- 基线补丁 `patch/` 目录内容不变（`vllm-ascend.patch` 仅修复本地工作区 CRLF 行尾，仓库内容无变化）
