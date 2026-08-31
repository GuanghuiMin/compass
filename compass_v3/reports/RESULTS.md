# COMPASS v2 — 实验结果汇总（2026-08-25）

协议：AppWorld `test_normal` 前 30 题；frozen agent = gpt-4.1（ACON ICL prompt，temperature 0，观测上限 10 000 字符）；窗口 B = 4096 tokens，保留最后一轮；最多 50 步；压缩器 LLM = gpt-4.1-mini。
所有数字可由 `scripts/report_phase2.py --root artifacts/verify_L` 与 `scripts/report_episodes.py --root artifacts/final_r1 --root2 artifacts/final_r2` 重新生成。

## 1. Phase 1 — 离线建图（`reports/phase1_offline.json`，v1 语料 11 episodes / 35 boundaries）

| 指标 | 值 |
|---|---|
| 确定性建图成功率 | 100% |
| handover 在预算内 | 97%（1 次 LLM 降级 fallback） |
| live recall（之后真被消费的变量是否在 handover 中） | **1.00** |
| done 意图带真实 API 调用证据 | 92% |
| LLM 建议被丢弃项 | 26 / 35 boundaries（主要 `bad_realizes`），**0 次整体拒绝** |
| 平均 handover 长度 | ≈1000 tokens |

对比 v1：27 boundaries 中 12 refused / 3 empty（`compass_v1/artifacts/diagnostic/six_task_v1/AUDIT.md`）。

## 2. Phase 2 — boundary 级验证（Trace 协议，`reports/phase2_L.md`）

Cohort：`artifacts/dev41L/openclaw` 30 条 OpenClaw 轨迹的全部 64 个 compaction boundary；两臂各 2 次 gpt-4.1 回放、最多 5 步；PRE = 原始历史（后期 boundary：上一份记录摘要 + 原始后缀），POST = 压缩后。指标 = POST − PRE 的 blocked（dE）/ refetch（dR）/ 并集（dU）均值与 bootstrap 95% CI。

| POST 压缩器 | dU@k1 | dU@k3 | dU@k5 | dE@k5 | dR@k5 |
|---|---|---|---|---|---|
| PRE 控制组绝对值 | 0.12 | 0.32 | 0.56 | — | — |
| OpenClaw（记录摘要） | +0.31\* | +0.56\* | **+0.78 [+0.53, +1.06]\*** | +0.33\* | +0.47\* |
| COMPASS v2（初版，产出值挂载） | +0.27\* | +0.58\* | +0.69\* | +0.41\* | +0.30\* |
| COMPASS v2 + 调用形式 | +0.18\* | +0.47\* | +0.55\* | +0.29\* | +0.30\* |
| **COMPASS v2 + 调用形式 + 完整前缀（最终）** | **+0.16\*** | **+0.40\*** | **+0.45 [+0.22, +0.68]\*** | +0.20\* | +0.27\* |

配对差（同一 boundary，POST_COMPASS − POST_OpenClaw；负 = 更少额外错误/重取）：

| | Δ@k1 | Δ@k2 | Δ@k3 | Δ@k4 | Δ@k5 |
|---|---|---|---|---|---|
| 最终 COMPASS | **−0.15\*** | −0.13 | −0.16 | **−0.23\*** | **−0.33 [−0.56, −0.09]\*** |

消融（同 cohort，2 samples/arm）：见 `reports/phase2_L.md`（`report_phase2.py` 自动追加；跑完后在本文件 §2.1 补表）。

## 3. Phase 3 子集 — end-to-end（`reports/episodes_final_dev30.md`，2 runs）

| 方法 | Acc run1 / run2 | Pass² | Pass@2 | mean score | 步数 | 压缩次数/ep | 压缩器 tokens/ep |
|---|---|---|---|---|---|---|---|
| Full context | 66.7 / 70.0 | 50.0 | 86.7 | 0.90 | 17.8 | 0 | 0 |
| OpenClaw | 43.3 / 40.0 | 23.3 | 60.0 | 0.82 | 21.1 | 2.1 | 13.0k |
| **COMPASS v2** | **43.3 / 56.7** | **33.3** | **66.7** | **0.84** | 18.3 | 1.9 | **4.7k** |

注：30 题 × 2 runs 的置信区间很宽（单题在不同 run 间 0.0↔1.0 很常见）；full-context 与压缩方法的差距主要在"完成 0.75–0.89 但漏一个要求"的题上。

## 4. 每一项方法改进对应的证据（验证器拆解 → 修复 → 复测）

| # | 观察到的失败 | 图层面的修复 | 效果 |
|---|---|---|---|
| 1 | POST 重读 API 文档 ×59、参数名写错 | `api_spec` 节点：把 `show_api_doc` 观测解析为精确签名并渲染 | smoke cohort dU@k5 +1.00 → +0.05 |
| 2 | LLM 把猜错 API 名泛化成"该 app 没有 X 能力"，agent 信任后放弃 | 否定性结论需该 app **完整未截断 API 列表**作证据；截断列表正则恢复并标注 | fd1f8fa_*、8749218_2、325d6ec_* 由放弃变为继续 |
| 3 | `show_account_passwords` 重取 ×30、`show_profile` ×10 | 打印未赋值的结果、字面量赋值、≤900 字符小观测整体保留 | — |
| 4 | agent 重复创建 4 次归档歌单（done 节点没带 playlist_id） | done 意图渲染其证据步产出的变量与值 | 30 题 Acc 33.3 → 53.3（单 run） |
| 5 | 首步参数错误 34 次（PRE 8）：`login(email=)`、错 app 的 token、猜 API 名 | 记录每次调用的**参数形式**（成功/失败+错误），token 来源沿数据流传播 | dU@k5 +0.69 → +0.55 |
| 6 | 共享 cohort 后期 boundary 只能从 OpenClaw 摘要重建 | 有状态压缩器在验证器中拿完整前缀 | dU@k5 +0.55 → +0.45，配对差显著 |

## 5. 协议与工程修正（影响所有方法）
- Trace 的 4000 字符观测截断砍掉 Spotify 91 个 API 的一半（Amazon 6.8k、Splitwise 7.5k 字符同理），full-context 也因此放弃任务；改为 10 000（`COMPASS_OBS_LIMIT`）。4000 协议下的旧结果保留在 `artifacts/dev41*`、`artifacts/verify_dev`（COMPASS 与 OpenClaw 无显著差异，refetch 由截断伪影主导）。
- Windows 下 agent 代码含 emoji → AppWorld 日志 gbk 崩溃（`21abae1_1` 在早期所有 run 的 full=0 皆为此）；`PYTHONUTF8=1`。
- Kimi-k2.7-code 经 Ollama Cloud：裸代码、无 CoT，temp 0 时重复同一调用（0/3）；temp 1.0 2/4；gpt-4.1 3/4 → agent 选 gpt-4.1。

## 6. 未完成 / 下一步
- 全量 168 题 ×2 runs（约 $450–500）：用户暂缓。
- 消融表补全后更新本文件与论文 §4。
- 剩余差距（COMPASS +0.45 vs PRE 0）：仍以 blocked 为主（参数/凭证错误、猜 API 名），候选方向：把"失败调用形式"与 spec 参数约束直接合并进 NEXT 项；对不存在的参数（如 `keywords=`）给出否定证据。
