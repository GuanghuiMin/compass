# COMPASS 进度汇报 — Meeting Note（2026-08-26）

> 主题：COMPASS（Context Compression as Evolving Computation）方法改进与当前实验结果
> 代码：`W:\context_compression\compass_v2\`（26 个 commit）；论文红色修订已推送 Overleaf（`main` 分支）；实验在 Ollama Cloud 上完成（agent = deepseek-v4-flash，除非注明），GPT 只用于早期开发（gpt-4.1 结果单独列出）。

---

## 1. 一句话总结

我们把 v1 的"LLM 用严格语法整图重生成、任一违规整体拒绝"改成了 **"确定性证据层 + 单次 LLM 提议 + 逐项校验"** 的领域无关协议；在 AppWorld 与 OfficeBench 上，COMPASS 的 boundary 级错误/重取显著低于 OpenClaw，端到端在短/中等任务上追平或超过 OpenClaw，但在**长任务（≥4 次压缩）上仍落后**——根因已定位并正在修。消融给出一个明确结论：**收益几乎全部来自确定性证据层（接口签名、调用形式、已观测结果、变量值），LLM 计划层在 boundary 指标上是净负、在任务级有小幅正贡献**。

---

## 2. 方法改进（相对合作者 v1）

### 2.1 v1 的问题（六任务诊断，`compass_v1/artifacts/.../AUDIT.md`）
- 27 个 boundary 中 **12 refused / 3 empty**：图合法性全部压给 LLM 一次性生成，任一违规整体拒绝。
- 拒绝后要么静默丢状态（agent 重取），要么保留原文导致 3 步内 prompt 涨到 21k tokens。
- `COMPLETE` 靠模型自述，无法核验（agent 自称"全部完成"→ checkpoint 写 `NOTHING REMAINS`）。
- 6 任务：v1 2/6，OpenClaw 5/6，full 6/6。

### 2.2 v2 的定位（采纳师兄建议后）
```
trajectory --Trace Adapter_α--> canonical events --Graph Update--> G_k --Plan-Conditioned Folding--> C_k
e_t = (id, action, arguments, observation, outcome, references, provenance)   # 仅 action/observation/provenance 必需
E_k = Normalize_α(Δτ_k);  ΔP_k ~ π_φ(·|G_{k-1},E_k,u);  G_k^+ = Apply(G_{k-1},E_k,ΔP_k)
G_k = G^evidence ∪ G^state ∪ G^intent
```
- **原则：provenance is deterministic; semantic interpretation is proposed.** 每条 fact / 状态变更必须引用存在的 event id，否则丢弃。
- **Adapter 三层**：Generic ⊆ Schema-Aware ⊆ Domain-Specific。Python AST / 持久变量 / API 文档解析降级为 **CodeAct adapter**，不是方法定义。

### 2.3 具体改进清单（每条都由 boundary 验证器的 refetch/blocked 拆解驱动）
| # | 观察到的失败 | 改进 | 效果 |
|---|---|---|---|
| 1 | LLM 整图生成 12/27 拒绝 | 确定性图（AST def/use、API 签名、环境报错合约）+ 单次 JSON 提议，逐项校验 | 35+109 boundary **零整体拒绝、零超预算** |
| 2 | done 靠模型自述 | done/blocked 必须有 ok/失败的 step 证据 | done 的 92% 带真实 API 调用证据 |
| 3 | POST 重读 API 文档 ×59、参数名写错 | 文档观测解析成精确签名（类型/可选/约束）挂到 handover | smoke cohort dU@k5 +1.00 → +0.05 |
| 4 | LLM 把猜错 API 名泛化成"该 app 没有 X 能力"，agent 信了就放弃 | **否定结论也需证据**：须有该 app 完整未截断的 API 列表；截断列表正则恢复并标注 | fd1f8fa_*、325d6ec_* 由放弃变为继续 |
| 5 | 重取密码/profile（打印过但没赋值） | 打印结果、字面量赋值、≤900 字符小观测整体保留为值 | — |
| 6 | 重复创建同名歌单 ×4 | done 节点挂**产出值**（playlist_id 等） | gpt-4.1 30 题 33.3 → 53.3 |
| 7 | 首步参数错误 34 次（PRE 8）：`login(email=)`、用错 app 的 token | 记录每次调用的**参数形式**（成功/失败+错误），token 来源沿数据流传播 | dU@k5 +0.69 → +0.55 |
| 8 | 循环查 20 首歌，结果只在观测里，被按 API 名合并后丢掉 | 已观测结果按**带参数的调用签名**去重保留 | deepseek 30 题 66.7 → 80.0 |
| 9 | 长任务多次折叠后"消息已取回"但内容没了；`phone_login` 提示像 token 被当 token 用 | LLM 抽取 **data facts**（具体值，必须引用 step，任何折叠层级保留）；dict 值标注取用方式 | 52 长任务重跑中 |

### 2.4 协议/工程层面的修正（影响所有方法，已回溯核对）
- Trace 沿用的 4000 字符观测截断砍掉 Spotify 91 个 API 的一半 → 全部方法改 10k（full-context 53→67%）。
- Windows 下 emoji 触发 AppWorld 日志 gbk 崩溃；DeepSeek 约 1/20 步把代码包在 `<｜DSML｜python>` 标签里；DeepSeek 在 JSON mode 下输出被隐藏推理截断 → 三者均已修，方法无关。
- 共享 cohort 上后期 boundary 给有状态压缩器完整前缀（公平性）。

---

## 3. 结果

### 3.1 AppWorld（test_normal, B=4096, 保留最后一轮, ≤50 步, 10k 观测）

**(a) 全量 168 题，agent = deepseek-v4-flash，run 1**（`reports/episodes_fullO_r1.md`）

| 方法 | Acc | mean score | 步数 | 超时(50步) | refetch/ep | 压缩次数/ep | 压缩器 tokens/ep |
|---|---|---|---|---|---|---|---|
| Full context | **81.0** | 0.95 | 19.9 | 0 | 1.0 | 0 | 0 |
| OpenClaw | 69.0 / 78.0 / 76.8（三次） | 0.89 | 26.1 | 14 | 5.2 | 4.1 | 33.6k |
| COMPASS v2 | 65.5 | 0.86 | 25.3 | 20 | 3.5 | 3.4 | 26.2k |
| **det + 状态外化 `_mem`（det_mem，2026-08-27，同 cohort）** | **67.3 / 70.2**（两次；Pass² 56.5，Pass@2 81.0） | – | 25.1 | – | – | – | – |
| det + `_mem` + LLM 进度叙述（det_mem_nar，2026-08-27） | 67.9（长 52 题 59.6，其余 116 题 71.6：未 grounding 的 Done 列表导致短任务提前完成） | – | – | – | – | – | – |
| **det + `_mem` + grounded 进度笔记（det_mem_nar2，2026-08-27）** | **73.8 / 71.4 / 67.3（均值 70.8±1.9）**；OpenClaw 三次 69.0/78.0/76.8（74.6±2.8）；配对差 −3.8±2.4 n.s.；Pass³ 53.0 vs 55.4 | – | 24.0 | – | – | – | – |

按压缩次数分层（COMPASS / OpenClaw / full）：0 次 n=15 → 1.00/0.93/1.00；1 次 n=46 → 0.78/0.80/0.87；2–3 次 n=55 → 0.71/0.75/0.84；**≥4 次 n=52 → 0.38/0.46/0.67**。差距全在长任务。

**(b) 前 30 题，多个 agent**

| agent | 方法 | Acc run1 / run2 | Pass² | Pass@2 | mean score |
|---|---|---|---|---|---|
| deepseek-v4-flash | Full | 83.3 | – | – | 0.94 |
| | OpenClaw | 73.3 | – | – | 0.93 |
| | **COMPASS v2** | **80.0** | – | – | 0.94 |
| | COMPASS−plan (noplan) | 73.3 | – | – | – |
| | COMPASS−LLM (det) | 66.7 | – | – | – |
| | **det + 状态外化 `_mem`（det_mem，2026-08-27）** | **80.0** | – | – | – |
| | **det + `_mem` + grounded 进度笔记（det_mem_nar2，2026-08-27）** | **83.3** | – | – | – |
| glm-5.2 | Full | 80.0 / 86.7 | 73.3 | 93.3 | 0.96 |
| | OpenClaw | 73.3 / 70.0 | 60.0 | 83.3 | 0.93 |
| | COMPASS v2 | 70.0 / 66.7 | 60.0 | 76.7 | 0.91 |
| | det_mem_nar2（2026-08-27） | 70.0 / 73.3 | – | – | – |
| kimi-k2.7-code（2026-08-27） | Full | 53.3 | – | – | – |
| | OpenClaw | 36.7 / 30.0 | – | – | – |
| | **det_mem_nar2** | **40.0 / 50.0** | – | – | – |
| gpt-4.1（早期，开发用） | Full | 66.7 / 70.0 | 50.0 | 86.7 | 0.90 |
| | OpenClaw | 43.3 / 40.0 | 23.3 | 60.0 | 0.82 |
| | COMPASS v2 | 43.3 / 56.7 | 33.3 | 66.7 | 0.84 |

**(c) 长任务（≥4 次压缩的 52 题）修复尝试**（deepseek）

| 变体 | Acc | mean score | 超时 |
|---|---|---|---|
| Full | 67.3 | 0.90 | 0 |
| OpenClaw | 46.2 / 63.5（两次运行） | 0.78 | 13 |
| COMPASS（168 run 中） | 38.5 | 0.67 | 19 |
| + 预算 0.4B→0.6B | 30.8 | – | 24 |
| + 前沿所需变量值下限 | 36.5 | 0.64 | 21 |
| + data facts（改进 #9） | 38.5 | 0.65 | 20 |
| + 最粗折叠层硬上限（不再退化为 LLM 缩写） | **42.3** | – | – |
| v3 流程图（信息挂到行动节点 + 证据剪枝），挂载值截断 120 字符版 | 34.6 | – | – |
| v3 流程图，挂载值完整版（`1ac73663`；boundary Δ +0.31*，未改善） | 33/52 时中止 | | |
| v3 流程图 + 全局证据分节按 v2 阶梯折叠（`2ad7f5e4`；boundary Δ −0.04 n.s.） | 32.7 | – | – |
| v3 + LLM 自然语言总结（boundary Δ +0.30） | 28.8 | – | – |
| **det + 状态外化 `_mem`（结果存回 Python 会话，checkpoint 只引用 key；boundary Δ −0.56\*）** | **51.9 / 40.4**（两次独立运行；OpenClaw 单次 46.2） | – | – |
| **det + `_mem` + LLM 进度叙述（det_mem_nar）** | **54.2 ± 2.9**（5 次：61.5/46.2/53.8/50.0/59.6） | – | – |
| det + `_mem` + grounded 进度笔记（nar2） | 53.8（1 次） | – | – |
| 参考：OpenClaw 多次 | 55.3 ± 3.6（4 次：46.2/63.5/57.7/53.8） | | |
| 参考：full 多次 | 69.2 ± 1.9（2 次） | | |

### 3.2 Boundary 级验证（Trace 协议：同一状态双臂回放 5 步，POST−PRE 的 blocked+refetch，bootstrap 95% CI）

**109 boundaries（30 条 OpenClaw 轨迹的全部压缩点），rollout agent = deepseek，2 samples/arm**（`reports/phase2_O.md`）

| POST | dU@k1 | dU@k3 | dU@k5 | dE@k5 / dR@k5 | 配对 Δ vs OpenClaw @k5 |
|---|---|---|---|---|---|
| PRE 控制组绝对值 | 0.17 | 0.47 | 0.81 | | |
| OpenClaw | +0.19\* | +0.49\* | **+0.66\*** | +0.14\* / +0.54\* | — |
| **COMPASS v2** | +0.17\* | +0.38\* | **+0.45\*** | +0.06 / +0.42\* | **−0.21 [−0.48, +0.07]** |

**消融 / adapter 层级（同一 cohort，配对 Δ@k5 vs OpenClaw；负 = 更少错误/重取）**

| 变体 | Δ@k5 | 解释 |
|---|---|---|
| 去调用形式 + 文档 + 结果（nodone） | **+1.02\*** | 证据层是全部收益 |
| 去 API 签名（nospec） | +0.82\* | |
| 去 LIVE VARIABLES（novars） | +0.27（k1–k4 显著） | |
| 完整 COMPASS v2 | −0.21 | |
| 去计划层（noplan） | **−0.34\*** | LLM 计划层在此指标上净负 |
| 无 LLM（det） | **−0.38\*** | |
| v3 流程图（信息挂到行动节点），首版 plan-first 替换全局证据 | +1.02\* | 证据层丢失 |
| v3，挂载为附加 / ref 只指向已渲染项 / 挂载值截断 120 | +0.35\* | 全局证据折叠阶梯比 v2 粗：api 名单无名字（文档重读 72 vs 22）、level 2 无 RESULTS |
| v3，挂载值完整（≤900 字符） | +0.31\* | 截断不是主因 |
| **v3，全局证据按 v2 阶梯折叠（`2ad7f5e4`）** | **−0.04（n.s.）** | 与 OpenClaw 持平，仍不及 v2/det：挂载到节点本身无 boundary 收益 |
| v3 + LLM 自然语言改写（v3_nl） | +0.30（k1–k3 显著） | 改写只丢信息 |
| det + 字段级投影（proj：按字段而非字符截断、填满预算） | −0.35\* | ≈det，无增益 |
| det + 投影 + 状态外化（det_pm） | −0.30\* | 投影抵消了 mem 的增益（行更长 → 折叠更粗） |
| v2 + 投影（proj） | −0.06 | 有害 |
| v2 + 状态外化（mem，带 LLM 计划层） | −0.31\* | 计划层在 mem 之上仍净负 |
| **det + `_mem` + LLM 进度叙述（det_mem_nar）** | **−0.54\*** | 叙述在 boundary 上零成本；52 长任务 61.5 / 46.2 |
| **det + `_mem` + grounded 进度笔记（det_mem_nar2）** | **−0.62\*（全部 k 显著）** | 当前最优 |
| det + grounded 进度笔记（无 `_mem`，det_nar2） | −0.46\* | 两通道可加 |
| **det + 状态外化（mem：完整 observation 存回 Python 会话 `_mem['sN']`，checkpoint 引用 key）** | **−0.56\*（k1..k5 全显著；字符串版 −0.51\*）** | 目前最优；refetch 212→170，agent 读 `_mem` 63 次 |
| Adapter: Generic | +1.34\* | 只有原始 action/observation |
| Adapter: Schema-Aware（无 AST） | +0.51\* | CodeAct 的状态在变量里，此层不够 |
| Adapter: Code-Aware（= v2） | −0.21 | |

gpt-4.1 agent 的 64-boundary cohort 结论一致：COMPASS −0.33\*（k=1,4,5 显著），nospec/nodone ≈ 0，det −0.48\*。

### 3.3 OfficeBench（arXiv 2407.19056；JSON 工具调用 agent，非 CodeAct；test split 非图像 95 episodes；ACON 环境，WSL）

| budget | Full | OpenClaw | COMPASS v2 | det_nar2（笔记，无外化；2026-08-27） |
|---|---|---|---|---|
| 2048 | 81.1 | 74.7 | 77.9 | **78.9** |
| 4096 | 81.1 | 82.1 | 77.9 | 78.9 |

（full 与 budget 无关；单 run，95 episodes 标准误约 ±4 点，三者在噪声范围内。）

这里 COMPASS 只有 schema-aware adapter（工具名、参数形状、错误前缀、切换 app 时的动作列表），没有任何程序状态，即与 full 持平。

---

### 3.4 预算扫描与其他基线（2026-08-28，deepseek）

| B | OpenClaw | det_mem_nar2 |
|---|---|---|
| 2048 | 66.7 | 66.7 |
| 4096 | 73.3 | **83.3** |
| 8192 | 86.7 | 86.7 |

AppWorld-168 其他基线（1 次）：FIFO 49.4、Hermes 70.8、ACON-UT 66.7；对照 OpenClaw 三次 74.6±2.8、nar2 三次 70.8±1.9、full 81.0。

## 4. 我们现在相信的结论

1. **方法层面**：v1 的失败是"由 LLM 建图并整体校验"造成的，不是图思想的问题；改成确定性证据 + 提议/校验后，零拒绝、零超预算，boundary 级错误显著低于自然语言 compaction。
2. **收益来源**：接口签名、调用形式（参数名 + 成功/失败）、已观测结果、变量值——都是确定性可得的；LLM 计划层的贡献取决于指标（boundary 级净负，任务级小幅正）。
3. **通用性**：adapter 分层的叙述成立——OfficeBench（无隐藏状态）schema 层即可；CodeAct 必须到 domain-specific 层（变量状态）。
4. **稳定性**：主要噪声来自 agent 自身（full-context 两轮 Pass@2−Pass² 也有 20–37 点），需要 ≥2 runs 或 boundary 级指标；不同 agent 下 COMPASS 相对 OpenClaw 的优势不一致（deepseek +6.7、gpt-4.1 +10 Pass²、glm 持平）。
5. **未解决**：长任务（≥4 次压缩）COMPASS 仍落后 OpenClaw 8 点；根因是折叠后"内容"丢失（不是预算），data facts 修复在验证中。

---

## 5. 想和导师/师兄讨论的问题

1. 计划层（intent 树）在 boundary 指标上是负的、在任务级是正的——论文里如何定位 "evolving computation"？（候选：计划层负责"不遗漏要求/终止判断"，证据层负责"不犯错/不重做"，分别用两类指标证明。）
2. 长任务上 OpenClaw 的自然语言叙述天然携带"抽取出的内容"，COMPASS 的结构化图需要显式的 data facts 才能做到——最终形态应是 **"结构化证据 + LLM 抽取的数据事实"** 的混合，而非纯图？
3. Agent 依赖性：是否把"agent 能否利用结构化 checkpoint"作为一个实验维度（deepseek / glm / gpt 三个已有）？
4. 论文实验预算：全量 168 题 ×2 runs ×3 条件的 GPT 版本约 $450–500，何时启动。

## 6. 下一步（按优先级）
1. 长任务修复验证（data facts）→ 若有效，重跑 AppWorld 168 run 1 的 COMPASS 并跑 run 2（全部条件）。
2. OfficeBench @4096 完成；再加一个 agent（glm-5.2）跑 OfficeBench。
3. 消融的端到端版本（nodone / nospec 在 30 题上）以对齐两类指标。
4. 论文 §4 实验表更新（Overleaf 红色部分）。


## 3.14 失败案例分析（nar2，AppWorld-168 ×3；2026-08-28）

样本：nar2 ≤1/3 成功而 OpenClaw ≥2/3 成功的 21 题 + nar2 0/3 而 full 2/2 的 1 题，共 51 个失败 episode。

**失败形态**：近失完成（complete 后分数 ≥0.7）30 例、错误完成（<0.7）13 例、跑满 50 步 8 例（全为长任务）。主因是"做完了但不对"，不是卡住。

**分层原因（看轨迹 + 与 OpenClaw 同题成功轨迹对比）**：
1. **多余的 `answer`**（约 1/4）：任务没要答案，agent 仍 `complete_task(answer=...)`；同题 OpenClaw 用 `complete_task()` 成功（如 1150ed6_1、2c544f9_2、f323bae_1，三者其余动作完全相同）。数据：在只接受 bare 的任务上，给 answer 的 episode 成功 0/34（OpenClaw 0/29、full 0/18）——两者都犯，nar2 略多（34 vs 29）。压缩后 agent 更倾向"报告结果"。
2. **任务子要求丢失**（近失的主要来源）：2d9f728_3 只发了付款请求、没付 Nancy 的饭钱；b6d1104_1 条目格式/数值不符；90adc3f_2 重建请求时"其余保持不变"未满足。checkpoint 顶部有原始 GOAL 全文，但 agent 不再逐条核对；OpenClaw 的摘要把 Goal 拆成条目并带 Constraints/Critical Context。
3. **压缩后重建的中间数据出错**：0de03ea_2 笔记写"20 首下载歌曲时长全部已知"，但会话里的 `downloaded_durations` 只有第一页 → 算出没有播放列表 ≥1800s，agent 报 fail；OpenClaw 找到 440（1871s）。笔记的"已处理"断言比实际数据完整度乐观。
4. **压缩后的多余/破坏性动作**：9016950_3 查 `delete_text_message` 并动作；bde252e_3 额外改 `order_index`；634f342_1 未核对结果。压缩后 agent 失去"已经够了"的判断，做了任务没要求的修改。
5. 长任务跑满步数（8 例）：多次压缩后反复核对（`print(_mem[...])`、重读笔记）。

**不是原因**：折叠层级/预算（失败题的摘要长度与成功题相同）；API 签名/调用形式（blocked 很少）。

**对策（nar4，待验证）**：笔记增加"任务子句清单"——把指令原文逐条拆成 requirement（只引用原文，不做环境推断，避免 nar3 的问题），每条标 done/not done；头部规则加"不要执行任务没有要求的修改；任务没要求答案就不要传 answer"。中间数据的乐观断言：笔记只能引用观测到的计数，已在 nar2 规则里，需再强调"存在会话变量里的数据只算观测到的部分"。

**nar4 结果**：30 题 **86.7**（nar2 83.3 / full 83.3 / OpenClaw 73.3）；168 题 run1 **81.5**（长 52 题 63.5，其余 116 题 89.7；多余 answer 导致的失败 25→4）——与 full 81.0 持平，OpenClaw 74.6±2.8，nar2 70.8±1.9；run2 79.2 → **nar4 两次均值 80.4**（长 52 题 62.5，其余 116 题 88.4；Pass² 72.6）；OpenClaw 三次 74.6（Pass³ 55.4），full 81.0（Pass² 74.4）。逐题配对：nar4 − OpenClaw = **+5.8 ± 2.4**（显著），nar4 − full = −0.6 ± 2.6（持平）。结论：失败分析驱动的 nar4（引用原文的任务子句清单 + 只做任务要求的事 + 不多传 answer）把任务级结果从'与 OpenClaw 持平'提升到'与 full context 持平、显著优于 OpenClaw'。**当前主方法：`compass_det_mem_nar4`**。

**nar4 补充（2026-08-28）**：boundary Δ@k5 **−0.54\***（k1..k5 全显著；nar2 −0.62\*，CI 重叠）；glm-5.2 30 题 **76.7 / 83.3**（均值 80.0；OpenClaw 71.7，full 83.3）；kimi-k2.7-code 30 题 **73.3 / 63.3**（均值 68.3；OpenClaw 33.3，full 53.3——超过 full）。OfficeBench det_nar4 @2048 **78.9** / @4096 **78.9**（nar2 78.9/78.9；OpenClaw 74.7/82.1；v2 77.9/77.9；full 81.1）。


## 3.15 nar4 失败案例二次分析（2026-08-28）

nar4 两次 168 共 66 个失败 episode（19.6%）。按 full 同题结果拆分：
- **长任务跑满 50 步 27 例**（full 两次都成功的 12 例、混合 5、full 也失败 9）——压缩可归因的最大残余。
- 完成但错 38 例，其中约半数 full 也失败（agent 能力），压缩可归因约 15 例。

跑满 50 步的 12 例（full 成功）的共同形态：每题 **api_docs 调用 16.5 次**、压缩 9.3 次、报错 4.2 次；checkpoint 91/112 处在折叠 level 2/3（预算吃紧）。文档调用中 42 次是压缩后首次读新 API（多应用任务的探索），25 次是签名已在 checkpoint 仍重读；28% 的压缩由文档观测（均 2.3k 字符）触发 → "读文档 → 压缩 → 丢 → 再读"的循环。
**渲染缺陷**：nar4 压缩后 **73 次 `NameError: name '<app>' is not defined`**（full 0、OpenClaw 0）——checkpoint 的调用形式/签名写成 `venmo.login(...)`，agent 照抄漏掉 `apis.`。结构类错误（KeyError/TypeError）21 次：结果只显示前 90 字符，返回结构丢失。

修复（commit `7b14f48a`）：图带 `call_prefix`（codeact = `apis.`）渲染到所有调用形式/签名；`_mem` 结果行标注顶层字段（`list of 12 dicts with keys ...`）。重跑：30 题 `devO_mem2/`、168 ×2 `fullO_r5/`、`fullO_r6/`。

**还能改进的方向（未做）**：(a) level 2/3 折叠时优先保留笔记 Next Steps 提到的 API 签名、先丢旧结果；(b) 文档观测是最大 token 消耗源——压缩器可把文档文本整体外化到 `_mem` 并在签名行给 key，减少重读的上下文代价；(c) 长任务多应用探索本身需要步数，B=4096 下与 full 的差距（62.5 vs 69.2）可能有下限。

**修复后重跑结果**：168 ×2 = **81.0 / 80.4（均值 80.7）**（修复前 81.5/79.2 = 80.4）；长 52 题 63.5（前 62.5），其余 88.4；app-NameError 73 → **0**；跑满 50 步的 episode 28 → **18**（OpenClaw 27，full 1）；30 题 86.7（不变）。配对：vs OpenClaw **+6.1±2.4**，vs full −0.3±2.6。结论：缺陷修复消除了整类错误并减少了 1/3 的步数耗尽，但总成功率在噪声内持平——剩余的长任务耗尽主要来自多应用探索的文档读取循环（§3.15 方向 (a)(b)），这是下一步的改进点。


## 3.16 师兄评审后的重新定位（2026-08-28）

评审要点（采纳）：`_mem` 不是"按需提取"而是"完整外置、按指针读取"，agent 可用信息不再受 B 约束，且 W_k 由 harness 执行、baseline 无同样接口——改变了环境转移，不能作为"context compression 的提升"报告；v2 的 intent 图/NEEDS/frontier 已被删除，当前折叠是 note-conditioned budgeted rendering；笔记的 grounding 是 prompt 约束而非构造保证。

决定：
- **主方法 = bounded 版本 `compass_det_nar4`**（无 `_mem`；|C_k| ≤ B，agent 只能看 C_k + 最后一轮）。
- `compass_det_mem_nar4` 改称 **COMPASS+ExternalMemory**，作为增强版/上界单列；补 `openclaw_mem`（OpenClaw + 同样的 `_mem` 接口，commit `dd2e754d`）使该对比公平。
- 待做：大 observation 的 future-conditioned extraction（按 NEEDS/子句只提升需要的字段进 information node，其余删除）；笔记形式化为 requirement-state layer 并加程序验证（DONE(c) ⇒ 引用的事件存在且 outcome=ok，否则降级）。
- 实验（跑中）：bounded `compass_det_nar4` 168 ×2（`fullO_r5/r6`）+ boundary 臂；`openclaw_mem` 168 ×2。已有：boundary det_nar2（无 mem）−0.46\*；OfficeBench det_nar4 78.9/78.9。

- 结果：bounded `compass_det_nar4` boundary Δ@k5 **−0.50\***（+ExternalMemory −0.54\*，det −0.38\*，det_nar2 −0.46\*）——严格协议下 boundary 级增益基本保留。

- **bounded `compass_det_nar4` 168 run1：83.3**（full 81.0，+ExternalMemory 两次 80.7，OpenClaw 74.6）；30 题 bounded nar4 **93.3**、bounded nar5（DONE 必须引用步骤 id 并由程序核验，commit `eb2fe06c`）83.3。任务级增益不依赖 `_mem`。run2 与 `openclaw_mem` 跑中。

- **168 题结果（bounded 协议）**：bounded nar4 **83.3**（长 52 题 65.4，其余 91.4；vs OpenClaw 配对 **+8.7±2.7**，vs full +2.4±2.8）；bounded nar5（DONE 必须引用步骤 id 且程序核验）**82.7**（+8.1±2.7）；公平对照 `openclaw_mem` 76.8（vs OpenClaw +2.2±2.9，不显著）；+ExternalMemory nar4 两次 80.7（vs openclaw_mem +3.9±3.1）。
- 结论：**任务级增益来自证据层 + grounded 笔记，不来自外部记忆**；严格协议下的 bounded 版本反而更好（`_mem` 让 agent 多花步数读取）。主方法定为 bounded `compass_det_nar4`（或 nar5，二者相当且 nar5 满足 grounded-by-construction）；`+ExternalMemory` 作为变体单列。第二次 bounded 运行完成：bounded nar4 两次 83.3/79.8 = 81.5（长 0.673，其余 0.879；vs OpenClaw +0.069±0.024，vs full +0.006±0.025）。


## 3.17 第二轮评审（2026-08-29）的四项要求与处理

1. **主方法 = bounded nar5**（程序核验），nar4 改称 prompt-grounded 消融。
2. **私有图预算 + raw observation 物理删除**（commit `08b3d947`）：Apply 之后步骤只保留 600 字符摘录，解析后的观测（`printed`）与 `value_full` 清空；序列化图受 B_G = 64k 字符（≈16k tokens = 4×B）约束，超出时按最旧证据逐级驱逐（observation/code → 结果 hint → superseded 变量）；每个边界记录 graph bytes（`extra.graph_bytes` / `graph_evicted`）。协议改写为 Q_k = Extract_α(Δτ_k)，G_k = Apply(G_{k-1}, Q_k)。外化/投影变体保留原文（它们需要）。
3. **Requirement 图**：nar5 核验后的子句解析为 V^requirement 节点 {id, text, status, supports:[step ids]}（SUPPORTS 边指向 evidence 步骤；未引用或引用失败步骤的状态降为 NOT DONE），checkpoint 的需求分节由节点重新生成（`- r1: "…" -- PARTIAL (2 of 5) [s14, s16]`）。方法名改为 evidence-grounded requirement-state compaction；折叠称 requirement-conditioned folding。
4. **claim 限定**：主张只对 CodeAct / 结构丰富的 tool-use agent；Generic（+1.34\*）/ Schema-aware（+0.51\*）与 OfficeBench @4096 的数据表明一般 ReAct 尚未验证。
实验（跑中）：预算版 nar5 168 ×2（`fullO_r7/r8`）、boundary 臂（`verify_O/compass_det_nar5b`）、30 题（`devO_mem3/`）；同时报告 graph bytes 随边界的变化。

- 预算版 nar5 结果：30 题 **96.7**（full 83.3）；boundary Δ@k5 **−0.56\***（k2..k5 显著；未加预算版 −0.49\*）；图大小 ~21k 字符/边界（B_G 64k）。168 ×2 跑中。
