# COMPASS v3.2：evidence-grounded requirement-state compaction（2026-08-29，第二轮评审后）

> 给合作者的完整版。**主方法 = bounded + budgeted `compass_det_nar5`**（私有图有预算、raw observation 在 Apply 后删除、需求子句为图节点并程序核验）。主张范围：CodeAct / 结构丰富的 tool-use agent；一般 ReAct 未验证。第 0 节是对两轮评审意见的逐条回应；第 1–3 节是方法；第 4 节是全部实验结果；第 5 节是否定结果与局限；**第 7 节是按第三轮评审（executable frontier）实现并跑完的三臂消融及其结论**。所有实验：agent 与压缩模型均为开源模型（Ollama Cloud），除注明外为 deepseek-v4-flash；AppWorld test_normal，窗口 B=4096，保留最后一轮，50 步上限；OfficeBench 95 个非图像测试 episode。

---

## 0. 评审意见与回应

| 评审要点 | 回应 | 数据 |
|---|---|---|
| `_mem` 是完整外置、按指针读取；W_k 由 harness 执行、不计步、baseline 无同样接口 → 改变了环境转移，agent 可用信息不再受 B 约束 | **同意，已从主方法移出。** 主方法改为 bounded 版本：\|C_k\| ≤ B，agent 只能访问 C_k + 最后一轮，无环境侧通道。`_mem` 版本改称 COMPASS+ExternalMemory，作为变体单列，并补 `openclaw_mem`（OpenClaw + 同样的 `_mem` 接口）做公平对照 | bounded 81.5 ≥ +ExtMem 80.7；`openclaw_mem` 76.8 vs OpenClaw 74.6（+2.2±2.9，不显著）。**增益不来自外部记忆** |
| v2 的 intent 图 / NEEDS / frontier 被删除；折叠已是 note-conditioned budgeted rendering，不再是 plan-conditioned graph contraction | 同意这个描述，文档改用 **requirement-conditioned folding** 一词。intent 图不是"放弃"，是被配对实验否定：结构化 LLM 计划层在 boundary 级 +0.25（更差），任务级无增益 | 见 §4.7 |
| 笔记的 grounding 是 prompt 指令，不是构造保证 | 同意。nar5 实现程序核验：DONE/PARTIAL 必须引用步骤 id，压缩器检查 `DONE(c) ⇒ ∃e cited: e∈G_k ∧ outcome(e)=ok`，否则降级为 NOT DONE (unverified) | 未加预算 nar5 79.2、预算版 nar5 81.0 vs nar4 81.5，均在噪声内；boundary −0.56\*（预算版）vs −0.50\* |
| 大 observation 应做 future-conditioned extraction，只提升未来需要的字段 | 未做。bounded 版本目前对大 observation 只保留调用签名 + 90–160 字符摘录，其余删除；按 NEEDS 提升字段是下一步 | – |
| Kimi 上超过 full 可能说明外部 memory 更易用，而非压缩忠实 | 同意；该数字来自 +ExtMem 版本。bounded 版本在 glm/kimi 上待补 | – |
| （第二轮）私有图 G_k 没有预算；E_k 含完整 observation 传入 Apply，'其余删除'只是文字 | 已实现：Q_k = Extract_α(Δτ_k)，Apply 后步骤只留 600 字符摘录，解析观测与 value_full 清空；序列化图受 B_G = 64k 字符约束（≈16k tokens），超出按最旧证据驱逐；每边界记录 graph bytes | 914 个边界：均值 37.5k，p95 65k，最大 70k（元数据下限），99 个边界驱逐；性能不变（81.0） |
| （第二轮）nar4 不能叫严格 grounded；主方法应为 nar5 | 采纳：主方法 = nar5；nar4 改称 prompt-grounded 消融 | nar5 预算版 81.0 vs nar4 81.5，噪声内 |
| （第二轮）已不是 graph planning；应形式化为 requirement graph（V^requirement ∪ V^evidence ∪ V^information，SUPPORTS/NEEDED_BY） | 采纳：核验后的子句成为 requirement 节点 {id, text, status, supports}，SUPPORTS 边指向 evidence 步骤；checkpoint 需求分节由节点生成；方法名改为 evidence-grounded requirement-state compaction，折叠称 requirement-conditioned folding。NEEDED_BY 边未做 | 见 §1、§3 |
| （第二轮）一般 ReAct 证据不足，主 claim 应限定 CodeAct | 采纳：主张限定为 CodeAct / 结构丰富的 tool-use agent；Generic +1.34\*、Schema-aware +0.51\*、OfficeBench @4096 78.9 < 82.1 作为未验证的证据如实列出 | – |

---

## 1. 协议（bounded）

```
Q_k = Extract_α(Δτ_k)                  # adapter：轮次 → 有界事件 (id, action, arguments, obs excerpt ≤600 chars, outcome, references, provenance)；raw observation 随后销毁
G_k = Apply(G_{k-1}, Q_k), size(G_k) ≤ B_G  # 确定性图更新，无 LLM；B_G = 64k chars，超出按最旧证据驱逐
N_k = Verify_G(Note_φ(N_{k-1}, E_k, u)) # LLM 写 requirement-state 笔记；nar5 再由 G_k 程序核验
C_k = Render_B(N_k, Fold(G_k | N_k))    # |C_k| ≤ 0.4·B；笔记先占预算，证据分节逐级折叠
```

agent 上下文 = `<history_summary> C_k </history_summary>` + 保留的最后一轮原文。G_k 与 N_k 是压缩器私有状态（随 checkpoint 持久化、增量更新），agent 看不到。G_k = V^requirement ∪ V^evidence ∪ V^information：requirement 节点（指令子句 + 核验状态）通过 SUPPORTS 边指向 evidence 步骤；information 节点（变量、结果、签名）通过 PRODUCES/CONSUMES 与步骤相连。

Adapter 三层（Generic ⊆ Schema-aware ⊆ Domain-specific）：

| 字段 | Generic | Schema-aware | CodeAct（AppWorld） |
|---|---|---|---|
| arguments（调用形式） | – | schema 参数名 | AST 抽 `apis.app.api(参数名…)` |
| outcome | – | 环境错误字段 | traceback 合约 |
| references（产物） | – | 返回对象 | 变量 def/use 数据流 + 来源 API |
| 接口知识 | – | 工具描述 | `show_api_doc/descriptions` 解析成签名、返回字段、API 名单 |

OfficeBench（JSON 工具调用）用 Schema-aware 层 + 笔记（已跑 `compass_det_nar4`；nar5 版待跑）。

---

## 2. 确定性证据层 G_k

渲染到 C_k 的分节（按顺序）：
- **CALL FORMS THAT WORKED / FAILED**：`apis.venmo.create_transaction(access_token, receiver_email, amount)`；失败附错误行。（修复：此前缺 `apis.` 前缀，agent 照抄导致压缩后 73 次 `NameError: name 'venmo'`，修后 0。）
- **API DOCS ALREADY READ**：各应用 API 名单（截断时标注）+ 精确签名与返回字段。
- **LIVE VARIABLES**：仍绑定的变量、来源 API、短摘录；凭证永不折没。
- **RESULTS ALREADY OBSERVED**：按带参数的调用签名去重的结果摘录。

折叠阶梯 level 0→3：逐级缩短各分节；前沿所需值有下限。

---

## 3. Requirement-state 层 N_k（主方法 nar5）

每次压缩一次 LLM 调用（≤450 tokens），输入 = 上一笔记 + 新轮次 + 任务指令。LLM 的输出先经 `Verify_G` 核验，核验后的子句解析为 requirement 节点 {id, text, status, supports}，checkpoint 的需求分节由节点重新生成（不是复述 LLM 文本）：

```
PROGRESS NOTE (advisory -- check remaining items against the evidence below; do only what the
task asks -- no extra modifications; pass an answer to complete_task only if the task asks for one)

## Task requirements (quoted from the instruction) and verified status
- r1: "Send $20 to each of my coworkers via venmo" -- PARTIAL (3 of 5) [s14, s16, s18]
- r2: "with a note, 'For Lunch'" -- DONE [s14, s16, s18]
- r3: "Refill venmo balance if you need to" -- NOT DONE (unverified)      ← LLM 写了 DONE 但未引用成功步骤，被核验降级
## Handled so far        （带计数，只写观测确认的）
## Not yet done / unverified
## Next Steps
```

规则：子句只能引用指令原文（不做环境推断）；不重复值；不得宣称完成；头部两条行为规则。核验：DONE/PARTIAL 必须引用 `[sN]`，且每个 sN 必须存在于 G_k 且 outcome=ok，否则改写为 NOT DONE (unverified)——`DONE(c) ⇒ ∃e∈G_k: SUPPORTS(e,c) ∧ outcome(e)=ok` 由构造保证。nar4 = 同一笔记但无核验（prompt-grounded），作为消融保留。

设计来源（失败分析，nar2 输给 OpenClaw 的 51 例）：43 例"完成但错"——多余的 `complete_task(answer=…)`（该情形 0/34 成功）、丢任务子句、笔记对会话数据过度乐观、压缩后做多余修改。

---

## 4. 完整结果

### 4.1 AppWorld test_normal 168 题（deepseek-v4-flash）

| 方法 | 各次 Acc | 均值 | 长 52 题 / 其余 116 | Pass² | 逐题配对 |
|---|---|---|---|---|---|
| Full context | 81.0 / 81.0 | 81.0 | 69.2 / 86.2 | 74.4 | – |
| OpenClaw | 69.0 / 78.0 / 76.8 | 74.6 ± 2.8 | 59.0 / 81.6 | 55.4（P³） | – |
| **bounded + budgeted nar5（主方法）** | **80.4 / 81.5** | **81.0** | 68.3 / 86.6 | 73.2 | vs OpenClaw **+6.3±2.7**；vs full +0.0±2.9 |
| bounded nar4（prompt-grounded 消融） | 83.3 / 79.8 | 81.5 | 67.3 / 87.9 | 73.2 | vs OpenClaw +6.9±2.4；vs full +0.6±2.5 |
| bounded nar5，未加图预算 | 82.7 / 75.6 | 79.2 | 62.5 / 86.6 | 71.4 | vs OpenClaw +4.6±2.5；vs full −1.8±2.5 |
| openclaw_mem（公平对照） | 76.8 | 76.8 | 63.5 / 82.8 | – | vs OpenClaw +2.2±2.9 |
| +ExternalMemory nar4 | 81.0 / 80.4 | 80.7 | 63.5 / 88.4 | – | vs openclaw_mem +3.9±3.1 |
| +ExternalMemory nar2 | 73.8 / 71.4 / 67.3 | 70.8 ± 1.9 | 57.1 / 77.0 | 53.0（P³） | vs OpenClaw −3.8±2.4 |
| +ExternalMemory det（无笔记） | 67.3 / 70.2 | 68.8 | 42.3 / 80.6 | – | |
| Hermes | 70.8 | | | | |
| ACON-UT | 66.7 | | | | |
| FIFO | 49.4 | | | | |
| COMPASS v2（旧，结构化计划层） | 65.5 | | 38.5 / 77.6 | | |

同配置重复运行的波动可达 9 点（OpenClaw 69→78）；所有对比以多次均值 ± SE 与逐题配对为准。

### 4.2 52 长任务（≥4 次压缩）多次运行

| 方法 | 各次 | 均值 ± SE |
|---|---|---|
| Full | 67.3 / 71.2 | 69.2 ± 1.9 |
| OpenClaw | 46.2 / 63.5 / 57.7 / 53.8 | 55.3 ± 3.6 |
| budgeted nar5（主方法） | 65.4 / 71.2 | 68.3 |
| bounded nar4 | 65.4 / 69.2 | 67.3 |
| +ExtMem nar4 | 63.5 / 63.5 | 63.5 |
| +ExtMem nar（未 grounding 笔记） | 61.5 / 46.2 / 53.8 / 50.0 / 59.6 | 54.2 ± 2.9 |
| +ExtMem det | 51.9 / 40.4 / 44.2 | 45.5 ± 3.4 |
| v2 | 38.5 | |

### 4.3 Boundary 级验证（Trace 协议：109 边界，同状态双臂回放 5 步，POST−PRE 的 blocked+refetch，配对 Δ vs OpenClaw，负=更好，\*=95% CI 不含 0）

| 变体 | Δ@1 | Δ@3 | Δ@5 |
|---|---|---|---|
| **budgeted nar5（主方法）** | −0.07 | −0.32\* | **−0.56\*** |
| bounded nar4 | −0.07 | −0.29\* | −0.50\* |
| bounded nar5，未加图预算 | −0.06 | −0.26\* | −0.49\* |
| bounded nar2 | −0.02 | −0.25\* | −0.46\* |
| +ExtMem nar2 | −0.13\* | −0.40\* | −0.62\* |
| +ExtMem nar4 | −0.09\* | −0.32\* | −0.54\* |
| +ExtMem det | −0.15\* | −0.33\* | −0.56\* |
| det（纯确定性证据层） | −0.04 | −0.20\* | −0.38\* |
| COMPASS v2（证据 + 结构化计划层） | −0.01 | −0.11 | −0.21 |
| v2 去计划层 | | | −0.34\* |
| v2 去变量 / 去签名 / 去调用形式+结果 | | | +0.27 / +0.82\* / +1.02\* |
| Adapter Generic / Schema-aware | | | +1.34\* / +0.51\* |
| OpenClaw | 0 | 0 | 0 |

### 4.4 前 30 题，多个 agent

| agent | Full | OpenClaw | v2 | budgeted nar5 | bounded nar4 | +ExtMem nar4 |
|---|---|---|---|---|---|---|
| deepseek-v4-flash | 83.3 | 73.3 | 66.7–80.0 | **96.7** | 93.3 | 86.7 |
| glm-5.2（两次） | 80.0 / 86.7 | 73.3 / 70.0 | 70.0 / 66.7 | 待跑 | – | 76.7 / 83.3 |
| kimi-k2.7-code（两次） | 53.3 | 36.7 / 30.0 | – | 待跑 | – | 73.3 / 63.3 |
| gpt-4.1（早期，开发用） | 66.7 / 70.0 | 43.3 / 40.0 | 43.3 / 56.7 | – | – | – |

glm/kimi 上的 +ExtMem 数字包含外部记忆通道，bounded 版本待补。

### 4.5 预算扫描（30 题，OpenClaw / +ExtMem nar2）
B=2048：66.7 / 66.7；B=4096：73.3 / 83.3；B=8192：86.7 / 86.7（full 83.3）。

### 4.6 OfficeBench（Schema-aware 适配器 + 笔记，无外化；无 Python 会话）

| B | Full | OpenClaw | COMPASS v2 | det_nar2 | bounded nar4 |
|---|---|---|---|---|---|
| 2048 | 81.1 | 74.7 | 77.9 | 78.9 | **78.9** |
| 4096 | 81.1 | 82.1 | 77.9 | 78.9 | 78.9 |

### 4.7 组件增减（deepseek）

| 变化 | 效果 |
|---|---|
| 证据层（det） | boundary −0.38\*（OpenClaw 0）；168 题 +ExtMem det 68.8 |
| + 自由叙述笔记（nar） | 长任务追平 OpenClaw，但短任务 71.6（未 grounding 的 Done 列表 → 提前完成） |
| + grounding（nar2） | 168 题 70.8，30 题 83.3 |
| + 任务子句清单 + 行为规则（nar4） | 168 题 81.5（bounded），30 题 93.3 |
| + 状态程序核验（nar5） | 79.2 / boundary −0.49\*，与 nar4 噪声内相当 |
| + 私有图预算 + 原文删除 + 需求节点（预算版 nar5，主方法） | 168 题 81.0（= full），boundary −0.56\*，30 题 96.7；图大小均值 37.5k 字符 |
| `_mem` 外化 | boundary 更好（−0.54 vs −0.50），任务级更差（80.7 vs 81.5）；给 OpenClaw 同样接口 +2.2 n.s. |
| 结构化 LLM 计划层（v2） | boundary +0.25，任务级无增益 → 移除 |

---

## 5. 否定结果与局限

否定（均为配对实验）：v3 流程图（信息挂到计划节点，最好 −0.04 n.s.）、结构化 checkpoint 的自然语言改写（+0.30）、字段级投影（按"更多字段"实现，行变长折叠更粗）、结构化 LLM 计划层、未 grounding 的 Done 列表、让模型写"约束栏"（写入 "presumably…" 之类推断，30 题 70.0）。

局限：
- 长任务与 full 仍有小差距（68.3 vs 69.2），来自多应用探索反复读文档（每次 2.3k 字符）触发压缩的循环。
- 大 observation 的 future-conditioned 字段提取未做。
- 无会话环境在大预算下略低于 OpenClaw（OfficeBench @4096 78.9 vs 82.1）。
- 约半数残余失败在 full context 下也失败（agent 上限）。
- glm/kimi 上的 bounded 版本待补；GPT 上未跑（需确认）。

---

## 6. 可直接用于论文的方法定义（一段）

COMPASS 是一个 bounded、evidence-grounded 的 requirement-state compaction 协议。在每个压缩边界，adapter 把被吸收的轮次确定性地提取为有界事件 Q_k（调用形式、outcome、产物与来源、接口签名、≤600 字符的观测摘录），原始观测随即销毁；私有图 G_k = V^requirement ∪ V^evidence ∪ V^information 在 B_G 预算下增量更新。一次 LLM 调用把任务指令的每个子句写成 requirement 节点并给出状态，状态必须引用支持它的 evidence 步骤，压缩器核验引用存在且成功，否则降级——因此 DONE 是 G_k 上可验证的谓词。checkpoint C_k（≤0.4B）由需求节点、已处理/未完成清单与逐级折叠的证据分节渲染而成，agent 只见 C_k 与最后一轮。在 AppWorld（CodeAct，B=4096）上，该协议使成功率与 full context 持平（81.0 vs 81.0）并显著高于自然语言压缩 OpenClaw（+6.3±2.7），boundary 级重取/阻塞显著更少（−0.56\*）；主张限定于 CodeAct 及结构丰富的 tool-use agent。

代码：`W:\context_compression\compass_v2`（主方法 `compass_det_nar5`；消融 `compass_det_nar4`；变体 `compass_det_mem_nar4`、`openclaw_mem`；核验 `ground_note` / 需求节点 `parse_requirements`，`graph/compressor.py`；预算 `Graph.enforce_budget`，`graph/build.py`）。数据与日志：PLAN.md §3.10–3.17。


---

## 7. 第三轮评审：executable frontier（已实现并跑完，2026-08-31）

### 7.1 实现（`src/compass/graph/requirements.py`，变体 `compass_frontier*`）
- **一次性分解**：episode 首个边界用 decompose 调用把指令拆成 requirement 节点；每个节点必须逐字引用指令片段（引擎校验 span ⊆ instruction，否则丢弃），可带 `expect`（对象数量）与 `ordered`（组内顺序）。
- **只允许局部算子**（引擎校验，LLM 只做语义判断）：`REFINE`（仅顶层、未完成、未细化，2–4 个粗粒度子目标）、`UPDATE_STATUS`（证据步骤必须存在且执行成功；带 expect 的节点 `DONE` 需 count ≥ expect，否则降级 PARTIAL）、`PROPOSE_NEXT`、`DECLARE_NEED`。父节点状态由子节点 roll-up，不可直接设置。
- **可执行 frontier**：F_k = 未完成叶子且（ordered 组内）前置已完成；`NEEDED_BY` 由 DECLARE_NEED 的 {api, fields} 与 information 节点匹配。frontier 证据与其祖先证据、NEEDED_BY 信息**不被 B_G 驱逐**并在渲染中优先。
- **需求条件化提取**（arm c）：大 observation 按 Needs(F_k) 的字段做 JSON-path 提取，生成 information 节点后再删原文。
- **下界渲染**：状态只声明"已确认完成什么"（confirmed done / at least N of M done / open），杜绝滞后的 NOT_STARTED 与轨迹矛盾。**完成守卫**：凡 count < expect 的子句，checkpoint 末尾列出 `COMPLETION CHECK: r3.2 at 7 of 12 …`。

### 7.2 三臂消融结果（52 长任务，同 cohort；指标不止 accuracy）

| 臂 | acc | 跑满 50 步 | 提前完成 | refetch/题 | boundary Δ@k5 | 30 题 |
|---|---|---|---|---|---|---|
| (a) nar5（冻结基线，2 次） | **65.4 / 71.2** | 9 / 9 | 10 / 7 | 5.1 / 4.8 | **−0.56\*** | 83.3–96.7 |
| (b) nar5 + 需求 frontier | 61.5 | 11 | 9 | 6.3 | −0.50\* | 83.3 |
| (b') (b) + 完成守卫 | 59.6 | 12 | 8 | 6.7 | – | 83.3 |
| (b'') (b) + 晚启用门控（2 次压缩后才启用） | – | – | – | – | – | 86.7 |
| — 需求树**替代**笔记（第一版接线） | 55.8 | 19 | 3 | 7.9 | −0.31\* | 83.3 |
| — OpenClaw（4 次） | 55.3 ± 3.6 | – | – | – | 0 | 73.3 |
| **(c) (b) + needs-conditioned 提取** | **63.5** | 10 | 11 | 6.2 | −0.33\*（旧接线，待重跑） | 80.0（旧接线） |

### 7.3 结论
1. **frontier 修好了自己的病**：相对"树替代笔记"的第一版，步数耗尽 19→11、refetch 7.9→6.3、boundary −0.31→−0.50、长任务 55.8→61.5，**明显高于 OpenClaw**。说明 plan→保留 这条链条本身是有效的工程机制。
2. **但它没有超过 nar5**：长任务 61.5 vs 68.3，boundary −0.50 vs −0.56。原因是 nar5 的已核验笔记本身就带覆盖计数与"未完成清单"，frontier 想提供的信息大部分已经在里面；额外的两次 LLM 调用与树的预算占用抵消了收益。
3. **完成守卫无增益**（59.6 vs 61.5，一题之差）：agent 并非因为"不知道还差多少"而提前完成。
4. **短任务上是净成本**：30 题 2.1–2.5 次压缩，frontier 三次均 83.3；晚启用门控回到 86.7。（注意该子集方差 ±13 点，nar5 自身也跑出过 83.3 与 96.7。）
5. **arm (c) 是 frontier 里最强的一环**：按 Needs(F_k) 做 JSON-path 字段提取后长任务 63.5（vs (b) 61.5、nar5 65.4/71.2、OpenClaw 55.3），即**用 plan 决定从大 observation 里提升哪些字段**确有价值，只是尚不足以超过 nar5 的均值。
6. **尚未验证的关键变体**：目前 NEEDED_BY 只用于**保护与排序**，还没有用作**删除准则**（"frontier 不需要的证据就丢掉"）。这可能才是 plan-conditioned retention 与 nar5 的真正分界点，也是下一步最值得做的单一实验。
