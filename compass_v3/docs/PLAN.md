# COMPASS v2 — 研究与实验计划

> 目标：把 COMPASS（Context Compression as Evolving Computation）从"LLM 整图重生成"改造成
> **确定性骨架 + 局部 LLM 语义决策** 的 text-attributed action graph 方法，在 AppWorld 上用
> Trace 的 boundary 级验证器和 end-to-end 成功率两条线验证，产出可写进 ICLR27 论文 §4 的结果。
>
> 日期：2026-08-25。本文档是 `/goal` 的执行蓝本；每个 Phase 有明确的 Gate（通过条件），不通过则不进入下一阶段。

---

## 0. 现状诊断（为什么 v1 有问题）

材料：`ICLR27_COMPASS.pdf`、`compass_v1/`（分支 `compass-v1-refinement-interface`，已 clone 到本目录）、`trace_ref/`（nokia Trace，已 clone）。

### 0.1 v1 的实际做法
- 状态 = 一张"future computation + information"图（两种节点：Computation / Information；关系 REFINES / REQUIRES / PRODUCES / ORDERED_BEFORE）。
- 每个 compaction boundary：LLM 读 `previous graph + Δτ slice`，用严格 block 语法（`BEGIN_REVISION … END_REVISION`, `ADD/REPLACE/COMPLETE/NOW_AVAILABLE …`）输出一次 **local revision**；解析 → 全量校验（cycle、availability、interface、anchor…）→ **任一违规即整体拒绝**，旧图原样保留。
- 渲染（handover）是确定性的："REFINED PLAN OVERVIEW / ACTIVE WORK / LATER COMPUTATIONS"，只写未来，不写历史。

### 0.2 证据（`compass_v1/artifacts/diagnostic/six_task_v1/AUDIT.md`, `artifacts/causal/refusal_retention_v1/REPORT.md`）
| 现象 | 数据 | 根因 |
|---|---|---|
| 拒绝率极高 | 27 boundaries：12 accepted / **12 refused** / 3 empty；拒绝原因 11 种且无主导（cycle、overlapping regions、undeclared label、markdown fence…） | 把"图合法性"全部压给 LLM 一次性生成；语法+语义约束太多 |
| 拒绝 = 静默状态丢失 | 3d9a636_2：拒绝后 handover 冻结 10 步，agent 重新读同一份 API doc（refetch） | refused 的 slice 被丢弃，信息既没进图也没保留 |
| 修补版 v1r 反向爆炸 | 拒绝后保留 slice：3 步内 downstream prompt 9.8k→21.2k tokens，updater 三次空输出，任务终止 | "全有或全无"的更新算子没有降级路径 |
| 错误完成无法核验 | f3f60f0_2：agent 自称"所有 album 已 like"，图记录 `NOTHING REMAINS`，但 album 4/5 未处理 | `COMPLETE` 是 model-declared，图无观测证据 |
| 端到端 | 6 任务：v1 2/6，OpenClaw 5/6，full 6/6；3/6 撞 50 步上限 | 综合以上 |
| 成本 | 单次 revision 输出曾达 16384 tokens 上限；prompt 中 slice 16–36KB | 全图重生成 + 长 slice |

结论：**图的思想是对的（v1 渲染出的 handover 质量明显高于 OpenClaw checkpoint，见 boundary_003 样例），坏在"由谁建图、怎么建、失败怎么办"**。这正是用户指出的重点：建图方式、节点颗粒度、节点特征、信息如何挂到未来节点。

### 0.3 Trace 能给我们什么
- 一个不依赖参考摘要的 boundary 级验证器：同一 pre-compaction 状态回放两次（PRE=原始历史，POST=压缩后），量 **blocked**（环境报错合约）与 **refetch**（重复已执行 API 签名）的差值。可作为**便宜、稠密、可归因**的中间指标，不必每次跑 end-to-end。
- 冻结的 590-boundary AppWorld cohort（train+dev, W=4096）：只有 hash 和 rollout 结果，**没有原始历史文本**，不能直接拿来建图；但 `trace_cc.collect.runner` 的 `AgentAdapter` 协议可直接复用来采我们自己的 cohort。
- `trace_cc.core.sigs/canon`：API 签名规范化——直接用作 step 节点的确定性特征。
- 论文 Table 3 里 "TRACE 77.4"（Kimi-K2.7-Code, test_normal 168）是同事方法在同一协议下的数字，是我们的**对标线**（不是 baseline 表里的 58.3）。

### 0.4 可用的离线语料
- `compass_v1/inputs/diagnostic/packets/*.json`：197 个 decision points（5 个 episode），字段 `packet_id / goal / rules / prefix[{reasoning, code, observation}]`；82 条人工标注（`practice_answer_key.json`，五类：ordinary_progress / progressive_refinement / structural_revision / terminal_transition / indeterminate）。
- `compass_v1/artifacts/diagnostic/six_task_v1/full_context/*.jsonl`：6 条完整 full-context 轨迹（`trajectory_steps[{step_id, reasoning, code, observation}]`）。
- 这些足够做 Phase 1 的建图研究（不需要环境）。

---

## 1. 方法重设计：COMPASS v2

### 1.1 设计原则
1. **确定性优先，LLM 补语义。** 凡是能从 CodeAct 轨迹里用 AST/regex/签名匹配确定的结构（步骤、变量定义-使用、API 调用、成功/失败、产出值），一律确定性抽取，永远不会"拒绝"。LLM 只做局部、小粒度、可验证的语义判断（意图归并、信息描述、消费预测）。
2. **图永远合法。** 任何 LLM 输出都被当作**建议**；经过校验的部分合并进图，不合法的部分丢弃（不是整体拒绝）。图的合法性由构造保证，不由 LLM 保证。
3. **降级阶梯，永不爆炸。** 图渲染超预算 → 更粗折叠 → 仍超 → 退化为 OpenClaw 风格 checkpoint（保证不比 baseline 差）。任何时候 handover ≤ B。
4. **完成需有证据。** 节点 `done` 只能由观测（API 调用成功、返回值满足产出 schema）或用户消息判定；LLM 只能提议。
5. **信息按"未来谁用"存活。** Information 节点通过**消费边**挂到未来节点；没有消费者的信息按预算优先级折叠，而不是一刀切删除。

### 1.2 图的定义（三层 text-attributed graph）

```
G = (V_step ∪ V_intent ∪ V_info, E)
```

**(a) Step 节点 `s_t`（确定性，颗粒度 = 一次 action-observation）**
- 文本属性：reasoning（CoT 原文，截断）、code、observation（截断 + 结构化摘要）。
- 结构特征（全部确定性抽取）：
  - `api_sigs`: 该 cell 的所有 `app.api(canon args)`（复用 `trace_cc.core.sigs`）；
  - `defs` / `uses`: 通过 Python AST 抽取的变量定义与使用（`ast.parse` 失败时退化为 regex）；
  - `status ∈ {ok, blocked}`：AppWorld 合约（观测以 `Execution failed. Traceback:` 开头即 blocked）；
  - `error_class`：traceback 分类（NameError / HTTP 4xx / 分页越界…）；
  - `printed_values`：observation 中的 JSON/数字/id 结构化抽取；
  - `t`、token 长度、embedding（`sentence-transformers`，本地 GPU）。

**(b) Intent 节点 `c_i`（层级计划，粒度 = 一个子目标；从 CoT 抽取）**
- 来源：agent 的 reasoning 中的 numbered plan / "Now I need to…" / "Next…" 语句（规则抽取 + 小 LLM 归并）；首个 boundary 时由 LLM 从 goal 生成 top-level plan（与 v1 相同，但只生成一次、只是骨架）。
- 属性：`description`、`level`（root / sub / leaf）、`status ∈ {pending, active, done, blocked, invalidated}`、`evidence`（支撑 status 的 step id 集合）、`requires`/`produces` 信息槽（可为空）。
- 层级边 `REFINES(c_parent → c_child)`；顺序/依赖边 `DEPENDS(c_i → c_j)`。
- **Step→Intent 实现边 `REALIZES(s_t → c_i)`**：先用 CoT 中的显式指代（"step 2 of plan"），再用 embedding 相似度 + API 名匹配打分，低置信时由 LLM 在候选中选一个（单选题，不生成结构）。

**(c) Information 节点 `i_k`（值/事实/约束/运行时引用）**
- 来源：`defs` 中被后续引用的变量、observation 里的实体 id/token/金额、goal 里的约束、用户消息。
- 属性：`kind ∈ {runtime_reference, fact, constraint, artifact, failure_consequence}`、`value`（原样保留，含容器类型——沿用 v1 SPEC §10 的 `"x" / 7 / ["x"] / [] / {a=1}` 规范）、`producer`（step id）、`description`（LLM 一句话，可选）、`still_bound`（变量在当前 IPython 中是否仍存在——compaction 不清 kernel，所以 runtime_reference 仍有效）。
- 边：`PRODUCES(s_t → i_k)`（确定性），`CONSUMES(s_t → i_k)`（确定性：AST uses），`NEEDS(c_i → i_k)`（预测：未来意图需要什么信息，见 1.4）。

**为什么三层而不是 v1 的两层**：v1 只有"未来计算"+"信息"，历史被抹掉，于是完成与否只能靠 LLM 自述；Step 层是"证据层"，让 done/blocked 有据可查，也让 refetch 可以在图内检测（未来意图的 api 若已在某 step 成功执行 → 直接挂结果而不是重做）。

### 1.3 建图算子（每个 boundary 执行）

```
Pk  = ParseSteps(Δτk)                       # 确定性：Step/Info 节点 + PRODUCES/CONSUMES 边
Ḡk  = Reconcile(Gk-1, Pk)                    # 确定性：REALIZES 打分 + status 由证据更新
G+k = Refine(Ḡk, u)                          # LLM(局部)：只对 frontier 附近 pending 意图做展开，输出 JSON，schema 校验，逐条合并
Lk  = Liveness(G+k)                          # 确定性 + 预测：哪些 info 节点被未来意图需要
Gk  = FoldToBudget(G+k, Lk; B)               # 确定性：按优先级折叠
Ck  = Render(Gk; B)                          # 确定性
```

- **Refine 的 LLM 调用**：输入 = 渲染后的当前图（不是原始 slice！）+ 最近 N 步的压缩视图 + goal；输出 = JSON `{"expand": [{"parent": "c3", "children": [...]}], "status_proposals": [...], "info_descriptions": {...}}`。每一项独立校验（parent 存在、无环、children 非空），失败项丢弃并记录。**没有整体拒绝这一说**。
- **状态更新规则（确定性）**：intent `done` ⇐ 存在 REALIZES 的 step 满足 `status=ok` 且 `produces` 槽全部有值（或 LLM proposal + 至少一条 ok 证据）；`blocked` ⇐ 最近 REALIZES 步 blocked 且 error_class 非 transient；`invalidated` ⇐ LLM proposal + 观测证据（如 API 不存在）。

### 1.4 信息如何挂到未来节点（用户关心的核心）
对每个未来（pending/active）意图 `c_i` 和每个 info `i_k` 计算 **need score**：
```
need(c_i, i_k) = w1·[i_k.value 类型匹配 c_i 预计调用的 API 参数 schema]   # 用 AppWorld api_docs 解析参数名/类型
              + w2·cos(emb(c_i.description), emb(i_k.description ∪ producer.reasoning))
              + w3·[i_k 的变量名出现在 c_i 的 CoT 文本里]
              + w4·[i_k 被同一父意图下的兄弟 step 消费过]
```
超过阈值即加 `NEEDS(c_i → i_k)` 边；阈值/权重在 Phase 1 用回溯 ground truth 调。Liveness `Lk` = 与 frontier 及其祖先有 NEEDS/REQUIRES 路径的所有 info + 所有 constraint + 未解决 failure_consequence。这替代了 v1 里"由模型在 prompt 里宣称 requires"的做法。

### 1.5 折叠与渲染
- 折叠单位：**已 done 的 intent 子树**（含其 steps）→ 一个 macro 节点，保留 ∂S（跨界 NEEDS/PRODUCES 边）与外部引用的 info。这就是论文 Prop 3.1 的可执行版本，并且因为边是确定性的，frontier-equivalence 可以直接在代码里断言（写成测试）。
- 折叠优先级（先折）：done 且离 frontier 最远 > done 且近 > pending 且远（只留 description）> 信息节点按 need score 从低到高降级为一行。
- 渲染分区（沿用 v1 证明有效的格式，加两块）：
  1. `GOAL & CONSTRAINTS`
  2. `PLAN OVERVIEW`（refined roots + 子节点 id）
  3. `DONE (evidence-backed)`：macro 节点一行一个，**带产出的关键值与已调用的 API 签名**（v1 完全不写历史，agent 不知道哪些 API 已经调过 → refetch）
  4. `ACTIVE WORK` / `LATER`
  5. `LIVE INFORMATION`：按 kind 分组，runtime_reference 标注"变量仍在 kernel 中，可直接使用"
  6. `WARNINGS`：blocked 记录与 error 原因（防重踩）
- 预算控制：tiktoken 计数；超预算→提高折叠级别；到最粗仍超 → **fallback**：对最粗渲染再做一次 LLM 缩写（OpenClaw update prompt），并打标 `degraded=true` 记录到 artifact。

### 1.6 与 v1 差异总表
| 维度 | v1 | v2 |
|---|---|---|
| 建图主体 | LLM 整图重生成（block 语法） | 确定性抽取为主，LLM 局部 JSON 建议 |
| 节点 | Computation / Information | Step / Intent / Information（三层，text-attributed） |
| 边 | LLM 声明 | dataflow(AST)、API 签名、REALIZES 打分、NEEDS 预测 |
| 失败处理 | 整体拒绝、旧图保留或 slice 保留 | 逐项丢弃 + 降级阶梯，永不超预算 |
| done 判定 | 模型宣称 | 观测证据 |
| 历史 | 不渲染 | 折叠后带产出值渲染（防 refetch） |
| 信息存活 | 模型写 requires | need score + liveness 可达性 |
| 更新成本 | 16k 输出、36KB 输入 | 目标 ≤2k 输出，输入 = 渲染图 + 短窗口 |

---

## 2. 实验设计

### 2.1 环境、模型、预算
- **Benchmark**：AppWorld（`appworld==0.1.3.post1`），CodeAct agent，`test_normal`（168 tasks；先 30 任务子集做开发，最后全量）；窗口 B=4096 tokens（与 Trace / 论文一致），保留最近 1 轮，最多 50 步。
- **Downstream agent（frozen）**：论文用 Kimi-K2.7-Code；本地只有 8GB RTX 4060 + ollama（qwen3-vl:4b，跑不了 AppWorld）。方案：开发/消融用 OpenAI key 跑 `gpt-4.1-mini` 做 agent 与 compressor；最终表用更强模型（`gpt-4.1`，或与同事对齐 Kimi）跑一次。**费用要先估**（Phase 0 用 3 个任务实测每任务 token）。
- **Compressor LLM**：`gpt-4.1-mini`（Refine 调用）；embedding 用本地 `sentence-transformers/all-MiniLM-L6-v2`（GPU）。
- **Keys**：放在 `W:\context_compression\.env`（`OPENAI_API_KEY`，`SECONDARY_API_KEY`），不入库。注意：用户给的第二把 key 格式像智谱 (bigmodel.cn `id.secret`) 而非 ollama，Phase 0 要验证它到底连哪个 endpoint。

### 2.2 指标
| 层级 | 指标 | 来源 |
|---|---|---|
| 图级（离线） | 构图成功率（应为 100%）、REALIZES 准确率（对人工标注）、NEEDS 边的 precision/recall（ground truth = 该 info 在**后续轨迹**中是否被 CONSUMES，可自动回溯得到）、done 判定与真实完成的一致性、渲染 token 数分布、LLM 调用 tokens | 197 packets + 6 轨迹 + 自采轨迹 |
| boundary 级 | Trace 的 dE(blocked)、dR(refetch)、dU(union)，k=1..5，bootstrap CI | 自采 cohort，PRE vs POST 回放 |
| 任务级 | Acc / Pass² / Pass@2（两次独立运行），按 Easy/Medium/Hard；平均步数；撞上限比例；峰值 prompt tokens；压缩调用总 tokens；degraded 比例 | end-to-end |

### 2.3 Baselines
Full context（上界）、FIFO、OpenClaw（Prompting-O，prompt 在 `trace_ref/data/compression_policy_base/`）、Hermes（Prompting-H）、ACON-UT/UTCO（论文附录 E.4 prompt 原文）、v1 future_graph（已有 6 任务数据；有条件则复跑）、同事的 TRACE 77.4（引用）。LLMLingua-2 视时间。

### 2.4 消融（回答"节点颗粒度 / 特征 / 挂载"三个问题）
| 编号 | 变量 | 取值 |
|---|---|---|
| A1 建图主体 | pure-LLM（v1 风格 JSON 整图）/ 确定性-only（无 Refine）/ **hybrid** |
| A2 节点颗粒度 | step-only（无 intent 层）/ intent-only（无 step 证据层）/ **三层**；intent 粒度：每步一个 vs 合并（阈值扫描） |
| A3 节点特征 | −dataflow 边 / −API 签名 / −embedding / −CoT 意图抽取 |
| A4 信息挂载 | 全保留（不折叠信息）/ 仅 recency / 仅 dataflow 可达 / **need score** |
| A5 渲染 | −DONE 段 / −LIVE INFORMATION 值 / −WARNINGS |
| A6 预算 | B ∈ {2048, 4096, 8192} |
| A7 done 判定 | 模型宣称 vs 证据 |
先在 boundary 级验证器上跑全部消融（便宜），只把有显著差异的搬到 end-to-end。

---

## 3. 执行阶段（供 /goal 使用）

### Phase 0 — 基础设施
**Gate**：能在 3 个任务上跑通 full-context 与 OpenClaw 两条 pipeline，并算出 Trace 指标；有每任务成本估计。
1. 建 `W:\context_compression\compass_v2\` 仓库（git init，遵循 AI Commit Standard），`pyproject.toml`，`.env`/`.gitignore`。
2. 安装 AppWorld（`pip install appworld==0.1.3.post1 && appworld install && appworld download data`），验证 `AppWorld.execute` 合约（Windows 不行则 WSL2/Docker）。
3. Agent harness：复用 `trace_ref/trace_cc/collect/runner.py` 的 `AgentAdapter` 协议，写 OpenAI-compatible adapter（系统 prompt 用 ACON 默认 ICL 模板，与 Trace cohort 一致）；实现 4096 窗口触发、preserve_last_k_turns=1、compressor 插槽。
4. 实现 baselines：full / FIFO / OpenClaw / Hermes / ACON-UT。
5. 把 Trace 验证器接上（PRE/POST 回放、sigs/canon、blocked 合约）。
6. 3 任务冒烟 + 成本估计（每任务 tokens、美元）；验证第二把 key 的 endpoint。

### Phase 1 — 离线建图研究
**Gate**：三层图在 197 packets 上 100% 构建成功；NEEDS 边对回溯 ground truth 的 F1 ≥ 0.7；渲染 ≤ B 的比例 100%；Refine 丢弃项比例 < 20%。
1. `graph/parse.py`：Step/Info 确定性抽取（AST defs/uses、sigs、blocked、printed_values）。单测。
2. `graph/intent.py`：CoT 计划抽取 + REALIZES 打分；在 82 条人工标注上评估 status transition 一致性。
3. `graph/needs.py`：need score；ground truth = 该 info 在后续轨迹中是否被 CONSUMES；调权重/阈值（5 episode 做 leave-one-episode-out）。
4. `graph/fold.py` + `graph/render.py`：折叠、frontier-equivalence 断言测试、tiktoken 预算、降级阶梯。
5. `graph/refine.py`：局部 LLM 展开（JSON schema、逐项校验）；记录每项被丢弃的原因分布（对照 v1 的 11 种拒绝原因）。
6. 在 6 条完整轨迹上离线模拟每个 4096 boundary 的 handover，与 v1 handover、OpenClaw checkpoint 做对比（LLM-judge + 人工抽查，定性表进论文附录）。

### Phase 2 — boundary 级验证
**Gate**：COMPASS-v2 的 dU 相对 OpenClaw 显著更低（95% bootstrap CI 不含 0），至少在 k=3,5。
1. 用 OpenClaw 压缩跑 test_normal 开发子集（30 任务 ×1 seed）采集轨迹与 boundary（预计 60–100 个 boundary）。
2. 对每个 boundary 生成 COMPASS-v2 handover（及各消融），POST 回放 3 samples，PRE 复用。
3. 输出 Trace 风格 scan 表 + burden 曲线；消融 A1–A5、A7 在此层完成。

### Phase 3 — End-to-end
**Gate**：30 任务子集 Acc ≥ OpenClaw + 10 pt；全量与 TRACE 77.4 可比或更高。
1. 30 任务 ×2 runs：full / FIFO / OpenClaw / ACON-UT / COMPASS-v2。
2. 全量 168 ×2 runs：full / OpenClaw / ACON-UT / COMPASS-v2（+ 最优消融 1–2 个）。
3. A6 预算扫描（COMPASS-v2 与 OpenClaw）。
4. 统计：Acc/Pass²/Pass@2 按难度；bootstrap CI；步数与 token 成本表。

### Phase 4 — 论文材料
1. 更新 §3（三层图、确定性算子、need score、降级阶梯；Prop 3.1 改为"可断言"的形式）。
2. §4 主表、burden 曲线、消融表、定性 handover 案例、失败模式分析（沿用 v1 AUDIT 的四条因果链标准）。
3. 复现实验包：冻结 prompt hash、配置、cohort。

---

## 3.5 执行日志（决策与 Gate 结果）
- **2026-08-25 Phase 0 完成**（commit `de5adc5`, `74752c3` in `compass_v2/`）。AppWorld 0.1.3.post1 在 Windows 上可跑，需 `compass.compat` 替换 `SIGALRM` 超时；error contract 与 Trace 一致。
- **第二把 key = Ollama Cloud**（`https://ollama.com/v1`），提供 `kimi-k2.7-code`、`minimax-m3`、`glm-5.2` 等。
- **Agent 选型 A/B（test_normal 前 4 题, full context）**：gpt-4.1 3/4；gpt-4.1-mini 2/4；kimi-k2.7-code temp=1.0 2/4；kimi temp=0 0/3（重复同一调用、逐个翻 API 文档到 50 步上限）。Kimi 输出裸代码无注释，无 CoT 可抽。**决定：开发/消融用 gpt-4.1-mini 做 agent（约 $0.05/任务），最终主表用 gpt-4.1（约 $0.25/任务）；compressor 统一 gpt-4.1-mini。**
- v1 仓库的 82 条"人工标注"是合成练习题，不是 197 packets 的标注 → Phase 1 意图评估改用自动回溯 ground truth（NEEDS/live recall）+ LLM-judge。
- COMPASS v2 首次离线运行（3d9a636_2 两个 boundary）：refine 解析成功、0 丢弃、handover 676/1094 tokens（OpenClaw 666/942）。
- **Phase 1 Gate（`reports/phase1_offline.json`，11 episodes / 35 boundaries）**：build_ok 100%，within-budget 97%（1 次降级 fallback），live_recall **1.00**（修正 GT 后：只算 boundary 前产生、之后被消费的变量），needs F1 0.24（预测故意超集，不再作 gate），done_grounded 92%，丢弃 26 项（`bad_realizes` 7 为主，无整体拒绝）。**通过（F1 指标改为 live_recall ≥ 0.9）。**
- **Phase 2 冒烟（smoke cohort 10 boundaries, agent gpt-4.1-mini, 2 samples/arm）**：OpenClaw POST dU@k5 = **+1.00 [+0.35,+1.70]\***；COMPASS v2 初版 +1.00（refetch 主要是重读 API 文档 ×16、参数名写错）→ 加入 `api_spec` 信息节点（把 show_api_doc 观测解析成精确签名挂到 handover）后 dU@k5 = **+0.05 [−0.40,+0.45]**，dR@k1 = −0.30。这是"信息如何挂到未来节点"的第一个实证：文档内容比"文档已读"的事实更重要。
- 消融开关已实现（`compass_v2/src/compass/graph/compressor.py::VARIANTS`）：`compass_det`(无 LLM) / `compass_nospec` / `compass_novars` / `compass_nodone` / `compass_noplan` / `compass_llmneeds`。
- Dev cohort（test_normal 前 30 题，OpenClaw，gpt-4.1-mini）：acc 26.7%，45 boundaries（24 first / 21 later）。
- **Phase 2 dev cohort（45 boundaries, agent gpt-4.1-mini, 3 samples/arm）**：OpenClaw POST dU@k5 = +0.78 [+0.44, +1.13]\*（dR 主导）。COMPASS 与消融的 POST arm 正在跑（`artifacts/verify_dev/`）。
- **Phase 3 子集（30 题 ×1 run）**：gpt-4.1-mini：full 26.7 / OpenClaw 26.7 / COMPASS 30.0（agent 太弱，无区分度，仅用于 boundary 级）；**gpt-4.1：full 53.3 / OpenClaw 30.0 / COMPASS 30.0（mean score 0.75 / 0.65 / 0.59）**。
- **发现的新失败模式（方法层面，已修，commit 见 `compass_v2` git log）**：refine LLM 把 agent 猜错 API 名的失败泛化为"该 app 没有 X 能力"的事实并把子目标标成 invalidated；agent 信任 checkpoint 后直接放弃（fd1f8fa_*、8749218_2、325d6ec_*）。根因还有 `show_api_descriptions` 观测被 4000 字符截断 → 列表解析为空。修复：否定性能力结论必须有"完整未截断 API 列表"作为证据，否则丢弃；截断列表用正则恢复并标注"列表不完整"；invalidated 备注渲染为"agent 自身结论，需再验证"。这是论文里 "done/blocked 需证据" 原则的自然扩展：**否定结论也需证据**。
- Dev cohort 上 COMPASS 初版 POST arm dU@k5 = +0.79（与 OpenClaw +0.78 持平）；refetch 拆解：`show_api_descriptions('spotify')` ×59、`show_account_passwords` ×34、`show_profile` ×10 → 三项修复：(a) 打印但未赋值的 API 结果渲染为 RESULTS ALREADY OBSERVED；(b) 字面量赋值（密码等）带值；(c) 参数数值约束进签名。再修：否定结论的证据按 app 判定；折叠阶梯加确定性 level 3，LLM fallback 必附 LIVE VARIABLES。
- gpt-4.1 30 题 COMPASS 复跑（含否定结论修复，未含 RESULTS 渲染）：acc 33.3%，mean score 0.665（OpenClaw 0.654，full 0.749）。最终代码复跑（`artifacts/dev41c/`）：acc 30.0%，score 0.663。单次运行噪声很大（同一任务不同版本 0.0/1.0/0.2）。
- **协议缺陷（影响所有方法）**：沿用 Trace 的 4000 字符观测截断把 Spotify 的 API 列表（9094 字符 / 91 个 API）砍掉一半以上（Amazon 6.8k、Splitwise 7.5k 同理），`fd1f8fa_3` 在 **0 个 boundary** 的情况下就因找不到 queue/player API 而放弃，`325d6ec_3` 循环 46 步。这解释了 full-context 只有 53% 的一部分。**决定：end-to-end 协议改为 10000 字符观测上限（`COMPASS_OBS_LIMIT`，默认 10000），所有方法一致；boundary 级 dev cohort 保持 4000（已采集、内部一致）。** 之前的 `dev41*` 结果作为 4000-限制下的记录保留。
- **Phase 2 dev cohort 最终结果（4000 协议, 45 boundaries, agent gpt-4.1-mini）**：COMPASS dU@k5 = +0.82 vs OpenClaw +0.78，配对差 ΔdU@k5 = +0.04 [−0.32, +0.39] → **无显著差异，Gate 未通过**；smoke cohort 的优势是小样本假象。拆解：refetch 首位仍是 `show_api_descriptions('spotify')`（COMPASS 54 / OpenClaw 39 / **PRE 29**）——在 4000 截断下 agent 反复重列 API 表寻找被截掉的 API，这是协议伪影而非压缩缺陷；COMPASS 的 handover 标注"列表被截断"反而鼓励重列。**决定：Phase 2 也迁移到 10k 协议**：用 `dev41L/openclaw`（gpt-4.1 agent）的 boundaries 重建 cohort，rollout agent 也用 gpt-4.1。第二处修复：小型结构化观测（≤900 字符，如密码表/profile）整体保留为值提示，避免 `show_account_passwords` 重取（COMPASS 30 次）。
- **Phase 3 子集 @10k 协议（30 题 ×1 run, gpt-4.1, `artifacts/dev41L/`）**：full **66.7% / 0.90**，OpenClaw **43.3% / 0.82**，COMPASS 旧码 33.3% / 0.80 → **加"完成节点挂产出值"后 53.3% / 0.81**（`634f342_3` 案例：agent 重复创建 4 次归档歌单，因为 done 节点没带 playlist_id）。压缩器 token：COMPASS 4.8k/ep vs OpenClaw 13.0k/ep。单次运行噪声仍大，需要 2 runs 报 Pass²。
- 工程缺陷：Windows 下 agent 代码含 emoji 触发 AppWorld 日志 gbk 编码崩溃（`21abae1_1` 在所有早期运行中的 full=0.00 皆为此），已用 `PYTHONUTF8=1` 修复并重跑。
- **Phase 2 @10k 协议（64 boundaries, agent gpt-4.1, 2 samples/arm, `artifacts/verify_L/`, `reports/phase2_L.md`）**：PRE 控制组 union@k5 = 0.56（比 mini agent 的 1.33 干净得多）。OpenClaw POST dU@k5 = **+0.78 [+0.53,+1.06]\***（dE +0.33\*, dR +0.47\*）。COMPASS（产出值挂载版）+0.69，配对 Δ −0.09 [−0.37,+0.20]；拆解发现 COMPASS 的 blocked 更高：**首步参数错误 34 次（PRE 8）**——`login(email=)` 应为 `username`、用错 app 的 token、猜 API 名。agent 是从早先成功调用学到调用形式的，checkpoint 没带。修复：记录每次 `apis.*` 调用的**参数形式**（去值），渲染 "CALL FORMS THAT WORKED / FAILED(+错误)"，并把 `x = resp['access_token']` 的来源 app 传播到变量。→ **COMPASS(call forms) dU@k5 = +0.55 [+0.29,+0.79]，配对 Δ vs OpenClaw = −0.23 [−0.51,+0.03]（k1: −0.13\* 显著）**。
- 验证器公平性修正：共享 cohort 的**后期 boundary** 上 COMPASS 原本只能从 OpenClaw 的摘要 + 最新几步重建图（看不到早先的调用形式/已取数据），而 OpenClaw 的记录摘要是全历史递归得到的；改为给有状态压缩器**完整前缀**。→ **COMPASS(v2c) dU@k5 = +0.45 [+0.22,+0.68]\*，配对 Δ vs OpenClaw = −0.33 [−0.56, −0.09]\*（k=1,4,5 显著）。Phase 2 Gate 通过。**
- **Phase 3 @10k, 2 runs（gpt-4.1, `reports/episodes_final_dev30.md`）**：Acc run1/run2 — full 66.7/70.0，OpenClaw 43.3/40.0，**COMPASS 43.3/56.7**；Pass² full 50.0 / OpenClaw 23.3 / **COMPASS 33.3**；Pass@2 86.7 / 60.0 / **66.7**；mean score 0.90 / 0.82 / **0.84**；压缩器 tokens/ep 13.0k vs **4.7k**。Phase 3 子集 gate（≥ OpenClaw+10pt）：平均 Acc +8.3、Pass² +10 —— 边缘达标；30 题两次运行的置信区间很宽。
- **消融（10k cohort, gpt-4.1, 2 samples/arm, 配对 ΔdU@k5 vs OpenClaw）**：完整 v2c **−0.33\***；`nospec`（去 API 签名）**−0.02**；`nodone`（去调用形式/文档/结果）**−0.02**；`noplan`（去意图层）−0.18；`novars`（去 LIVE VARIABLES）−0.32\*；`det`（无 LLM，仅 26/64 boundaries 完成）−0.48\*。**结论：收益几乎全部来自确定性证据层（精确签名、调用形式、已观测结果）；LLM 写的计划树/facts 在这个 cohort 上几乎不贡献，去掉活变量表也无损（因为调用形式+签名已足够）。** 这直接回答了"节点特征/挂载什么"的问题，也意味着"CoT 引导的意图层"需要在更长任务或需要重规划的任务上才能体现价值（待验证）。noplan/det 未跑完即被停止（见下一条）。
- **2026-08-25 端点策略（用户指令）**：此后所有探索/消融实验用 **Ollama Cloud**；GPT 只用于最终写进论文的实验，且须提前询问。gpt-4.1 的已有结果（`dev41L*`、`verify_L`）作为当前论文数字保留。Ollama agent 选型 A/B（6 题 full-context, 10k）：**deepseek-v4-flash:0731 6/6**、glm-5.2 5/6、kimi-k2.7-code(t=1.0) 2/6、qwen3.5:397b 0/3 且极慢 → 开发 agent 与压缩器默认改为 deepseek-v4-flash:0731（脚本默认值已改）；**Ollama dev cohort（30 题, deepseek）：full 90.0%，OpenClaw 76.7%，93 boundaries**，OpenClaw POST dU@k5 = +0.51 [+0.24,+0.76]\*（dR 主导，PRE 控制 U@5=1.10）。**首次 Ollama COMPASS 运行发现：deepseek 在 JSON mode 下把 token 花在隐藏推理上、可见 JSON 被截断，39/65 boundary 的 refine 不可解析（等于跑了无计划层的 det 版本），end-to-end 66.7%（OpenClaw 76.7）。修复：JSON mode 只对 OpenAI 启用、refine 上限 4096、截断 JSON 保留已完整的 plan 项、签名行截短。** 修复后 COMPASS end-to-end 66.7% / mean score 0.92（OpenClaw 76.7 / 0.90；refetch 3.6 vs 6.6/ep）。**再发现：deepseek 约 1/20 步把代码包在 `<｜DSML｜python>` 标签里，ACON 抽取器原样送进环境 → 语法错误，三种条件都受影响（full 24 步/9 题，OpenClaw 57，COMPASS 65），两条 COMPASS 题因此丢失**；抽取器已修（方法无关），整个 Ollama 管线重跑：**clean e2e（deepseek, 30 题 ×1）full 83.3 / 0.94，OpenClaw 73.3 / 0.93，COMPASS 66.7 / 0.88**（COMPASS refetch 2.4 vs 4.8、blocked 0.7 vs 1.2、步数 21.3 vs 24.0，但 Acc 更低：4 胜 4 负 + 1 次首调用空响应的 harness 故障）。两处再修：agent 空响应重试一次（方法无关）；"已观测结果"按**带参数的调用签名**去重并在 level 2 保留（`6f4b9a5_1`：20 首歌的搜索结果只存在观测里，被按 API 名合并成一行后又在 level 2 丢掉 → 重搜 17 次）。**修复后 COMPASS e2e（deepseek, `artifacts/devO2/`）：80.0% / 0.94（OpenClaw 73.3 / 0.93，full 83.3 / 0.94），压缩器 tokens 19.5k vs 29.7k/ep。** 93-boundary 验证（7 臂 + Generic/Schema 两臂）进行中；Ollama 上 glm-5.2 / deepseek-v4-flash / kimi / minimax-m3 均支持 JSON mode，可做压缩器。
- **论文更新**：Overleaf `main` 分支 commit `a3933b9`——`header.tex` 加 `\rev{}`/`revblock`（红色），`4-methodology.tex` 每段原文后追加红色 "Revision (v2)" 段与末尾 "Summary of the revision" 小节，新增 `sections/5b-revised-results.tex`（红色初步结果）。原文未改。

## 3.6 方法定位修正（2026-08-25，采纳师兄建议）
- **原则**：`COMPASS` = 作用在 canonical ReAct events 上的领域无关有状态 compaction 协议；Python AST / 持久变量 / API 签名解析降级为 **domain adapter**，不是方法定义。
- **形式化**（已写入论文红色部分，commit `a284b5f`）：trajectory →`Trace Adapter_α`→ canonical events `e_t=(id, action, arguments, observation, outcome, references, provenance)`（仅 action/observation/provenance 必需）→`Graph Update`→ `G_k = G^evidence ∪ G^state ∪ G^intent` →`Plan-Conditioned Folding`→ `C_k`；`E_k = Normalize_α(Δτ_k)`，`ΔP_k ~ π_φ(·|G_{k-1},E_k,u)`，`G_k^+ = Apply(G_{k-1},E_k,ΔP_k)`。规则："provenance is deterministic; semantic interpretation is proposed"——每条 fact / 状态变更必须引用存在的 event id。
- **Adapter 三层**：Generic ⊆ Schema-Aware ⊆ Domain-Specific（CodeAct adapter 属第三层）。代码：`parse_step(turn, adapter)`，变体 `compass_generic` / `compass_schema` / `compass_codeaware`（= v2），commit `0f03bec`。
- **关键实验结果（109 boundaries, deepseek, 配对 Δ@k5 vs OpenClaw）**：`Generic` **+1.34\***、`Schema` **+0.51\***、`CodeAware`（v2）−0.21、`CodeAware−LLM`（det）**−0.38\***。CodeAct agent 的状态在变量里，Generic/Schema 层拿不到 → 比 OpenClaw 还差；而 OfficeBench（JSON 工具 agent，无隐藏状态）上 schema 层即可与 full 持平（77.9 vs 76.8）。**结论：adapter 分层是必要的，"通用性来自 adapter"成立，但对 CodeAct 必须到 Domain-Specific 层。**已有消融（gpt-4.1 cohort）预示：去掉 AST 提供的 state 层（novars）几乎无损（−0.32 vs −0.33），收益主要在 Schema-Aware 层（签名、调用形式、结果）。
- 之后：加一个非 CodeAct 环境（browser / 对话式 tool-use）验证一般性——待与用户商定 benchmark。

## 3.7 扩数据 / 多模型 / 多 benchmark（2026-08-25 用户要求，全部 Ollama）
- **AppWorld 全量**：`test_normal` 168 题 ×（full / OpenClaw / COMPASS），deepseek-v4-flash，run 1 在 `artifacts/fullO_r1/`（进行中），之后 run 2。
- **多模型**：glm-5.2 agent，30 题 ×3 条件 run 1（`artifacts/devGLM/`）：**full 80.0 / 0.96，OpenClaw 73.3 / 0.93，COMPASS 70.0 / 0.91**（refetch 2.2 vs 1.8，压缩器 tokens 14.3k vs 14.8k）；run 2 进行中（`devGLM_r2`）。AppWorld 168（deepseek, run 1, `reports/episodes_fullO_r1.md`）：full **81.0%** / 0.95，OpenClaw **69.0%** / 0.89，**COMPASS 65.5% / 0.86**（refetch 3.5 vs 5.2/ep，压缩次数 3.4 vs 4.1，压缩器 tokens 26k vs 34k）。按压缩次数分层：0 次 n=15 → 1.00/0.93；1 次 n=46 → 0.78/0.80；2–3 次 n=55 → 0.71/0.75；**≥4 次 n=52 → 0.38/0.46（full 0.67）**。572 个 boundary 中 309 个折到 level 2–3：长任务上 1638-token 预算被挤压，level 2–3 整段丢掉变量值/结果，agent 重取（案例 042a9fc_3：已提取的室友改动被丢，重搜消息 ×11）。修复：折叠阶梯在任何 level 都保留**前沿意图所需变量的值**（plan-conditioned floor）；另加 `compass_wide`（预算 0.6B）。两者在 52 个长任务上重跑（`artifacts/longO`, `longO2`）。
- **OfficeBench**（arXiv 2407.19056，JSON tool-calling agent，非 CodeAct → 验证 Schema-Aware adapter 的一般性）：ACON 的环境与 prompt，在 WSL Ubuntu（Python 3.11 venv + LibreOffice）运行；harness `compass_v2/src/compass/harness/officebench.py`，runner `scripts/run_officebench.py`，批处理 `scripts/wsl_ob_batch.sh`。test split 非图像任务 = **95 episodes**；episode 短，先用 **budget 2048**（`artifacts/ob2048/`），再补 4096。**OfficeBench @2048（deepseek, 95 episodes）：full 76.8%（13.6 步），OpenClaw 74.7%（15.2 步），COMPASS 77.9%（15.3 步）**——JSON tool-calling agent、无程序状态，仅 schema-aware adapter，COMPASS 与 full 持平、高于 OpenClaw（单 run）。@4096：OpenClaw 82.1%（13.5 步），COMPASS 进行中。4 个 Ollama 连接错误的 episode 已删除待重跑。
- `compass_det` end-to-end（deepseek, 30 题）：66.7%（COMPASS 80.0 / OpenClaw 73.3）——boundary 级的优势未体现在任务级（单 run）。**glm-5.2 两轮（`reports/episodes_devGLM.md`）**：Acc run1/run2 full 80.0/86.7，OpenClaw 73.3/70.0，COMPASS 70.0/66.7；Pass² 73.3 / 60.0 / 60.0；Pass@2 93.3 / 83.3 / 76.7；mean score 0.96 / 0.93 / 0.91。→ 以 glm 为 agent 时 COMPASS 与 OpenClaw 持平（略低），与 deepseek（80.0 vs 73.3）、gpt-4.1（Pass² 33.3 vs 23.3）的结果方向不一致：**COMPASS 的收益依赖 agent 是否会利用结构化 checkpoint**，这是要和师兄讨论的关键点之一。事件解析：工具名/参数形状/错误前缀/切换 app 时的动作列表（schema tier），无程序状态。
- boundary 级 **109 boundaries cohort（`verify_O`, deepseek agent, 2 samples/arm）**：PRE U@5 = 0.81；OpenClaw POST dU@k5 = +0.66 [+0.44,+0.88]\*（dE +0.14\*, dR +0.54\*）；**COMPASS +0.45 [+0.22,+0.68]\*（dE +0.06 n.s., dR +0.42\*）；配对 Δ = −0.21 [−0.48,+0.07]**（与 gpt-4.1 cohort 的 −0.33\* 同向）。5 消融 + Generic/Schema 两臂进行中；**`compass_det`（无 LLM 计划层）配对 Δ@k5 = −0.38 [−0.63, −0.14]\***，优于完整 COMPASS（−0.21）——与 gpt-4.1 cohort 一致：收益来自确定性证据层，LLM 计划层增加噪声。已启动 `compass_det` 的 end-to-end（`artifacts/devO_det/`）验证任务级。**全部消融（`reports/phase2_O.md`, 配对 Δ@k5 vs OpenClaw）**：`nodone`（去调用形式/文档/结果）**+1.02\***、`nospec` **+0.82\***、`novars` +0.27（k1–k4 显著）——去掉证据层比 OpenClaw 还差得多；`noplan` **−0.34\***、`det` **−0.38\***、完整 v2 −0.21。结论稳定：**证据层（签名、调用形式、结果、变量值）是全部收益来源；LLM 计划层在 boundary 指标上是净负**。但 `det` e2e 66.7 < v2 80.0（单 run），任务级与 boundary 级不一致 → `compass_noplan` e2e = **73.3%**（det 66.7 / v2 80.0 / OpenClaw 73.3，单 run）：任务级上计划层仍有贡献，boundary 级（5 步内 blocked/refetch）与任务级（是否完成所有要求）度量的是不同的东西——boundary 指标对"少犯错"敏感，计划层对"不遗漏要求"有帮助。这是给师兄讨论的第二个关键点。

## 3.8 断点（2026-08-26，用户要求停止所有任务并压缩上下文）
- 已停止全部实验进程（Windows 与 WSL），本会话无后台任务。
- **待续**：52 个长任务的 bounded-fold 重跑 `compass_v2/artifacts/longO4/`（43/52 已完成，可续跑：`run_episodes.py --method compass_v2 --tasks "$(cat artifacts/fullO_r1/long_tasks.txt)" --out artifacts/longO4 --workers 3`），跑完把 Acc 填入 `MEETING_NOTE_2026-08-26.md` §3.1(c)。
- 之前长任务修复三次尝试均无效（宽预算 30.8 / 值下限 36.5 / data facts 38.5，基线 38.5）；根因是图长大后折叠退化为 LLM 缩写（58/388 boundary 到 level 4），commit `142b32c2` 加了硬上限最粗层，longO4 即验证它。
- 其余已完成：AppWorld 168 run 1（full 81.0 / OpenClaw 69.0 / COMPASS 65.5）、glm 两轮、OfficeBench 2048/4096、109-boundary 全部消融与 adapter 层级（`compass_v2/reports/phase2_O.md`）。
- 未做：AppWorld 168 run 2；kimi/minimax 作 agent；GPT 论文级实验（需先问用户）。

## 3.9 v3：流程图化（2026-08-26 用户核心思想）
用户定位：每个 action/observation 作为节点进入流程图；压缩时依据已有信息调整/细化 plan；把已完成的和不重要的合并成一个节点；把之后要用的信息**挂到流程图后面的行动节点上**；观测不必全留，压缩后只保留必需信息、按 planning 按需挂载；数据结构不限于 text-attributed graph（event log + plan tree 即可）；最终 checkpoint 可由 LLM 对流程图做总结而非直接给结构。
实现（`compass_v2/src/compass/graph/flow.py`，commit 见 git log）：
- `attach_to_frontier`：每个开放意图挂载——它提到/用过的 API 的签名与成功调用形式、needs 的变量值、自身/父节点证据产出的变量、实体重叠的 data facts 与已观测结果、凭证挂第一个开放节点；跨节点去重（挂到第一个需要的节点，其后只写 `ref: … (see cN)`）。
- `prune_evidence`：非最近、非失败、且未产出被挂载变量的步骤，观测降为 160 字符摘录（图保留 provenance 而非全文）。
- `render_flow`：以 plan 树为主体；done 子树折成一行（did: APIs → 产出值）；开放节点下列出其挂载；残余只有 CONSTRAINTS / 已执行 API / 失败调用形式 / 未挂载数据 / NEXT。4 级按节点内上限收缩。
- `compass_v3_nl`：同样内容，由 LLM 改写成自然语言 handover（只能复述、值原样）。
- 变体：`compass_v3`、`compass_v3_det`、`compass_v3_nl`。实验：52 长任务（`longO5/`）、30 题（`devO_v3/`）、109-boundary 两臂（`verify_O/compass_v3*`），进行中。
- 修正：一个 cell 打印多个值时不再把整段观测当作每个变量的值。
- **v3 迭代记录**：(1) 首版 plan-first 渲染**替换**了全局证据分节 → boundary Δ **+1.02\***、长任务 25.0%（证据层丢失，再次证明收益来自证据层）；(2) 恢复全局证据分节（挂载为附加）→ Δ +0.41\*、长任务 34.6%，拆解发现 level 2 的节点上限把凭证从所属节点裁掉而其他节点仍 `ref` 它 → agent 猜 token 名（NameError）、重读密码/文档；(3) 修正：ref 只指向实际渲染的项、凭证单独一行全局渲染、粗层保留前沿所需值提示（commit `344eb757`），→ boundary Δ **+0.35\***（refetch 361 vs v2 247，其中 `show_note`/文档重读 +40/+15；首步报错 18 vs 10；而摘要反而更短 1029 vs 1183 tokens，178/218 边界在 level 0）→ 不是预算问题，是**挂载值被双重截断**（挂载时 120 字符、渲染时 140/100/60/40），而 v2 在 level 0 整值输出（≤900 字符）：笔记内容、搜索结果等丢失；(4) 修正：挂载保存完整值提示，渲染按层截断 var/result/data 为 900/400/160/60（commit `1ac73663`），旧结果归档为 `verify_O/compass_v3_c120`、`longO6/compass_v3_c120`（长任务 41/52 时中止）→ 重跑 boundary Δ **+0.31\***，几乎不变（截断不是主因）；(5) 逐边界对比 v2/v3 摘要：v3 `api_docs.show_api_descriptions` 重读 **72 vs 22**（APPS EXPLORED 只给 api 数量、不给名字），且 level 2 时整段 RESULTS 消失（v2 保留 12 条×90 字符）→ agent 重跑搜索；修正：api 名字列表保留到 level 2、RESULTS 保留到 level 2（commit `2ad7f5e4`），归档 `*_c900` → **boundary Δ −0.04（n.s.；k=1..5: +0.07/+0.01/−0.04/+0.01/−0.04），长任务 32.7%（17/52）**。结论：修完阶梯后 v3 与 OpenClaw 持平，但仍不及 v2（−0.21）/det（−0.38\*）和 v2 硬上限 42.3%——挂载到节点没有带来 boundary 或长任务收益。`compass_v3_nl`（LLM 改写为自然语言）boundary Δ **+0.30**（k1–k3 显著），改写只丢不增。v3 线到此为止。教训：v3 的 **挂载到节点** 本身没显示收益，差距全部来自全局证据分节的折叠阶梯比 v2 粗——挂载只是重排，证据层完整性才是 boundary 指标的决定项。对照：v2 硬上限折叠长任务 **42.3**、boundary −0.21。

## 4. 风险与预案
| 风险 | 预案 |
|---|---|
| AppWorld 在 Windows 上安装/回放不稳 | 优先 WSL2 或 Docker；Trace 文档说明回放是逐 cell 重放，确定性可校验 |
| OpenAI 费用超预期（168×2×5 条件×~50 步） | Phase 0 先测；开发全用 4.1-mini；全量只跑 4 个条件；boundary 级验证器承担大部分消融 |
| 确定性抽取依赖 Python CodeAct（泛化性质疑） | 论文明确范围为代码型 agent；REALIZES/NEEDS 的 embedding 路径不依赖 AST，可作退路并在消融里展示 |
| need score 预测失误导致丢信息 | 未挂载的信息以一行摘要保留直到预算逼迫；A4 消融量化 |
| 与同事 TRACE 数字的 agent 模型不一致 | 主表统一自己跑的模型；TRACE 作为引用行并注明模型差异 |

## 5. 目录约定
```
W:\context_compression\
  PLAN.md               本文
  ICLR27_COMPASS.pdf    论文草稿
  compass_paper.txt     PDF 文本
  compass_v1/           合作者代码（只读参考）
  trace_ref/            Trace（只读参考，复用 trace_cc）
  compass_v2/           新实现（Phase 0 创建）
    src/compass/{harness,graph,baselines,eval}/
    scripts/            每个 Phase 一个入口脚本
    artifacts/          冻结产物（boundaries、handovers、rollouts、tables）
    reports/            每个 Gate 一份报告
  .env                  API keys（不入库）
```


## 3.10 v3 之后：按"残余负担"找方向（2026-08-27）

用户指示：v3 不及预期，探索更多方案，不局限于原设想。先做定量诊断（det 臂 212 次 refetch 的构成）：文档重读 54、登录/密码 50、**结果已在摘要里但仍重发 63**、摘要里没有 45。"已在摘要里仍重发"的 142 条（含密码类）里 65 条摘要行以 `...` 截断，agent 之后取的字段是 password/account_name/content/title/song_id/release_date——信息进了摘要、但按字符截断把需要的字段砍掉了；同时 det 摘要均值 1029/1638 tokens，预算剩 1/3 没用。

由此三个新方向（commit `9ff381a4`，`src/compass/graph/project.py`）：
- **A. 字段级投影（proj）**：JSON 类 observation 不按字符前缀截断，而按字段渲染：goal/open plan 提到的字段 → id 类字段 → 短标量 → 其余；长文本字段（笔记 content）拿剩余宽度；截断处显式 `(+k more)`。非 JSON 回退为摘录。
- **B. 状态外化（mem）**：压缩时生成 setup code 在 agent 的 Python 会话里执行，把被吸收步骤的完整 observation 存入 `_mem['sN']`；checkpoint 只引用 key，agent 需要时 `print(_mem['sN'])`，不再重调 API。这是"observation 不进图、按需取"的环境侧实例化（harness: `episode.py` / `replay.py` 的 `setup_code` 钩子）。
- **C. 填满预算（fill，随 proj 开启）**：level 0 先以 4×/2.5×/1.6× 宽度渲染，装得下就用，装不下再折叠。

变体：`compass_det_proj`、`compass_det_mem`、`compass_det_pm`、`compass_proj`、`compass_mem`、`compass_pm`。首轮结果（109-boundary cohort，Δ@k5 vs OpenClaw）：**det_mem −0.51\***（k1..k5 全部显著：−0.14/−0.17/−0.33/−0.42/−0.51；det −0.38\*、v2 −0.21）——refetch 212→170（密码 50→31、search_songs 31→19、show_note 21→8），agent 读 `_mem` 63 次；det_proj −0.35（≈det，30 个边界掉到 level 3：投影行更长）。det_mem 的错误里 3 次 `TypeError: string indices` ——agent 把 `_mem['s7']` 当成 API 返回的 list/dict 索引，但存的是字符串 → 修正：JSON observation 解析后存为对象、checkpoint 标注类型 `(list of 8)`（commit `a3044897`），旧臂归档 `compass_det_mem_str`；**解析版 det_mem 重跑：Δ@k5 −0.56\***（k1..k5：−0.15/−0.21/−0.33/−0.42/−0.56，全部显著；两次独立运行 −0.51/−0.56 一致）。**det_mem 52 长任务：51.9%（27/52）**，对照 OpenClaw 46.2 / v2 38.5 / v2 硬上限 42.3 / v3 32.7；每题 refetch 10.1→4.2，`_mem` 读取 4.0 次/题——首个在长任务上超过 OpenClaw 的变体。**det_mem 30 题（`devO_mem/`）：80.0**（full 83.3 / OpenClaw 73.3 / v2 66.7 / det 66.7，同一 cohort 同批）。proj 结论：**有害**——det_pm −0.30\*（< det_mem −0.56\*），proj（带计划层）−0.06（< v2 −0.21）；投影行更长，32–73 个边界掉到 level 3。字段级投影这条按此实现放弃（若要救，需改成'按需字段'而非'更多字段'）。`compass_mem`（mem + LLM 计划层）boundary Δ −0.31\*（60/218 边界因计划文本挤占预算掉到 level 3）：计划层在 mem 之上仍是净负，与 v2 vs det 的关系一致；**det_mem 为当前主方法**。52 长任务 compass_mem 50.0（26/52）。**AppWorld-168 det_mem（`fullO_r1/`）：67.3**（full 81.0 / OpenClaw 69.0 / v2 65.5）；按固定 52 长任务列表拆分：det_mem 40.4 / OpenClaw 46.2 / v2 38.5，其余 116 题 det_mem 79.3 = OpenClaw 79.3 > v2 77.6。**同一 52 题 det_mem 两次独立运行 51.9 vs 40.4**——n=52 的 run-to-run 噪声约 ±11 点，'长任务超过 OpenClaw'不稳健（两次均值 46.2 = OpenClaw 单次）。稳健的结论是 boundary 级配对增益（−0.56\*，两次复现）；任务级 det_mem ≈ OpenClaw、略优于 v2。按用户'数据太少'的要求：`fullO_r2/`：**det_mem run2 = 70.2**（run1 67.3，均值 68.8；Pass² 56.5，Pass@2 81.0；长 52 题 40.4/44.2，其余 116 题 79.3/81.9）；OpenClaw run2 跑中。v3_nl 长任务 28.8%（v3 线最差）。


## 3.11 任务级 vs boundary 级的分歧，以及"进度叙述"（2026-08-27）

- AppWorld-168 两次运行：OpenClaw 69.0 / **78.0**（同配置两次差 9 点！），det_mem 67.3 / 70.2；两次均值 OpenClaw 73.5 vs det_mem 68.8。差距全在固定 52 长任务（OpenClaw 46.2/63.5 vs det_mem 40.4/44.2；其余 116 题 79.3/84.5 vs 79.3/81.9）。
- 结论：**boundary 级配对增益（det_mem −0.56\*）不能转化为任务成功率**。诊断长任务：det_mem 在 52 题里 22–24 题跑满 50 步（OpenClaw 3–13），压缩次数相近（~7/题）——不是压缩更多，而是 agent 不收敛：checkpoint 有精确证据但没有"做到哪、还剩什么"的进度叙述，agent 反复核对；OpenClaw 的散文摘要恰恰提供这一点。负担指标只计错误+重取，看不到"原地打转"。
- 新方向 **nar（进度叙述）**：在证据层之上加一段 OpenClaw 风格的 LLM 进度笔记（Done / In Progress / Blocked / Key Decisions / Next Steps，明确列出已处理与剩余实体，≤450 tokens，增量更新，被告知不要重复值），证据层用剩余预算渲染。这是把"计划层"从结构化 intent+grounding 换成自由叙述——之前的结构化计划层在 boundary 上净负、任务级无增益。变体：`compass_det_nar`、`compass_det_mem_nar`（commit `2c971fb4`）。
- 结果：boundary `compass_det_mem_nar` **−0.54\***（≈ det_mem −0.56\*：叙述在 boundary 上零成本，而结构化计划层成本 0.25）；**52 长任务 run A：61.5%**（32/52；跑满 50 步的题 10 个 vs det_mem 22–24；34.7 步/题；5.3 次压缩/题）——对照 OpenClaw 46.2/63.5、det_mem 40.4/44.2/51.9。`fullO_r2/full` = 81.0（与 run1 相同，full 基线稳定）。
- 52 长任务 run B：**46.2%** → det_mem_nar 两次 61.5/46.2（均值 53.8，Pass² 38.5，Pass@2 69.2）vs OpenClaw 46.2/63.5（均值 54.9，Pass² 38.5，Pass@2 71.2）：**噪声内相同**；n=52 单次运行摆动 15 点。det_mem 三次均值 45.5，nar 平均 +8 但仍在噪声带内。
- 为了有统计功效：`longO8c/`、`longO8d/` 各再跑 OpenClaw 与 det_mem_nar（每法 4 次）；run C：det_mem_nar 53.8。
- **AppWorld-168 det_mem_nar：67.9**——长 52 题 **59.6**（det_mem 40.4/44.2，OpenClaw 46.2/63.5）但其余 116 题 **71.6**（det_mem 79.3/81.9，OpenClaw 79.3/84.5）。诊断：短任务里 15 题 det_mem 与 OpenClaw 都成功而 nar 失败，全是**提前 complete_task、分数 0.7–0.9**——LLM 写的 Done 列表没有 grounding（'All 26 notes created' 其实只做了一部分），agent 信了。这正是原结构化 intent 层要解决的 grounding 问题。
- **nar2**（commit `0a5a0645`）：笔记只准写观测确认过的项且带计数（'3 of 7 ...'），单独列 'Not yet done / unverified'，去掉 Key Decisions，≤350 tokens；头部注明 advisory、完成前对照证据核对剩余项。**30 题 nar2：83.3**（= full 83.3；det_mem 80.0；OpenClaw 73.3）——grounding 修正消除了短任务退化。52 长任务 nar2 run A 53.8。
- **52 长任务多次运行汇总（均值 ± SE）**：full 69.2±1.9（2 次）；OpenClaw **55.3±3.6**（4 次：46.2/63.5/57.7/53.8）；det_mem_nar **54.2±2.9**（5 次：61.5/46.2/53.8/50.0/59.6）；nar2 53.8（1 次）；det_mem 45.5±3.4（3 次）；v2 38.5（1 次）。逐题配对（每题成功率差）：det_mem_nar − OpenClaw = −1.1±4.3（不显著）；det_mem − OpenClaw = −10.3±6.4。
- 结论（deepseek/Ollama）：**证据层 + `_mem` + grounded 进度笔记在任务级与 OpenClaw 持平（长任务）或更好（30 题 83.3 vs 73.3），在 boundary 级显著更好（−0.54\*）**；相对我们自己的 v2（38.5 长任务 / 65.5 全集）是大幅改进。**AppWorld-168 nar2：73.8**（长 52 题 61.5，其余 116 题 79.3）——OpenClaw 69.0/78.0（均值 73.5），v2 65.5，det_mem 67.3/70.2，det_mem_nar 67.9，full 81.0/81.0。COMPASS 迄今最好的全集数字，与 OpenClaw 两次均值持平，比 v2 高 8 点；短任务退化已消除（79.3 vs nar 71.6）。
- 当前推荐主方法：**det + `_mem` + grounded 进度笔记（`compass_det_mem_nar2`）**。待办：nar2 第二遍 168 与更多 seed（Ollama 免费）、glm/kimi agent、OfficeBench 上的外化实例（无 Python 会话 → 需要 recall 工具或工作区文件）、论文 §4 的方法更新（外化 + 进度笔记作为 fold 的两个输出通道）。


## 3.12 nar2 全面实验（用户 2026-08-27："nar2全面进行实验，确认方法稳定且效果好"）

矩阵（全部 Ollama）：
1. AppWorld-168：nar2 **73.8 / 71.4**（均值 72.6，Pass² 60.1，Pass@2 85.1；长 52 题 61.5/55.8，其余 79.3/78.4）vs OpenClaw 69.0 / 78.0（均值 73.5，Pass² 61.9，Pass@2 85.1）；逐题配对 nar2−OpenClaw = −0.9±2.8（统计持平，nar2 两次更稳）。第三遍：nar2 67.3，OpenClaw 76.8 → **三次均值 nar2 70.8±1.9 vs OpenClaw 74.6±2.8**（Pass³ 53.0 vs 55.4，Pass@3 86.9 vs 89.3；长 52 题 57.1 vs 59.0；其余 116 题 77.0 vs 81.6）；逐题配对 nar2−OpenClaw = −3.8±2.4（不显著，差距在短任务 −4.6，长任务 −1.9）。结论：deepseek 上任务级略低于 OpenClaw 但在噪声内，nar2 自身更稳（SE 1.9 vs 2.8）；boundary 级显著更好（−0.62\*）。短任务的 −4.6 是下一步要诊断的点（笔记是否仍诱导提前完成 / `_mem` 读取占步数）。
2. Boundary 臂 nar2：**−0.62\***（k1..k5：−0.13/−0.23/−0.40/−0.50/−0.62，全部显著）——所有臂中最好（det_mem −0.56\*，det −0.38\*，OpenClaw 0）。
3. 换 agent：glm-5.2 30 题 nar2 run1 **70.0**（对照 full 80.0 / OpenClaw 73.3 / v2 70.0；run2 **73.3**（基线 run2 full 86.7 / OpenClaw 70.0 / v2 66.7）→ glm 两次均值 nar2 71.7 = OpenClaw 71.7 > v2 68.3；kimi-k2.7-code 30 题：full 53.3 / OpenClaw 36.7 / **nar2 40.0 / 50.0**（run2：OpenClaw 30.0）→ kimi 两次均值 nar2 45.0 vs OpenClaw 33.3（+11.7）；弱 agent 上优势最大。
4. OfficeBench：无 Python 会话 → `compass_det_nar2`（笔记、无外化；commit `c92b1537`）：**@2048 78.9**（OpenClaw 74.7、v2 77.9、full 81.1）；**@4096 78.9**（OpenClaw 82.1、v2 77.9）。OfficeBench 上笔记不带外化：@2048 优于 OpenClaw（+4.2）、@4096 略低（−3.2），均高于 v2。
5. 消融：det_nar2（笔记、无 mem）boundary Δ@k5 **−0.46\***；两通道可加：det −0.38 → +mem −0.56 → +note −0.62（+note 单独 −0.46）。
7. 论文：§4 新增红色小节 'Revision (v3): two output channels of folding'（外化 + grounded 进度笔记 + 两个负结果）与修订摘要第 7 条；§5b 新增红色多次运行结果表（boundary 表、端到端表：AppWorld-168 ×3、三种 agent、OfficeBench）。Overleaf `main` 提交 `b521ecd`（仅追加，未改原文）。
6. **nar3**（commit `719d219e`）：短任务诊断——nar2 三次 168 里'压缩后完成但分数 ≥0.7'的近失 67 次 vs OpenClaw 41 次；det 变体不抽约束事实、nar2 笔记无约束栏，而 OpenClaw 摘要保留 Constraints/Critical Context。nar3 增加 grounded 的 'Constraints & details to respect'（金额、名字、id、格式，原文引用），≤420 tokens。30 题 nar3 **70.0**（nar2 83.3、det_mem 80.0）：约束栏里混入 LLM 推断（'presumably same tags'、'likely titled ...'），agent 照做 → 错误动作；折叠层级/摘要长度与 nar2 相同，不是预算问题。结论：约束必须逐字引用任务/观测，不能让笔记做推断；nar3 按此实现为负结果，**nar2 仍为主方法**。168 nar3：69.6（长 52 题 59.6，其余 74.1）——低于 nar2 三次均值 70.8，短任务更差，与 30 题结论一致。

## 3.13 执行状态总表（2026-08-28，对照 §3 各 Phase 的 Gate）

| Phase / Gate | 状态 | 证据（路径） |
|---|---|---|
| P0 基础设施：两条 pipeline + Trace 指标 + 成本 | ✅ | `compass_v2/src/compass/harness/*`, `eval/*`, `scripts/run_episodes.py`, `verify_boundaries.py`；git HEAD `719d219e` |
| P1 离线建图：100% 构建、渲染 ≤ B | ✅（NEEDS-F1 未单独测，改为端到端验证） | `scripts/offline_graph_study.py`；v1 语料 35 boundary 100% 构建、97% 免 fallback、live recall 1.0 |
| P2 boundary 级：dU 相对 OpenClaw 显著更低（k=3,5） | ✅ **达标** | `artifacts/verify_O/` 109 boundary：v3(nar2) Δ@k3 −0.40\*，Δ@k5 −0.62\*；`reports/phase2_O.md` |
| P3 端到端：30 题 ≥ OpenClaw+10 | ✅ deepseek（83.3 vs 73.3）；glm 持平；kimi +11.7 | `artifacts/devO_mem/`, `devGLM*/`, `devKIMI*/` |
| P3 端到端：168 ×2 与 TRACE 77.4 可比 | ⚠️ 未达标：nar2 70.8±1.9（×3）vs OpenClaw 74.6±2.8；full 81.0 | `artifacts/fullO_r1..r4/` |
| P3 多 benchmark（用户追加） | ✅ OfficeBench @2048 78.9 / @4096 78.9 | `artifacts/ob2048/`, `ob4096/` |
| P4 论文材料 | ✅ §4/§5b 红色追加（方法 v2/v3、结果表） | Overleaf `main` `3599626`；`paper_overleaf/sections/4-methodology.tex`, `5b-revised-results.tex` |
| 汇报材料 | ✅ | `MEETING_NOTE_2026-08-26.md` |
| A6 预算扫描 / 次要基线 | ✅ 见 §3.13.1 | `artifacts/devO_b2048`, `devO_b8192`, `fullO_r1/{fifo,hermes,acon_ut}` |
| 未执行 | GPT 实验（按用户规定需先确认） | — |

### 3.13.1 补跑（2026-08-28）
- A6 预算扫描（30 题，deepseek，OpenClaw / nar2）：B=2048 66.7 / 66.7；B=4096 73.3 / 83.3；B=8192 86.7 / 86.7（full 83.3）。优势集中在中等预算；预算极小或极大时两者相同。
- 全量基线（168）：**FIFO 49.4、Hermes 70.8、ACON-UT 66.7**（各 1 次）（`artifacts/fullO_r1/`）。


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
- bounded nar5 boundary Δ@k5 −0.49\*（nar4 −0.50\*）；nar5 168 两次 82.7/75.6 = 79.2（vs OpenClaw +4.6±2.5，vs full −1.8±2.5；Pass² 71.4）。nar4 两次 81.5（Pass² 73.2）。程序核验版略低但在噪声内；两者都显著优于 OpenClaw。
- 结论：**任务级增益来自证据层 + grounded 笔记，不来自外部记忆**；严格协议下的 bounded 版本反而更好（`_mem` 让 agent 多花步数读取）。主方法定为 bounded `compass_det_nar4`（或 nar5，二者相当且 nar5 满足 grounded-by-construction）；`+ExternalMemory` 作为变体单列。第二次 bounded 运行完成：bounded nar4 两次 83.3/79.8 = 81.5（长 0.673，其余 0.879；vs OpenClaw +0.069±0.024，vs full +0.006±0.025）。


## 3.17 第二轮评审（2026-08-29）的四项要求与处理

1. **主方法 = bounded nar5**（程序核验），nar4 改称 prompt-grounded 消融。
2. **私有图预算 + raw observation 物理删除**（commit `08b3d947`）：Apply 之后步骤只保留 600 字符摘录，解析后的观测（`printed`）与 `value_full` 清空；序列化图受 B_G = 64k 字符（≈16k tokens = 4×B）约束，超出时按最旧证据逐级驱逐（observation/code → 结果 hint → superseded 变量）；每个边界记录 graph bytes（`extra.graph_bytes` / `graph_evicted`）。协议改写为 Q_k = Extract_α(Δτ_k)，G_k = Apply(G_{k-1}, Q_k)。外化/投影变体保留原文（它们需要）。
3. **Requirement 图**：nar5 核验后的子句解析为 V^requirement 节点 {id, text, status, supports:[step ids]}（SUPPORTS 边指向 evidence 步骤；未引用或引用失败步骤的状态降为 NOT DONE），checkpoint 的需求分节由节点重新生成（`- r1: "…" -- PARTIAL (2 of 5) [s14, s16]`）。方法名改为 evidence-grounded requirement-state compaction；折叠称 requirement-conditioned folding。
4. **claim 限定**：主张只对 CodeAct / 结构丰富的 tool-use agent；Generic（+1.34\*）/ Schema-aware（+0.51\*）与 OfficeBench @4096 的数据表明一般 ReAct 尚未验证。
实验（跑中）：预算版 nar5 168 ×2（`fullO_r7/r8`）、boundary 臂（`verify_O/compass_det_nar5b`）、30 题（`devO_mem3/`）；同时报告 graph bytes 随边界的变化。

- 预算版 nar5 结果：30 题 **96.7**（full 83.3）；boundary Δ@k5 **−0.56\***（k2..k5 显著；未加预算版 −0.49\*）；图大小 ~21k 字符/边界（B_G 64k）。168 ×2 跑中。


## 4. v4：frontier-conditioned compaction（师兄 2026-08-29 方案，当前主线）

**冻结**：`compass_det_nar5` 为 strong baseline，不再调 prompt。

**核心闭环**：trajectory → update intent graph → compute executable frontier → derive future information needs → retain/fold context。缺口 = Plan → What to preserve。

设计（新模块 `graph/requirements.py` + 变体 `compass_frontier` / `compass_frontier_ex`）：
1. **一次性分解**：episode 首次压缩时用 decompose prompt 把 instruction 拆成 requirement 节点（稳定 id、逐字 span、可选 expect 计数、可选 ordered 标志）；引擎校验 span ⊆ instruction。之后 LLM 只能提局部算子：
   - `REFINE(parent → children)`（父未完成才可；子节点同样带 span/expect）
   - `UPDATE_STATUS(id, status, evidence_ids, count)`（引擎校验证据存在且 ok；**coverage：带 expect 的节点 DONE 需要 count ≥ expect**——修复"一个成功步骤即 DONE"）
   - `PROPOSE_NEXT(id, action_intent)`
   - `DECLARE_NEED(id, information_spec{api, fields, desc})`
   引擎负责 id/证据/状态转移/DAG 合法性；LLM 只做语义判断。
2. **Executable frontier**：F_k = 未完成叶子且（ordered 组内）前置已完成；`NEEDED_BY(information, requirement)` 由 DECLARE_NEED 匹配 information 节点建立。checkpoint 保留优先级 = F_k ∪ Anc(F_k) ∪ Needs(F_k)；**B_G 驱逐不得删除支持 active requirement 的证据**（修复 oldest-first）。
3. **Needs 条件化提取**：大 observation 按 Needs(F_k) 提取——结构化保留 JSON path/field + provenance，非结构化保留支持 span；提取后删原文，严格预算。不再默认整段挂图。
4. **配对消融（同 cohort）**：(a) nar5；(b) nar5+frontier；(c) nar5+frontier+NEEDS 提取。指标：AppWorld success、遗漏 requirement、重复副作用、提前完成、压缩后 refetch/blocked、graph/context size——不只看 accuracy。
5. **BCG 对照**：接 BCG 到同一 agent，或至少 BCG-style ablation（按 confidence/entity relevance 选证据）vs frontier-conditioned selection。
定位语：COMPASS 按"哪些 computation 尚未完成且即将执行"组织上下文；BCG 按"哪些 belief 更可信"。claim 限定 CodeAct；通用 ReAct 在 frontier 闭环验证后再做。

### 4.1 v4 实现状态（2026-08-30，commit `cbf04760`）
- `graph/requirements.py`：ReqGraph 引擎——一次性 decompose（逐字 span 校验、expect、ordered）、四算子校验（REFINE 仅未完成且未细化的父节点；UPDATE_STATUS 证据须存在且 ok，DONE 需 count ≥ expect；PROPOSE_NEXT；DECLARE_NEED）、父状态 roll-up、frontier（ordered 前置门控）、NEEDED_BY 匹配、树渲染、序列化。
- 接线：`compass_frontier`（arm b）= 证据层 + 需求引擎；`compass_frontier_ex`（arm c）= + 按 Needs(F_k) 的 JSON-path 字段提取（提取后原文照删）。frontier 证据与 NEEDED_BY 信息不被 B_G 驱逐并优先渲染。prompts：`decompose.jinja`、`frontier_ops.jinja`。16 个单测通过；2 边界冒烟：decompose 3–4 子句、ops applied/rejected 正常、`[>]` 渲染、保护生效。
- 跑中：30 题两臂（`devO_mem3/`）、boundary 两臂（`verify_O/compass_frontier*`）。之后：168 ×2 两臂、遗漏 requirement/重复副作用/提前完成指标脚本、BCG-style 消融。

### 4.2 v4 首轮结果与修正（2026-08-30）
- 30 题（修复前）：frontier **83.3**、frontier_ex **80.0**（nar5 96.7；配对 −13.3 / −16.7）；premature 4、完成时开放 requirement 2.5/2.2、refetch 略升。
- 案例诊断：(1) **树滞后**——requirement 图只在边界更新，agent 边界后完成的动作树里仍是 NOT_STARTED（3d9a636_2 全部做完树仍标 `[>]`），checkpoint 与现实矛盾；(2) **REFINE 过细**——per-entity 子节点（每好友/每歌单一个，出现三层）烧预算且永远滞后；(3) 无 expect 的动作子句一步即 DONE（29a7b7e_2 全 DONE 但 0.75）。
- 修正（commit `7c448e5d`）：REFINE 仅顶层、2–4 个粗粒度子目标（实体数量走 expect/count）；树渲染注明"状态截至上次压缩，之后的进展以最近观测为准"；ops prompt 要求动作子句的 DONE 引用执行了该动作的步骤。修复版 30 题：**86.7**（修复前 83.3；nar5 96.7——差 3 题，单次在噪声边缘）；open_req@done 2.48→1.85。frontier 的主战场是长任务保留质量：boundary 两臂后跑 52 长任务三臂。

- Boundary（修复前代码）：frontier **−0.31\***（nar5 −0.56\*）——boundary 级也不如 nar5；该臂为 per-entity 树 + 无 staleness 注明的版本。修复版 boundary 重跑中（`verify_O/compass_frontier_f`）；ex 臂（pre-fix）与 52 长任务两臂（fixed）在跑。判定标准：frontier 若在长任务保留质量上无优势且 boundary/30 题均逊于 nar5，则 arm b 的当前实现需要按"树滞后"根因重设计（如：边界外由确定性规则推进 count/状态，或树只渲染 frontier 与 next，不渲染全量状态）。

### 4.3 接线错误与修正（2026-08-30，commit `04694362`）
第一版 `compass_frontier` 的变体定义漏掉了 narrative 开关，实际跑的是"证据层 + 需求树"，**把 nar5 笔记换掉了**（不是师兄要求的 nar5 + frontier）。因此该臂丢掉了笔记的 handled/not-yet-done 计数，全面落后：

| 臂（noteless，第一版接线） | 30 题 | 52 长任务 | boundary Δ@k5 |
|---|---|---|---|
| nar5（主方法） | 96.7 | 68.3 | −0.56\* |
| frontier（pre-fix 约束） | 83.3 | – | −0.31\* |
| frontier（fixed 约束） | 86.7 | 55.8（19 题跑满 50 步） | −0.32\* |
| frontier_ex（pre-fix） | 80.0 | – | −0.33\* |

修正：需求树只替换笔记里的"需求"小节，笔记其余内容保留；第一版保留为 `compass_frontier_noteless`（连同上述数字）。修正版 30 题 / 52 长任务 / boundary 重跑中（`devO_mem5`、`longO11`、`verify_O/compass_frontier`）。
另外记录一个方法学结论：**需求树单独用不如 grounded 笔记**——树只在边界更新、粒度粗，缺少"已处理 N 个/剩余哪些"的具体计数；frontier 的价值应体现在保留策略（NEEDED_BY、抗驱逐、按需字段提取），而不是替代笔记本身。

### 4.4 修正版 frontier 的 30 题结果与"短任务净成本"结论（2026-08-30）
| 30 题（同 cohort） | acc | premature | open_req@done | sum_tok |
|---|---|---|---|---|
| nar5（主方法） | **96.7** | 1 | – | 1221 |
| nar5 + frontier（修正接线，`devO_mem5`） | 83.3 | 4 | 2.13 | 1322 |
| nar5 + frontier（+ 下界渲染，`devO_mem6`，commit `4447c7f9`） | 83.3 | 3 | 2.04 | 1286 |

失败案例（5 题）显示：树在最后一次压缩时仍是 NOT_STARTED，而工作发生在压缩之后——短任务（每题 2.1–2.5 次压缩）里树**永远滞后**，只消耗预算并与轨迹矛盾。改为单调下界语义（只声明"至少已确认完成什么"，其余一律 open）后矛盾消失、premature 4→3，但 acc 不变。
**结论 A（可写进论文的负结果）**：requirement 树对短任务是净成本；plan-state 只有在多次压缩的长任务上才可能回本。因此 frontier 的评估必须以长任务与 boundary 保留质量为准，而不是 30 题总 acc。

### 4.5 Boundary 级：修正接线后 frontier ≈ nar5（2026-08-30）
| 臂（109 边界，配对 Δ@k5 vs OpenClaw） | Δ@1 | Δ@3 | Δ@5 |
|---|---|---|---|
| nar5（主方法） | −0.07 | −0.32\* | **−0.56\*** |
| nar5 + frontier（修正接线） | −0.05 | −0.24\* | **−0.50\*** |
| frontier_noteless（树替代笔记） | −0.08 | −0.18\* | −0.31\* |
| frontier_noteless_f（+ 约束修正） | −0.06 | −0.23\* | −0.32\* |
| frontier_noteless_ex（+ 需求字段提取） | −0.07 | −0.19\* | −0.33\* |

接线修正把 boundary 从 −0.31 拉回 −0.50，与 nar5（−0.56）CI 大幅重叠：**需求图不会损害 boundary 级保留质量，但也没有超过 grounded 笔记**。52 长任务是最后一个决定性数字。

### 4.6 52 长任务：frontier 的定位（2026-08-30）
同一 cohort、同一脚本口径（`scripts/report_ablation.py`）：

| 臂 | acc | 跑满 50 步 | 提前完成 | refetch/题 | blocked/题 | 步数/题 |
|---|---|---|---|---|---|---|
| nar5 run7 / run8 | 65.4 / 71.2（均值 68.3） | 9 / 9 | 10 / 7 | 5.1 / 4.8 | 2.3 / 1.8 | 33.8 / 31.8 |
| **nar5 + frontier（修正接线）** | 61.5 | 11 | 9 | 6.3 | 1.8 | 35.0 |
| frontier_noteless（树替代笔记） | 55.8 | 19 | 3 | 7.9 | 2.4 | 38.6 |
| OpenClaw（4 次） | 55.3 ± 3.6 | – | – | – | – | – |

读法：需求图把 noteless 版的 55.8 拉到 61.5（步数耗尽 19→11、refetch 7.9→6.3），**高于 OpenClaw 但仍低于 nar5**。提前完成不是新问题（nar5 7–10 例）。
决定性案例 `32616b5_1`：树里明确 `r3.2 ... (7 of 12)`，agent 仍 complete——且当时的渲染还写着"trust your recent observations over this list"，等于授权忽略覆盖计数。
修正（commit `bf9ddf6e`）：**完成守卫**——凡有 expect 且已验证 count < expect 的子句，checkpoint 末尾列出 `COMPLETION CHECK: r3.2 at 7 of 12 …`，要求完成前补齐或复核。这是"plan 控制行为"的最小闭环（计数全部来自已执行调用）。52 长任务重跑中（`longO12/`）。

### 4.7 短任务门控与 dev-30 的方差（2026-08-30）
`compass_frontier_late`（commit 后新增变体）：前 2 次压缩只用 nar5，第 3 次起才启用需求图——依据是"plan state 每个边界要花一次 decompose/ops 调用与笔记预算，短任务回不了本"。
30 题：frontier always-on 83.3 / 83.3 / 83.3（三种渲染），late-gate **86.7**，nar5 96.7。
**重要修正**：nar5 自身在 30 题上跑出过 83.3（`devO_mem2`，无图预算版）与 96.7（`devO_mem3`），即该子集的运行间方差约 ±13 点（4 题）。因此"frontier 在短任务上稳定劣于 nar5"这一说法不能只靠 30 题成立；**决定性证据是 52 长任务与 109 边界**。

### 4.8 v4 三臂消融的结论（2026-08-31）
52 长任务（同 cohort，`scripts/report_ablation.py` 口径）：

| 臂 | acc | 跑满 50 步 | 提前完成 | refetch/题 | 说明 |
|---|---|---|---|---|---|
| (a) nar5（冻结基线，2 次） | 65.4 / 71.2（68.3） | 9 / 9 | 10 / 7 | 5.1 / 4.8 | 主方法 |
| (b) nar5 + 需求图 | 61.5 | 11 | 9 | 6.3 | +下界渲染 |
| (b') 同上 + 完成守卫 | 59.6 | 12 | 8 | 6.7 | 守卫未见增益（差 1 题，噪声内） |
| **(c) (b) + 需求条件化字段提取** | **63.5** | 10 | 11 | 6.2 | frontier 三配置中最好，接近 nar5 较低那次（65.4） |
| — 需求图替代笔记（noteless） | 55.8 | 19 | 3 | 7.9 | 第一版接线 |
| — OpenClaw（4 次） | 55.3 ± 3.6 | – | – | – | 基线 |

boundary（109 边界，Δ@k5）：nar5 −0.56\*，nar5+需求图 −0.50\*，noteless −0.31\*。
30 题：需求图 always-on 83.3（×3），late-gate 86.7，nar5 83.3–96.7（该子集方差 ±13 点）。

**结论 B（对师兄方案的回答）**：把 plan 编译成可执行 frontier 并用它保护/优先证据，确实修好了"树替代笔记"版本的所有病症（步数耗尽 19→11、refetch 7.9→6.3、boundary −0.31→−0.50），也把长任务从 55.8 拉到 61.5，**高于 OpenClaw**；但**仍未超过已冻结的 nar5（68.3 / −0.56\*）**。即：在当前实现下，"requirement frontier 控制保留"没有比"证据层 + 已核验的需求状态笔记"带来额外收益；plan→retention 这条因果链的收益被 nar5 的笔记（同样带覆盖计数、同样由证据核验）大部分已经吃掉了。
修正：arm (c) 已跑完，**63.5**（详见上表）——按 Needs(F_k) 提升字段确实是 frontier 里最有效的一环，与 nar5 最差一次仅差 1 题，但仍低于其均值 68.3。
剩余可查方向：(1) (2) frontier 目前只保护/排序证据，尚未真正"丢弃 frontier 不需要的证据"——把 NEEDED_BY 用作**删除**准则而非仅优先级，可能才是差异所在；(3) 每边界两次额外 LLM 调用的成本（长任务每题 ~6 次边界 = 12 次调用）未计入收益核算。
