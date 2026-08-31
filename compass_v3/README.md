# COMPASS v3.2 — evidence-grounded requirement-state compaction

Context compression for long-horizon tool-using agents, evaluated on AppWorld (CodeAct) and OfficeBench.
This branch (`compass-v3-requirement-state`) is a complete, self-contained rewrite; it does not share history with `main` / `compass-v1-refinement-interface`.

**Start here**

| What | Where |
|---|---|
| Method + full results + answers to both review rounds | [`docs/METHOD_v3.2_requirement_state_compaction.md`](docs/METHOD_v3.2_requirement_state_compaction.md) |
| Chronological research log (every experiment, every negative result, §3.5–3.17) | [`docs/PLAN.md`](docs/PLAN.md) |
| Advisor report with all tables | [`docs/MEETING_NOTE_2026-08-26.md`](docs/MEETING_NOTE_2026-08-26.md) |
| Raw episode / boundary outputs (JSON per task, logs) | `artifacts/` (see map below) |

## The method in one paragraph

At every compaction boundary the adapter extracts the absorbed turns deterministically into bounded events Q_k
(call forms, outcome, produced values with provenance, API signatures, ≤600-char observation excerpt); the raw
observation is then discarded. A private graph G_k = V^requirement ∪ V^evidence ∪ V^information is updated
incrementally under a byte budget B_G. One LLM call rewrites the task instruction as requirement clauses with a
status; every DONE/PARTIAL must cite the evidence steps that support it, and the compressor verifies that the cited
steps exist and succeeded, otherwise the status is downgraded — so DONE is a predicate checkable on G_k. The
checkpoint C_k (≤ 0.4·B) is rendered from the requirement nodes, the handled / not-yet-done lists and the folded
evidence sections; the agent sees only C_k plus the last raw turn. No environment-side memory, no structured
LLM plan graph (both were tested and rejected — see the docs).

## Headline numbers (AppWorld test_normal, 168 tasks, deepseek-v4-flash via Ollama, B = 4096, 50 steps)

| Method | Acc (2 runs) | vs OpenClaw (paired) | vs full context |
|---|---|---|---|
| Full context | 81.0 | – | – |
| OpenClaw (3 runs) | 74.6 ± 2.8 | – | – |
| **COMPASS v3.2 (`compass_det_nar5`)** | **80.4 / 81.5 = 81.0** | **+6.3 ± 2.7** | +0.0 ± 2.9 |
| Hermes / ACON-UT / FIFO | 70.8 / 66.7 / 49.4 | | |

Boundary-level burden (109 boundaries, Δ@5 vs OpenClaw): −0.56\*. 30-task subset: 96.7 (full 83.3, OpenClaw 73.3).
Claims are scoped to CodeAct / structure-rich tool-use agents; the generic-ReAct adapters are interfaces, not validated results.

## Layout

```
src/compass/graph/parse.py       adapter tiers: turn -> bounded event (generic / schema / codeact)
src/compass/graph/build.py       Graph: evidence + information nodes, dataflow, byte budget (enforce_budget)
src/compass/graph/compressor.py  CompassCompressor, VARIANTS, progress_note, ground_note, parse_requirements
src/compass/graph/render.py      fold ladder + checkpoint rendering
src/compass/harness/             AppWorld / OfficeBench episode loops, LLM client, baselines (openclaw, hermes, acon_ut, fifo, openclaw_mem)
src/compass/eval/                Trace-style boundary verifier (replay, burden, signatures)
scripts/run_episodes.py          end-to-end runs        scripts/verify_boundaries.py   boundary-level verification
scripts/report_episodes.py       tables                 scripts/report_phase2.py       boundary tables
prompts/progress5_*.jinja        the requirement-state note prompts (main method)
tests/test_graph.py              13 unit tests (pytest)
```

Variants (`VARIANTS` in `compressor.py`): `compass_det_nar5` (main), `compass_det_nar4` (prompt-grounded ablation),
`compass_det` (evidence layer only), `compass_v2` (old structured plan layer), `compass_det_mem_nar4`
(+ExternalMemory, upper-bound variant), `openclaw_mem` (fairness control), adapter tiers `compass_generic` / `compass_schema`.

## Artifacts map

```
artifacts/fullO_r1..r8/<method>/   168-task runs (r5/r6: bounded nar4, nar5 without graph budget, openclaw_mem; r7/r8: main method)
artifacts/verify_O/<arm>/          boundary cohort (109 boundaries): rollouts.jsonl per arm; compass_det_nar5b = main method
artifacts/devO*, devGLM*, devKIMI* 30-task runs per agent;   artifacts/longO*   52 long tasks;   artifacts/ob2048, ob4096   OfficeBench
```

## Running

```
python -m venv .venv && .venv/Scripts/pip install -e .   # AppWorld: pip install appworld && appworld install && appworld download data
cp .env.example .env                                      # OPENAI_API_KEY / SECONDARY_API_KEY (Ollama Cloud)
PYTHONUTF8=1 python scripts/run_episodes.py --method compass_det_nar5 --tasks <ids> --out artifacts/run1 --workers 3
PYTHONUTF8=1 python scripts/verify_boundaries.py --episodes artifacts/devO/openclaw --out artifacts/verify_O/x --post-method compass_det_nar5 --n-first 2 --n-later 2 --workers 3
pytest tests -q
```

Defaults: agent and compressor = `deepseek-v4-flash:0731` on Ollama Cloud; `--agent-model / --comp-model` switch models
(`glm-5.2`, `kimi-k2.7-code`, or OpenAI with `--agent-provider openai`). Every script is resumable (existing outputs are skipped).

## Open items for the next plan

1. Future-conditioned extraction of large observations (promote only the fields the requirement nodes need; NEEDED_BY edges).
2. Bounded main method on glm-5.2 / kimi (only the +ExternalMemory variant was run there), and on OfficeBench with nar5.
3. The remaining long-task gap to full context (68.3 vs 69.2) is a doc-reading → compaction loop on multi-app tasks.
4. Paper §4 / §5b on Overleaf still describe the `_mem` version; needs the bounded reframing.
5. GPT-4.1 runs for the paper (not started; policy: ask first).
