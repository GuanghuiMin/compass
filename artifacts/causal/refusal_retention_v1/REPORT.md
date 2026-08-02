# Refusal-retention causal diagnostic — STOPPED after cell 1 of 5

An outcome-informed causal mechanism diagnostic of `future_graph_v1r`, halted after the first of
five planned cells. It is not a confirmatory experiment, not a pilot, and no statistical claim is
made from one cell.

## Why it stopped

Not a method result and not a corrupt artifact. **`load_run` refuses to read a `future_graph_v1r`
run**, because the online loader hard-codes the method it will accept:

```python
# src/future_graph/online_run.py:32,609
METHOD = "future_graph_v1"
...
if manifest["operator"] != OPERATOR or manifest["method"] != METHOD:
    raise ArtifactError(f"online manifest: {manifest['method']!r} by "
                        f"{manifest['operator']!r} is not this method")
```

The guard exists so a baseline can never be read as this method, and it was written when only one
method existed. `future_graph_v1r` sets `manifest["method"]` to its own name, exactly as it should,
and the loader then rejects its own artifacts.

The protocol for this diagnostic requires strict-loading each artifact after each task and not
starting the next task when validation fails. Source may not be modified before or during the run.
Both conditions hold, so the run stopped after cell 1 with the evidence frozen and nothing patched.

**A second, narrower gap in the same loader.** `_check_counts` reconciles provider attempts against
the attempts recorded inside boundary and attempt records. When a revision attempt exhausts its
provider retries it produces no `RevisionRecord` at all, so its provider calls have nothing to
reconcile against. Cell 1 recorded 10 provider attempts and 7 across its records — the missing 3
are the terminal `ExhaustedAttempts`. Only reachable on an operational stop, and it did not affect
this run because the completed-run checks are skipped for a non-completed status.

## Cell 1: `83a7951_3`

Ran to step 18 of 50, terminating on an updater operational failure. Score 0.30, task not
successful.

```text
step  6   boundary 0 ACCEPTED, committed and spliced
step 15   revision attempt 1 REFUSED — unaccounted_crossing_relation, 17033 B retained
step 16   revision attempt 2 REFUSED — COMPLETE names a new label, 19101 B retained
step 17   revision attempt 3 REFUSED — COMPLETE names a new label, 36152 B retained
step 18   revision attempt 4 — three empty completions, ExhaustedAttempts, run stopped
```

### The retention machinery worked exactly as declared

Every invariant the version exists to establish held:

| check | result |
| --- | --- |
| refusals recorded under `attempts/`, never `boundaries/` | 3 attempts, 1 boundary, no crossover |
| committed boundary index advanced only on the accepted transition | 1 committed, 3 refused, indices independent |
| retained slice present exactly once at offset zero in every later submission | `slice_001` in attempts [1,2,3] at offsets [0,0,0]; `slice_002` in [2,3] at [0,0]; `slice_003` in [3] at [0] |
| no deletion, no duplication, no identical resubmission | slice ledger verified clean |
| no pending transition left behind | none |
| host state byte-identical across every refusal | verified per refusal by field diff and whole-structure hash |
| adapter performed no concatenation | intervals came from the host; the prefix checks passed on all three |

The state that `future_graph_v1` deleted stayed visible. Each retained slice contains
`supervisor.show_account_passwords`, `splitwise.login`, `venmo.login` and `venmo.show_transactions`,
and slice 3 additionally `splitwise.show_activity` — the credentials, both logins and the
transaction data, all still in the live conversation at step 17.

### But the retained context broke the updater within three steps

```text
step   downstream prompt   above 4096   unabsorbed   retained tokens
 14         9 041             4 945          0              0
 15         9 841             5 745          1          4 675
 16        15 905            11 809          2          9 939
 17        21 232            17 136          3         20 363
 18        updater returned three empty completions in a row
```

Three refusals in three consecutive steps took the downstream prompt from 9.8k to 21.2k tokens and
the updater's input to a 36 KB slice. The updater then produced no visible content on three
identical attempts and the operational policy — unchanged — stopped the cell.

This is a consequence of retention, not an independent outage: the provider had answered on every
prior call in this same run, and the failure arrived precisely when the retained interval was
largest. High context size is explicitly not a stop condition for this diagnostic, so the run did
not stop for it; it stopped for the loader defect above.

Three slices totalling 20 363 tokens were **unabsorbed at task end**. Nothing absorbed them,
because the cell ended before another transition could commit.

### What this cell does not show

**Boundary 0 being accepted at step 6, against step 35 in `future_graph_v1`, is not evidence for
retention.** In v1 the first revision was refused on an availability violation and in v1r the
equivalent first revision was accepted. Both saw the same empty graph and the same kind of slice;
that is a different draw from the same model, not the mechanism under test. Attributing it to
retention would be reading a sampling difference as a causal effect.

What the cell does show is narrower and still useful: the retention machinery is correct, and the
cost it incurs is steep enough to end an episode on its own.

## Comparative table

```text
future_graph_v1, OpenClaw and full_context:  previously completed diagnostic cells
future_graph_v1r:                            outcome-informed causal mechanism diagnostic
```

| task | future_graph_v1 | future_graph_v1r | openclaw | full_context |
| --- | --- | --- | --- | --- |
| 83a7951_3 | ✗ 0.30, 50 steps, cap | ✗ 0.30, 17 steps, updater operational failure | ✗ 0.60, 31 steps | ✓ 1.00, 38 steps |
| 986aa4e_2 | ✗ 0.20, 50 steps, cap | not run | ✓ 1.00, 32 steps | ✓ 1.00, 26 steps |
| 3d9a636_2 | ✗ 0.00, 50 steps, cap | not run | ✓ 1.00, 22 steps | ✓ 1.00, 18 steps |
| 29a7b7e_2 | ✓ 1.00, 24 steps | not run | ✓ 1.00, 28 steps | ✓ 1.00, 20 steps |
| d194965_1 | ✓ 1.00, 26 steps | not run | ✓ 1.00, 19 steps | ✓ 1.00, 25 steps |

Token efficiency is not compared across methods: the older provider-usage fields are incomplete and
the full-context peak field is a cumulative total. Compression cost is reported only within v1r,
from its own deterministic instrumentation, above.

## Decision

**D — the diagnostic is invalidated as run, by an artifact-tooling failure rather than by the
method.** One cell of five is not a basis for A, B or C.

The one cell that ran is not worthless, and points where B would: retention removed the deletion
exactly as specified and the machinery is provably correct, but the retained interval grew fast
enough to take the updater out within three steps. If the remaining four cells behave similarly,
the answer is that the transition invariant is now right and effective compression is unsolved.
That is a hypothesis from one cell, not a finding.

## Unresolved defects

1. **`load_run` method whitelist** — `online_run.py:32,609` accepts only `future_graph_v1`, so no
   `future_graph_v1r` artifact can be strict-loaded. Blocks the diagnostic. Not fixed here: source
   may not change during a run.
2. **`_check_counts` and terminal `ExhaustedAttempts`** — a revision attempt that exhausts its
   provider retries writes no record, so its provider calls cannot reconcile. Reachable only on an
   operational stop.
3. **`56_run_future_graph_pilot.py` difficulty injection** — unchanged, still out of scope.

## Provenance limitation

The trace implementation is committed locally at `5fb7bc75f76e4009586f13ecd67956f0dafaf805` and
**intentionally not pushed**. This diagnostic is therefore not remotely reproducible. A one-commit
patch is archived on this machine only, outside every tracked repository, and is deliberately absent
from this repository because it may contain private code.
