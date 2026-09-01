# Six-task diagnostic: three conditions and an artifact-only failure audit

This six-task, one-replicate diagnostic is intended to expose integration or severe behavioral
problems before a larger matched comparison. It is too small and stochastic to establish
comparative effectiveness.

Two things in this report were produced differently and must not be read as one experiment.

**Original diagnostic.** `future_graph_v1` against `full_context`, on six tasks drawn from the
committed metadata-only selection before any of these results existed.

**Outcome-informed extension.** The six `openclaw` cells were added *after* observing the poor
future-graph result, in order to tell "compression at 4K is hard here" apart from "the graph method
is failing". They are not confirmatory evidence and are not part of the pre-registered 120-cell
pilot, which was stopped and not resumed.

No significance test is computed and no superiority is claimed. Token efficiency is **not**
compared: the provider intermittently omits `usage`, so several counters read zero in both
conditions, and `full_context`'s `full_peak_tokens` field is a cumulative input total rather than a
peak. Only success, score, steps, termination and graph-internal behaviour are comparable here.

`83a7951_3` shares a scenario prefix with `83a7951_2` in the protocol-validation corpus. It was
neither excluded nor replaced.

---

## 1. Three conditions side by side

| task | d | future_graph_v1 | openclaw | full_context |
| --- | --- | --- | --- | --- |
| f3f60f0_2 | 1 | ✗ 0.875, 30 steps, task_completed | ✓ 1.00, 19 steps, task_completed | ✓ 1.00, 14 steps, task_completed |
| 29a7b7e_2 | 1 | ✓ 1.00, 24 steps, task_completed | ✓ 1.00, 28 steps, task_completed | ✓ 1.00, 20 steps, task_completed |
| 3d9a636_2 | 2 | ✗ 0.00, 50 steps, max_iterations | ✓ 1.00, 22 steps, task_completed | ✓ 1.00, 18 steps, task_completed |
| d194965_1 | 2 | ✓ 1.00, 26 steps, task_completed | ✓ 1.00, 19 steps, task_completed | ✓ 1.00, 25 steps, task_completed |
| 83a7951_3 | 3 | ✗ 0.30, 50 steps, max_iterations | ✗ 0.60, 31 steps, task_completed | ✓ 1.00, 38 steps, task_completed |
| 986aa4e_2 | 3 | ✗ 0.20, 50 steps, max_iterations | ✓ 1.00, 32 steps, task_completed | ✓ 1.00, 26 steps, task_completed |

```text
future_graph_v1   2 / 6      mean 38.3 steps    3 of 6 hit the 50-step cap
openclaw          5 / 6      mean 25.2 steps    0 hit the cap
full_context      6 / 6      mean 23.5 steps    0 hit the cap
```

Every condition ran the same six tasks at the same 50-step cap; the two compressed conditions used
the same 4096-token window. `full_context` is a ceiling reference and its interface is not matched.

## 2. Future-graph plumbing

27 boundaries: **12 accepted, 12 refused, 3 empty**. Zero integration failures, zero atomicity
violations, zero host desynchronisations, zero leftover pending records; every artifact
strict-loads. All 27 continuations carried their handover, and every task shows a later boundary
consuming an action generated under an earlier one. The loop is not the problem.

| task | refusals at boundary | graph empty at refusal | stale-handover steps | boundaries |
| --- | --- | --- | --- | --- |
| f3f60f0_2 | b1 | no | 9 | 4 |
| 29a7b7e_2 | b1 | no | — (last boundary) | 2 |
| 3d9a636_2 | b1, b4 | no | 10, 7 | 6 |
| d194965_1 | b1 | no | — (last boundary) | 2 |
| 83a7951_3 | b0, b1, b2 | **yes, all three** | 10, 5, 12 | 7 |
| 986aa4e_2 | b1, b2, b3, b4 | no | 7, 6, 8, 6 | 6 |

Refusal causes, none dominant: `overlapping_affected_regions` ×3, `NOW_AVAILABLE` naming
already-available information ×2, validation `cycle` ×2, `undeclared_new_label`, `unknown_anchor`,
`availability`, `leaf_interface_edge`, `internal_information_declared_as_output`, a markdown fence
before `BEGIN_REVISION` with no `END_REVISION`, and a declared information node with no
`description`.

Two provider calls returned `finish_reason=length` at exactly 16384 completion tokens, both on
`986aa4e_2` boundary 2. Both produced no usable visible content and were therefore classified as
empty completions and retried under the existing operational policy, which is why that boundary
records three attempts. Nothing was reclassified and no retry policy changed.

## 3. Task-level diagnosis

### f3f60f0_2 — downstream tool/execution error unrelated to graph state

The load-bearing case, and the graph is **not** the supported cause.

Goal: "Like all the songs and albums in my Spotify song and album library, respectively, that I have
not liked yet." Two requirements, songs and albums.

Both were present in the first accepted graph: `c6`/`c7`/`c8` for songs and `c10`/`c11`/`c12` for
albums, under one refined root. At boundary 2 the accepted graph carried the album obligation
concretely — `[c7] Like each of the 4 unliked albums (IDs 2, 8, 14, 16)` refined into four
`spotify.like_album` leaves. The final observation confirms albums 2, 8, 14 and 16 were liked at
`2023-05-18T12:00:00`, i.e. during this run.

The evaluator requirement that stayed unsatisfied is albums **4 and 5**: both appear in
`show_album_library` and neither appears in `show_liked_albums`, whose final contents are
`{2, 3, 8, 14, 16}`.

They were missing because the consumer's own enumeration was wrong, not because the graph dropped
them. The trajectory shows repeated pagination loops with incorrect break conditions
(`if len(songs) < 5: break` against a larger page size) and three separate occurrences of
`Exception: Number of requests exceeded the limit of 1000. This potentially means that you have an
infinite loop in your code.` The set `[2, 8, 14, 16]` is the consumer's own computed
`albums_to_like`, printed in the trajectory before the graph recorded it. The graph faithfully
recorded a consumer conclusion that was already wrong.

The one refusal (b1) is **not** the supported cause: the album obligation survived it intact and was
still concrete two boundaries later.

Secondary, and worth stating plainly: `COMPLETE` is model-declared, so when the consumer concluded
"All songs and albums in the library are already liked" the graph had no basis on which to disagree,
and boundary 3 rendered `NOTHING REMAINS`. The graph cannot detect a wrong completion claim; it can
only record one.

### 3d9a636_2 — refusal-induced state loss **and** accepted-transition omission

Goal: reset Venmo friends to match phone contacts.

At the first refusal (b1, step 16) the agent was mid-computation over contact and Venmo user sets;
the action failed with a `NameError`, and the handover stayed frozen at b0's for 10 steps. The
discarded slice contained the contact enumeration.

The rediscovery is visible. At step 10 the agent read `show_api_descriptions(app_name='phone')`;
after the b2 boundary it read **the same document again** at step 26, and after b5 it read
`show_api_doc(venmo, add_friend)` at step 48. The automated post-refusal repeat counter reports
0–1 because it only compares an action against its own boundary's slice; these re-reads span
earlier boundaries and the counter under-counts them.

The accepted transitions also under-specify. The final handover at step 48 contains a single active
leaf, `[c2] Get phone contacts and current venmo friends, then add/remove to match`, with no
concrete add/remove work — the plan had collapsed back to one coarse obligation at the point where
concrete steps were needed. The task reached the cap having liked nothing.

### 83a7951_3 — refusal-induced state loss, the clearest case

Goal: record Splitwise payments matching today's Venmo payments, in each expense's group.

Boundaries 0, 1 and 2 were all refused **against an empty graph**, so the agent received
`NOTHING REMAINS` three times while real state was discarded:

| refused boundary | what the discarded slice contained |
| --- | --- |
| b0 (17326 B) | app and API discovery, `supervisor.show_account_passwords` |
| b1 (18578 B) | `splitwise.login`, `splitwise.show_groups` |
| b2 (16040 B) | `splitwise.show_group_expenses`, `show_no_group_expenses`, `venmo.show_all_transactions` |

That is the credentials, the login, the group list, the expense list and the Venmo transaction
list — the entire evidence base of the task — dropped from the compressor's input while the agent
kept its Python state but lost all textual context three times.

The first accepted graph arrived at boundary 3, **step 35 of 50**, and it did recover the substance:
it holds the three Venmo payments with amounts, recipients and transaction ids, and a Splitwise
login obligation. But the immediately following action was
`show_api_descriptions(app_name='splitwise')` — rediscovering documentation already read twice — and
only 15 steps remained for work that had not started. Score 0.30.

### 986aa4e_2 — refusal-induced state loss, from a four-boundary chain

Goal: prepare a Spotify playlist for a Todoist-managed Edinburgh trip.

Boundaries 1 through 4 were refused consecutively, so the handover was frozen at boundary 0's for
27 steps while four slices were discarded. The three causes acted differently:

- **`overlapping_affected_regions` (b1, b3)** — the model named a computation and something it was
  refined into in the same revision. Nothing was applied; the whole revision was lost.
- **`NOW_AVAILABLE` on already-available information (b2, b4)** — the model tried to establish
  information that already existed, which the parser refuses because only a promised result can
  become available. Again the entire revision was lost, including the parts that were fine.
- **`finish_reason=length` ×2 (b2)** — both exhausted the 16384-token budget with no usable visible
  content, were classified as empty completions and retried, and the third attempt returned a
  revision that was then refused for the `NOW_AVAILABLE` reason above. The truncation cost latency,
  not correctness.

Boundary 5 was accepted and the task reached the cap eight steps later.

### 29a7b7e_2 and d194965_1 — why one refusal did not hurt

Both succeeded with exactly one refused boundary, and the reason is **position, not count**: in both
tasks the refusal was the *last* boundary. The agent finished inside the stale-handover window —
24 steps with boundaries at 11 and 20, and 26 steps with boundaries at 16 and 21 — so the discarded
slice was never needed by a later revision and the frozen handover never had to carry work it did
not describe.

This is the cross-task result that matters most: **refusal count alone does not predict failure.**
Both successes had one refusal. `f3f60f0_2` also had one refusal and failed for an unrelated
reason. The failures with refusal chains are the ones where a refusal fell early, on an empty or
young graph, with substantial work still ahead.

## 4. Cross-task audit summary

| task | refusals | earliest supported failure mechanism |
| --- | --- | --- |
| f3f60f0_2 | 1 (late, recovered) | downstream execution error: broken pagination, request-limit exhaustion, wrong `albums_to_like` |
| 29a7b7e_2 | 1 (last boundary) | — succeeded |
| 3d9a636_2 | 2 | refusal-induced state loss at b1, compounded by accepted graphs that collapsed the plan to one coarse leaf |
| d194965_1 | 1 (last boundary) | — succeeded |
| 83a7951_3 | 3, all on an empty graph | refusal-induced state loss: the whole evidence base discarded, first usable graph at step 35 of 50 |
| 986aa4e_2 | 4 consecutive | refusal-induced state loss: handover frozen 27 steps across a four-boundary chain |

Accepted boundaries that omitted an unfinished requirement: **one** — `3d9a636_2`, whose late
accepted graphs carried a single coarse obligation instead of the concrete add/remove work.

Accepted boundaries whose handover contained the requirement but the consumer did not act on it:
**one** — `f3f60f0_2` boundary 2 named albums 2, 8, 14, 16 concretely and the consumer executed
exactly those, having itself failed to discover that 4 and 5 were also unliked.

No scalar verifier, threshold or reward was created.

## 5. Decision

**B. OpenClaw materially outperforms `future_graph_v1` on the same tasks; the current graph
transition/handover method is specifically failing.**

The extension was added precisely to make A distinguishable from B, and it does. At the same
window, on the same tasks, with the same downstream model and step cap, OpenClaw reached 5/6 where
the graph method reached 2/6, and no OpenClaw cell hit the step cap while three graph cells did.
Compression at 4K is evidently not what defeats these tasks.

The audit places the mechanism, and it is not one thing:

- **Three of four failures are refusal-induced state loss** (`3d9a636_2`, `83a7951_3`,
  `986aa4e_2`), and the damage tracks *when* a refusal lands rather than how many there are. A
  refusal on an empty or young graph discards the only record of what has been learned and hands
  the agent `NOTHING REMAINS`; a refusal as the final boundary costs nothing.
- **One of four is not a graph failure at all** (`f3f60f0_2`) but a consumer pagination bug, which
  the graph then faithfully recorded and completed on the consumer's own say-so.
- **Accepted-transition quality is a real but secondary problem**: exactly one task shows an
  accepted graph dropping to a coarse obligation where concrete work was needed.

What this does *not* support is that the handover representation itself is unusable. Where the
graph was accepted and concrete, the consumer acted on it correctly — `f3f60f0_2` boundary 2 is the
clearest example, where the agent executed precisely the four album likes the graph named.

The next method question is therefore about the *transition*, not the rendering: a refused revision
currently costs the entire slice, and that cost is concentrated exactly where the graph is youngest
and the slice most valuable. Nothing in this report justifies patching the individual refusal causes
one at a time — they were eight distinct causes across twelve refusals, which is the signature of a
protocol that is hard to write correctly rather than of a few fixable bugs.

## 6. Unresolved code defect

`56_run_future_graph_pilot.py` injects `row["difficulty"] = cell["difficulty"]` after exporting a
compatibility row. The run artifact carries no `difficulty` field, so `check_row_against_artifact`
rebuilds the row without it and refuses the aggregation. It did not affect this diagnostic — the
cells were driven directly rather than through that driver — and the same mistake made by hand
during this audit was caught immediately by the guard, which is the guard working. It must be fixed
before any future batch execution. Not fixed here: no future-graph batch run is authorised.
