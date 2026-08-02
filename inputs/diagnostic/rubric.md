# First-pass annotation rubric

One label per decision point, for all 197. Frozen before any annotator sees a packet, so that the hash
an annotation file records is the hash of something that already existed.

## What you can see, and what you cannot

A packet holds the goal, the fixed rules, and the CodeAct history through the observation that has just
arrived: every step as its reasoning, its code and its observation. That is the whole of the evidence.

You may not consult anything later in the trajectory. The question is never "what turned out to work",
it is "what does this observation establish about the work that remains". A path that later succeeded
was not necessarily the only correct one, and a path that later failed was not necessarily wrong here.

## The five classes

Exactly one per packet.

### `ordinary_progress`

The observation advances state or information inside the plan that already exists. Nothing has to be
added to the remaining computation, removed from it, replaced, or reordered.

A computation going from pending to done is this. Obtaining a value that the existing plan already
expected to obtain is this. The frontier moving forward, on its own, is this.

### `progressive_refinement`

The observation permits or requires an existing coarse computation to be decomposed or instantiated
more concretely, without contradicting, invalidating or rerouting any previously committed computation
or dependency.

The remaining structure genuinely expands — that is what separates it from `ordinary_progress`, where
only availability or completion changed. Nothing earlier is shown to have been wrong — that is what
separates it from `structural_revision`.

Typical shapes:

- a coarse goal can now be split into several concrete subtasks;
- "look up the records" was known, and the specific interface is known only now;
- the object to act on was known, and the identifier that binds an argument arrives only now;
- a future node deliberately left abstract is expanded into a local subgraph.

This class exists because the graph is not expected to hold a complete atomic plan from the start. Far
work stays coarse and near work becomes executable. Expanding it is the normal course of the method,
not a repair, and counting it as one would make almost every step look like a failure of the earlier
graph.

### `structural_revision`

The observation makes the correct remaining computation structure different from what it was: something
previously committed is now wrong, impossible, or pointed at the wrong thing.

Subtypes, **one or more**, because a single observation can both reveal a prerequisite and kill the
branch that needed it:

- `new_prerequisite`
- `path_or_branch_invalidated`
- `goal_or_constraint_revised`

**`new_prerequisite` is narrower than it sounds.** It applies only when the new prerequisite changes an
execution structure that was already established — work that was committed and reachable now cannot
proceed as committed. When the prerequisite is simply an internal step of a coarse goal that had never
been expanded, that is `progressive_refinement`. Discovering that an interface needs a token is a
revision if a concrete call had already been committed without one, and a refinement if the call was
still an abstract intention.

### `terminal_transition`

The observation moves the task into its ending, with a stage:

- `ready` — every substantive obligation is complete and the next action is the terminal one;
- `confirmed` — this step *was* the terminal action, and its observation only confirms the episode
  ended.

Only `ready` is used for terminal localization. `confirmed` is reported separately: the corpus contains
every episode's last step, and counting those would manufacture easy positives.

### `indeterminate`

The frozen prefix and the rules do not settle it. Record it and move on.

Never resolved by reading later trajectory, and never revisited once any method's output is known. A
point left indeterminate stays out of the main comparison; that is the cost of not guessing, and it is
cheaper than a label chosen to fill a table.

## When two of them seem to apply

**Revision wins.** If one observation both expands an abstract goal and invalidates something already
committed, the class is `structural_revision`. The expansion is recorded in the second pass, not by
weakening the top-level label.

`terminal_transition` wins over `ordinary_progress` when the remaining obligations are finished, even
if the observation itself looks routine.

## The first-pass record

```json
{
  "packet_id": "...",
  "event_class": "ordinary_progress | progressive_refinement | structural_revision | terminal_transition | indeterminate",
  "revision_subtypes": [],
  "terminal_stage": null
}
```

`revision_subtypes` is non-empty only for `structural_revision`. `terminal_stage` is `ready` or
`confirmed` only for `terminal_transition`, and null otherwise.

## What the first pass does not collect

No obsolete work, no unaffected work, no prerequisite text, no correct frontier, no required execution
detail. Those belong to the second pass, over the locked `structural_revision` and
`terminal_transition` points only. Asking for them at all 197 points would buy fatigue and the label
noise that comes with it.

## On the attestation

An annotation file records an annotator identifier and a statement that the annotator had not seen the
graphs, plans, summaries or audits for these trajectories beforehand. That is a self-declaration
recorded by a program. It is not evidence: no file system can establish what a person has read. The
guarantee comes from how the work is divided and from disclosing the procedure, and the paper should
say so rather than point at this field.
