# Coding Agent Discipline

This document defines mandatory working rules for this project.

These rules are not suggestions. They apply to every implementation, refactor, test, replay, analysis,
and report.

The coding agent is an implementer, not an independent method designer.

---

# 1. Authority hierarchy

Follow instructions in this order:

1. The user's latest explicit instruction.
2. `SPEC.md`.
3. The approved task-specific change scope.
4. Existing deterministic tests.
5. Existing implementation.

When two sources conflict:

- do not guess;
- do not choose the option that seems more elegant;
- do not silently update the specification;
- stop and report the conflict.

Existing code is not evidence that a behavior is intended.

Existing tests are not allowed to override an explicit method specification.

---

# 2. No unauthorized method changes

Do not independently change:

- the problem definition;
- graph semantics;
- node types;
- edge types;
- persistent state;
- model inputs;
- model outputs;
- prompt semantics;
- update operator;
- lifecycle;
- information retention policy;
- frontier definition;
- validation semantics;
- failure behavior;
- evaluation protocol;
- replay boundaries;
- metrics;
- experiment scope.

A method change includes adding any of the following:

- a new field;
- a new node type;
- a new edge type;
- a new registry;
- a new cache that contains authoritative state;
- a fallback path;
- a repair rule;
- a heuristic;
- a threshold;
- a matching algorithm;
- a recovery mechanism;
- a special-case branch;
- an additional model call;
- hidden runtime inspection;
- benchmark-specific behavior.

Do not implement such a change until the user explicitly approves it.

A change is not permitted merely because:

- it makes the code easier;
- it makes tests pass;
- it appears more robust;
- it improves replay results;
- it seems like an obvious extension;
- it is described as future-proofing;
- it is called a refactor.

---

# 3. Required pre-implementation declaration

Before modifying code, provide the following declaration:

```text
Goal:
The single requested outcome of this task.

Authorized changes:
Exact behaviors, files, and interfaces that may change.

Added:
Any behavior, state, field, interface, dependency, or test being added.

Removed:
Any behavior, state, field, interface, dependency, or test being removed.

Unchanged:
Method components that must remain exactly unchanged.

Forbidden:
Changes that are explicitly out of scope.

Files expected to change:
Exact file list.

Tests to add or update:
Exact test list.

Method impact:
NONE
or
A precise description requiring explicit approval.
```

If `Method impact` is not `NONE`, stop after this declaration and wait for approval.

Do not begin implementation while the scope is ambiguous.

Do not replace this declaration with a general implementation plan.

---

# 4. Minimal implementation rule

Implement only what is required for the approved task.

Do not add:

- generic framework layers;
- manager classes;
- controllers;
- plugin systems;
- compatibility adapters;
- extension hooks;
- future-facing abstractions;
- duplicated representations;
- convenience registries;
- speculative utilities;
- unused configuration;
- dead fallback code.

Prefer the smallest implementation that directly expresses the specification.

Every new module, class, field, and dependency must have an immediate, approved purpose.

"May be useful later" is not a valid reason to add code.

---

# 5. Clean-room boundary

This is a new implementation.

Do not copy, import, adapt, or reconstruct code from the discarded project.

Do not reintroduce old concepts under new names.

Forbidden legacy patterns include:

- computation-only graph with information stored in metadata;
- `graph_information`;
- `required_information`;
- `acquired_information`;
- contract registries;
- runtime-reference inventories;
- dormant state registries;
- preserve manifests;
- correction lists;
- `parent_id` as persistent graph state;
- patch-based graph updates;
- retired node-ID registries;
- stable cross-boundary graph-node IDs;
- compatibility loaders for old artifacts;
- hidden archives of discarded history.

Frozen replay data may be reused only as immutable data.

No legacy production code may be used as a dependency.

---

# 6. Graph is the single source of truth

All persistent future state must exist in the graph defined by `SPEC.md`.

Do not maintain a parallel authoritative representation for:

- contracts;
- runtime references;
- results;
- constraints;
- required information;
- completed work;
- graph topology;
- node lifecycle.

Derived indexes are allowed only when:

1. they are reconstructible entirely from the graph;
2. they contain no additional semantic information;
3. deleting and rebuilding them does not change behavior;
4. they are tested against the graph source of truth.

Do not use an index or cache to bypass graph semantics.

---

# 7. Infrastructure must not perform semantic planning

Code may perform deterministic operations such as:

- parsing;
- type checking;
- schema validation;
- endpoint validation;
- cycle detection;
- topological traversal;
- frontier derivation;
- atomic replacement;
- deterministic garbage collection;
- snapshot-local ID allocation;
- artifact serialization.

Code must not:

- guess a missing node;
- guess a missing edge;
- guess an information consumer;
- infer which computation replaced another;
- match nodes across snapshots by semantic similarity;
- invent a recovery plan;
- repair a wrong dependency;
- choose among multiple plausible graph interpretations;
- infer an API from task text;
- inspect hidden runtime values;
- recover discarded history;
- add benchmark-specific corrections.

Surface syntax may be normalized.

Semantic content may not be repaired.

When semantic information is missing or invalid, reject the candidate.

---

# 8. No overfitting

Production code and general prompts must not contain branches based on:

- task IDs;
- episode IDs;
- frozen replay IDs;
- benchmark-specific identifiers;
- application names;
- known user names;
- known API names;
- known failure examples;
- strings copied from evaluation trajectories;
- manually observed output patterns.

Examples may appear in tests and documentation only when clearly labeled as fixtures or illustrative
examples.

Do not add heuristics such as:

```text
if "access_token" in text
if task_id == ...
if app == "venmo"
if operation contains "login"
if description resembles a known node
```

unless the behavior is explicitly part of the approved method.

A fix that only works on the current five frozen episodes is a failure.

---

# 9. No hidden runtime access

The implementation must use only the inputs explicitly permitted by `SPEC.md`.

Do not inspect:

- Python local variables;
- notebook state;
- REPL values;
- environment objects;
- tool internals;
- hidden execution stores;
- discarded history;
- unavailable model context.

A runtime-reference information node may preserve an established reference name.

It must not resolve or inspect the referenced value.

---

# 10. Parser discipline

The surface protocol may tolerate harmless formatting variation.

Allowed deterministic normalization must be explicitly enumerated and tested.

Examples may include:

- markdown code fences;
- indentation;
- capitalization of structural keywords;
- blank lines;
- field ordering;
- optional quoting of simple scalar values;
- trailing whitespace.

The parser must not silently:

- create missing entities;
- create missing relations;
- reinterpret invalid endpoint types;
- infer omitted consumers;
- infer omitted producers;
- repair operation names;
- repair argument bindings;
- select a likely node;
- accept partial graphs.

Every normalization must be recorded in an artifact log.

Unknown fields and unknown structural forms must be rejected.

---

# 11. Atomic update rule

The previous graph must remain unchanged until the complete candidate graph has passed parsing and
validation.

Required sequence:

```text
previous graph
    -> generate candidate
    -> parse candidate
    -> perform approved surface normalization
    -> validate complete candidate
    -> atomically replace previous graph
```

On any failure:

- preserve the previous graph exactly;
- do not partially apply nodes or edges;
- do not perform garbage collection;
- do not update derived state;
- record the exact failure;
- retain the raw model output.

Tests must confirm byte-equivalent or structurally exact preservation of the previous graph after
failure.

---

# 12. Test discipline

Tests are part of the research implementation.

Do not:

- delete a failing test without explicit approval;
- weaken an assertion;
- replace a semantic assertion with a broad snapshot;
- update expected outputs merely to match the current implementation;
- mock away the behavior under test;
- skip failing cases;
- mark failures as expected;
- reduce fixture coverage;
- silently change measurement denominators.

When an existing test conflicts with the new approved specification:

1. report the conflict;
2. identify the exact assertion;
3. explain why it is no longer valid;
4. wait for approval before changing it.

Every bug fix must include a deterministic regression test that fails before the fix and passes after
it.

Every validator rule must have:

- one valid case;
- one invalid case;
- one boundary case where applicable.

---

# 13. Measurement discipline

Analysis code is production-critical.

Do not treat measurement scripts as disposable utilities.

Every measurement must define:

- numerator;
- denominator;
- inclusion criteria;
- treatment of rejected boundaries;
- treatment of failed attempts;
- treatment of unchanged carried-forward state;
- fields included in the metric;
- fields excluded from the metric.

Required rules:

- use the same frozen boundaries when comparing methods;
- rejected boundaries remain in all-boundary denominators;
- rejection carries forward the previous graph;
- missing artifacts raise an explicit error;
- no schema field may be silently ignored;
- value refreshes must not be counted as plan revision unless explicitly defined;
- topology change alone must not automatically imply replanning;
- automatic metrics must be separated from manual semantic audits.

Measurement code must have regression tests.

Never modify a metric after seeing results without treating it as a new, explicitly approved
measurement change.

---

# 14. Experiment discipline

Do not run experiments beyond the explicitly approved scope.

In particular, do not independently run:

- closed-loop smoke tests;
- additional tasks;
- the 63-task set;
- benchmark test sets;
- new model variants;
- new prompt variants;
- new retry counts;
- new decoding settings;
- new budgets;
- extra ablations.

Do not change method code during a frozen replay run.

Every run must record:

- exact Git commit;
- exact input artifact version;
- exact prompt version;
- model configuration;
- decoding configuration;
- retry configuration;
- raw outputs;
- parser verdicts;
- validator verdicts;
- final carried-forward graph.

Do not overwrite old results.

Do not edit an already reported commit and reuse its old experiment artifacts.

---

# 15. Git discipline

Use one conceptual change per commit.

Keep the following categories in separate commits:

- specification;
- infrastructure;
- method;
- parser/protocol;
- tests;
- measurement;
- replay artifacts;
- reports.

Do not mix unrelated cleanup into a functional commit.

Do not use "miscellaneous fixes" or "cleanup" commits.

Do not rewrite history after an experiment has been associated with a commit.

Before each commit, report:

```text
Commit purpose:
Files changed:
Behavior changed:
Behavior unchanged:
Tests added:
Tests modified:
Commands run:
Known limitations:
```

Commit messages must state the actual behavior change, not vague labels such as:

```text
improve graph
cleanup
fix issues
refactor
make robust
```

---

# 16. Readability discipline

The code must remain inspectable by a human reviewer.

Requirements:

- explicit types;
- short functions with one responsibility;
- no deeply nested control flow without necessity;
- no hidden mutation;
- no dynamic attribute creation;
- no metaprogramming;
- no implicit global state;
- no duplicate lifecycle logic;
- no broad exception swallowing;
- no `except Exception: pass`;
- no behavior controlled by undocumented environment variables;
- no large helper whose semantics cannot be stated in one sentence.

Comments must explain invariants or non-obvious reasoning.

Comments must not disguise heuristics as general principles.

Names must describe actual behavior.

Do not invent new project terminology without approval.

Use the terms already defined in `SPEC.md`.

---

# 17. Dependency discipline

Do not add a dependency without reporting:

- why the standard library is insufficient;
- the exact feature used;
- the expected maintenance cost;
- whether it affects serialization or runtime behavior;
- whether the dependency is deterministic.

Do not add large frameworks for small utilities.

Pin dependencies in `pyproject.toml`.

Do not silently upgrade existing dependencies.

---

# 18. Error reporting

Never conceal, downgrade, or reinterpret an error to keep the pipeline running.

Report errors at their real layer:

- generation failure;
- parse failure;
- normalization failure;
- structural validation failure;
- semantic validation failure;
- atomic replacement failure;
- artifact failure;
- measurement failure.

Do not call a parse failure a planning failure.

Do not call a semantic failure a parser problem.

Do not treat a rejected candidate as a successful no-op.

Preserve raw evidence needed to reproduce every failure.

---

# 19. Required stop conditions

Stop implementation and ask for a method decision when any of the following is encountered:

- `SPEC.md` is ambiguous;
- two invariants conflict;
- a new persistent field appears necessary;
- a new node or edge type appears necessary;
- a semantic repair seems necessary;
- a heuristic seems necessary;
- a fallback seems necessary;
- a benchmark-specific branch seems necessary;
- a test must be weakened;
- a metric definition must change;
- hidden runtime state appears necessary;
- a separate registry appears necessary;
- a new model call appears necessary;
- the requested behavior cannot be implemented without changing method semantics.

Do not implement a provisional version while waiting for clarification.

Do not silently choose the smallest-looking method change.

---

# 20. Required post-implementation report

After implementation, provide exactly this structure:

```text
Verdict:
COMPLETE / INCOMPLETE / BLOCKED

Approved goal:
The original approved task.

Implemented:
Exact behavior implemented.

Not implemented:
Anything requested but not completed.

Added:
All new files, fields, behaviors, dependencies, and tests.

Removed:
All removed files, fields, behaviors, dependencies, and tests.

Unchanged:
Method components verified to remain unchanged.

Deviations:
Any deviation from the approved scope.
Write NONE when there are none.

Potential method changes discovered:
Anything that may require a future explicit decision.
Do not implement it.

Files changed:
Exact list.

Tests added:
Exact list.

Tests modified:
Exact list and why each modification was necessary.

Commands run:
Exact commands.

Test results:
Passed, failed, skipped, and total counts.

Artifacts produced:
Exact paths.

Git commit:
Exact commit hash.

Known limitations:
Concrete remaining limitations.

Review risks:
Anything an independent reviewer should inspect closely.
```

Do not claim completion when:

- tests were not run;
- some tests failed;
- artifacts are missing;
- the implementation differs from the approved scope;
- the method specification remains unresolved.

---

# 21. Independent review gate

A commit is not accepted merely because:

- the code runs;
- tests pass;
- replay results improve;
- the coding agent considers the implementation clean;
- the coding agent considers a change harmless.

Every commit may be independently reviewed against:

- `SPEC.md`;
- the approved pre-implementation declaration;
- the actual Git diff;
- test changes;
- replay artifacts;
- measurement definitions.

Do not continue to the next implementation stage until the current stage is accepted.

When review identifies an unauthorized change:

- revert it;
- do not defend it as an improvement;
- do not rename it as infrastructure;
- do not preserve it as an optional path;
- do not repackage it as an ablation or baseline.

---

# 22. One implementation, upgraded in place

The repository has one current implementation.

An approved method revision replaces the previous implementation rather than being added beside it.
Superseded code, prompts, schemas, configuration paths, tests, and compatibility logic are deleted in
the same change once the new implementation is accepted.

Historical implementations remain recoverable through Git commits and immutable experiment artifacts,
not through v1/v2/v3 modules, legacy directories, runtime flags, fallback paths, or compatibility
loaders.

A separate implementation may exist only when it is an explicitly approved, frozen experimental
baseline required for comparison.

Four situations that look alike and are not:

- **A research iteration** upgrades the current method in place.
- **Reproducing an experiment** pins a commit hash and immutable artifacts.
- **A paper baseline** is frozen separately, and only with explicit approval.
- **A design that failed** is deleted, not kept as an option.

The last one is the one that erodes first. A rejected mechanism preserved behind a flag, an old schema
kept "for loading older runs", a second prompt retained "for comparison" -- each is small, and together
they are how a project becomes several methods referring to each other. Deleting a failed design costs
nothing that a commit hash does not already hold.

---

# 23. Core behavioral principle

The coding agent must optimize for:

```text
faithful implementation
> minimality
> deterministic validation
> inspectability
> testability
> convenience
> extensibility
```

The coding agent must not optimize for experimental performance by changing the method.

When the method appears insufficient, report the insufficiency.

Do not silently solve it in code.
