# Future computation and information graph

The persistent state at a compaction boundary is the remaining computation together with the
information that remaining computation requires or will produce. Nothing else survives. There is no
history, no transcript, no metadata dictionary beside the graph, and no registry the graph refers into.

At every boundary the whole of that state is regenerated from the previous state and the new
trajectory slice, validated as a whole, and swapped in as a whole. The previous graph is evidence about
what remains; it is not a structure to be edited.

This document is the contract the implementation must satisfy. Where it says *must*, a test enforces it.

---

## 1. What the state has to answer

1. Which computations still need to occur.
2. What information each of them requires.
3. What information each of them will produce.
4. Which computations depend on which.
5. Which established information has to survive because something ahead consumes it.
6. Which information can go because nothing ahead consumes it.

Question 6 is answered by the graph, not by a judgement call at generation time: information with no
consumer is removed deterministically.

---

## 2. Entities

Two node types. No others, and no subtypes.

### 2.1 Computation

A meaningful unresolved future computation — "obtain a usable Venmo access token", "verify that every
requested payment succeeded". Not an API call, not a past step, not a variable, not an observation, not
a reminder to continue.

Three kinds, and the words are used in exactly these senses throughout this document and the prompt:

| | |
| --- | --- |
| **refined computation** | has outgoing `REFINES` edges, and therefore an interface (§3.1) |
| **abstract leaf** | no refinement children, and no operation: an obligation the plan holds and has not broken down |
| **concrete leaf** | no refinement children, described at executable detail with its operation and arguments |

There is no field saying which. The edges say it, and a leaf is any computation with no children.

**An abstract leaf is a leaf, not a small refined computation.** It has no children, it uses
`REQUIRES` and `PRODUCES` like any other leaf, it carries no interface, and it reaches the frontier
when its dependencies are met — because working out how to do something is work, and the way it gets
worked out is that the next regeneration refines it.

A refined computation carries no `operation` and no `arguments`. The work is its children's, and so is
the dataflow: §3.1 gives it an interface instead.

```python
@dataclass(frozen=True)
class ComputationNode:
    id: str
    description: str
    operation: str | None
    arguments: Mapping[str, ArgumentValue]
```

```python
ScalarValue = str | int | float | bool | None

@dataclass(frozen=True)
class InformationReference:
    information_id: str

ArgumentValue = ScalarValue | InformationReference
```

`operation` and `arguments` are optional and stay absent until the trajectory establishes them. A value
the agent has already established should be referenced through an information node rather than copied
in as a literal.

There is no status field, and there is none by omission rather than oversight: **every computation in
the graph is unresolved by definition**. A resolved one is not marked, it is gone. There is no
`parent_id`, no `acquired_information`, no `required_information`, no `runtime_refs`, no
`execution_contract_ids`, no history and no corrections list. Each of those was a place where state
accumulated outside the structure that was supposed to hold it.

### 2.2 Information

Something that matters to at least one remaining computation.

```python
class InformationKind(str, Enum):
    FACT = "fact"
    CONSTRAINT = "constraint"
    RESULT = "result"
    CONTRACT = "contract"
    RUNTIME_REFERENCE = "runtime_reference"
    FAILURE_CONSEQUENCE = "failure_consequence"


@dataclass(frozen=True)
class InformationNode:
    id: str
    kind: InformationKind
    description: str
    available: bool
    payload: InformationPayload | None
```

```python
@dataclass(frozen=True)
class ScalarPayload:          value: Scalar
@dataclass(frozen=True)
class ListPayload:            values: tuple[Scalar, ...]
@dataclass(frozen=True)
class MappingPayload:         values: tuple[tuple[str, Scalar], ...]
@dataclass(frozen=True)
class RuntimeReferencePayload: name: str
@dataclass(frozen=True)
class ContractPayload:        operation: str
                              parameters: tuple[str, ...]
                              constraints: tuple[str, ...]
```

Payloads are flat. No nested list, no nested mapping, no serialized Python object, no stringified
`dict` or `list` anywhere in a payload or a description. A runtime reference stores the *name* the
agent bound, never the value behind it, and code never reads the runtime to find that value.

`available=False` means "a future computation is expected to produce this". `available=True` means the
task, the fixed rules, or the observed trajectory established it.

### 2.3 Which payload may sit on which kind

A kind that does not constrain its payload is not a kind, it is a label, and every count of contracts
or runtime references would be counting labels.

| | |
| --- | --- |
| `ContractPayload` | only on `CONTRACT` |
| `RuntimeReferencePayload` | only on `RUNTIME_REFERENCE` |
| an available `CONTRACT` | must carry a `ContractPayload` |
| an available `RUNTIME_REFERENCE` | must carry a `RuntimeReferencePayload` |
| anything with `available=False` | carries no payload at all |

`CONTRACT` and `RUNTIME_REFERENCE` are available-only kinds.

An unavailable expected interface or runtime artifact is represented as a `RESULT` describing what a
future computation will establish. After it is established, a later snapshot may represent it as an
available `CONTRACT` or `RUNTIME_REFERENCE` with the corresponding typed payload.

This settles a contradiction rather than tidying one. A runtime reference that is not available yet
would otherwise say both "the agent bound this name" and "this does not exist yet". The transition is
between snapshots, never a flip of availability on one node.

---

## 3. Relations

```python
class Relation(str, Enum):
    PRECEDES = "precedes"                  # Computation -> Computation
    REQUIRES = "requires"                  # Information -> Computation
    PRODUCES = "produces"                  # Computation -> Information
    REFINES = "refines"                    # Computation -> Computation, parent -> child
    INTERFACE_INPUT = "interface_input"    # Information -> refined Computation
    INTERFACE_OUTPUT = "interface_output"  # refined Computation -> Information
```

Those six endpoint pairings are the only valid ones. `PRECEDES` expresses an execution dependency
that is not already carried by produced information, and it expresses nothing else: it never means
containment.

`REFINES` runs from a refined computation to each computation it was broken into. It says what is part
of what, in the graph as it stands now, and it is the only relation that says so.

A computation has at most one incoming `REFINES` edge — it belongs to one unit of work — and a refined
computation may have as many outgoing ones as it has children. Refinement may nest: a child may itself
be refined.

What is still not in the graph is provenance across a transition. That the refined computation in the
previous snapshot was replaced by a different set of children in this one is a fact about the
transition, and it belongs in the transition artifact where analysis can read it. `REFINES` is not
that: it holds between two computations that are both present now, and it is re-stated by every
regeneration like everything else in the graph.

### 3.1 The interface across a refinement boundary

A refined computation does not require or produce. It declares what it needs from outside itself and
what it will leave behind, and its leaves do the consuming and producing:

| | |
| --- | --- |
| `INTERFACE_INPUT` | information the refined work needs from outside |
| `INTERFACE_OUTPUT` | information the refined work will establish for what comes after |

**The interface and the execution below it name the same information nodes.** Not a copy of the value,
not a second node describing the same thing, not an entry in a table beside the graph, and never a
match inferred from a name or a description. Identity is the whole mechanism, and it is what lets one
token, one identifier or one confirmed interface exist once and be consumed by leaves in different
branches.

**The interface is complete, not optional.** It is not a set of annotations that must be realized if
present; it is exactly the set of information that crosses the boundary, and §6 invariant 10 compares
the two in both directions. A refinement that reaches outside itself without saying so is refused, and
so is one that declares something which never crosses. Without that, declaring an interface would be a
courtesy and the coupling this design exists for would not hold.

---

## 4. Identity

**Node ids are local to one snapshot.** `c1, c2, c3` for computations, `i1, i2, i3` for information.
They carry no meaning across a boundary and must never be interpreted across one.

There is therefore no retired-id registry, no stable-identity rule, no cross-boundary reuse protocol
and no similarity matching to decide whether two nodes are "the same computation". A regeneration is a
new snapshot; identity is what the snapshot says, not what a label used to mean.

What *is* stable is content: a runtime reference keeps the exact name the agent bound, and a contract
keeps the exact operation. Those are information, not graph identity.

This deletes four failure modes observed in the previous implementation at once — ids colliding with
retired ones, content sliding between ids when a node was inserted, an id being overwritten by an
unrelated computation, and the model treating `n1` as "the first row" rather than as a name.

---

## 5. Structure

One `nx.MultiDiGraph`. Node attribute `entity_type`, validated dataclass payload, edge attribute
`relation`.

The graph is the single source of truth. No authoritative dictionary beside it holds contracts, runtime
references, facts, requirements or completed work. An index may exist only as a cache that can be
rebuilt from the graph alone, and a test rebuilds it to prove that.

---

## 6. Invariants

Validation is strict about meaning and tolerant about surface syntax. It reports **all** violations of a
candidate graph, not the first one: short-circuiting on the first error means a wrong rejection hides
everything behind it, which is exactly how one mis-fired check in the previous implementation masked
whatever else was in the same graph.

1. **Types.** Every node is a computation or an information node. Every edge carries one known
   relation, and its endpoints match that relation's allowed types.
2. **Acyclicity.** The whole directed graph is acyclic, checked with
   `nx.is_directed_acyclic_graph`. There is no second, hand-written cycle checker.
3. **Liveness.** Every information node has at least one outgoing `REQUIRES` edge to a surviving
   computation. Information kept because it once mattered is dead information.
4. **Availability.** `available=False` requires exactly one incoming `PRODUCES` edge.
   `available=True` may have no producer.
5. **Argument references.** Every `InformationReference` in an argument names an existing information
   node, and that node has a `REQUIRES` edge to the computation using it. A missing edge is a semantic
   failure; the validator does not add it.
6. **No history.** No completed or invalidated computation is present.
7. **No orphan structure.** No dangling edge, no duplicate id within a snapshot, no unknown relation.
8. **One parent.** Every computation has at most one incoming `REFINES` edge.
9. **Roles.** A refined computation carries no `operation`, no `arguments`, and no `REQUIRES` or
   `PRODUCES` edge. A leaf carries no `INTERFACE_INPUT` or `INTERFACE_OUTPUT` edge, because an
   interface with no descendants is realized by nothing.
10. **Complete boundaries.** For a refined computation `p`, write `D(p)` for its refinement
    descendants and `L(p)` for its descendant leaves.

    An information node **crosses in** when some leaf in `L(p)` requires it and no computation in
    `D(p)` produces it. The nodes declared `INTERFACE_INPUT` on `p` are exactly those.

    An unavailable information node **crosses out** when exactly one leaf in `L(p)` produces it and
    at least one consumer lies outside `D(p)`. The unavailable nodes declared `INTERFACE_OUTPUT` on
    `p` are exactly those. A declared output that is **already available** carries no producer inside
    `D(p)` at all and still has a consumer outside it — what established it has left the graph, which
    is how a completed computation collapses while its output stays for whatever consumes it next.

    Both are set equalities, so each is refused from both sides: an undeclared crossing, and a
    declaration that crosses nothing. Something a refinement both produces and consumes internally is
    not an interface and must not be declared as one.

    Each boundary is judged alone, and that is what makes the rules compose. Information one child
    establishes and another child consumes crosses each of those two boundaries and not the boundary
    of the parent they share.

Invariants 8 to 10 are checked on the candidate as a whole, and every traversal they need terminates
on a graph that violates them: a candidate with a refinement cycle or a child with two parents is
exactly the input these checks exist to report, so nothing may follow the hierarchy as though it were
sound.

### 6.1 What validation deliberately does not check

Whether a contract or a runtime reference was *really* established by the environment is **not** a code
check. It is a manual audit item (§11).

This is a deliberate reversal of the previous implementation, on evidence. There, a name could be
referenced only if the environment had just produced it or the current graph still held it. Anything
the graph dropped once became permanently unmentionable, so pruning and validation together formed a
ratchet: `venmo.login` was observed at boundary 4, dropped at 5 because no node happened to associate
it, and referring to it at boundary 7 was then an error — as was `activity_page0`, a name the agent had
bound at boundary 1 and never unbound. Five of twelve recorded refusals were the system forbidding
mention of something that existed.

A check whose input is "what the current state happens to hold" cannot be a check on reality. Either it
draws on cumulative evidence from every slice seen so far — which is the external registry this design
removes — or it is not a code check at all. It is not a code check.

---

## 7. Lifecycle

At a boundary:

1. The previous graph stays untouched.
2. The generator returns a complete candidate graph.
3. The candidate is parsed; only surface syntax is normalized.
4. The candidate is validated whole, collecting every violation.
5. Dead information — no outgoing `REQUIRES` — is removed deterministically. Code removes it; code
   never infers who a consumer *should* have been.
6. If valid, the candidate replaces the previous graph atomically.
7. If not, the previous graph survives byte-identically and nothing is collected, deleted or merged.
8. Raw output, parsed candidate, normalization log, verdicts and the resulting graph are all saved.

A completed computation does not appear in the new graph. If its result is still needed, it appears as
an available information node wired to the computations that consume it; if it is not needed, it and
its result both go.

This applies at every level. A refined computation whose children have all run leaves with them, and
what survives of the whole subtree is the information the rest of the work consumes. A refinement that
turned out to be wrong is replaced the same way: the children are simply absent from the next graph,
and what their failure established becomes an available `failure_consequence` required by whatever
replaced them. Information the removed branch was the last consumer of has no `REQUIRES` edge left and
is collected by rule 5 above; information another branch still requires keeps that edge and stays.
Nothing here is a judgement, and nothing reconnects an edge the generator did not write.

Failure is not stored as an error message. A failure that changes what remains becomes an available
`failure_consequence` information node required by the revised computation — "search_user is not a
supported Venmo interface", required by "find users through the confirmed search_users interface".

---

## 8. Partial and resumable computations

Work is often interrupted halfway: three pages of four retrieved, nine of twelve payments made. There
is no `in_progress` status for this, and adding one would be the wrong repair. A status word says that
something is unfinished without saying what was achieved or what is needed to carry on, so the resume
state would live in a label the next collapse cannot carry.

The graph already expresses it, and this is how it must be expressed.

**What was achieved becomes available information.** The runtime reference holding the partial
accumulator, the cursor or page number to continue from, which items are done, which are left. Each is
an information node, available, required by the computation that will continue the work.

**What remains becomes a computation.** Not "retrieve the pages" again, which would repeat work, but
"continue retrieving the remaining pages", requiring the information above.

**The finished artifact is separate information, not yet available.** The complete collection is its
own node with `available=False`, produced by that computation. A partial accumulator and a complete
collection are two nodes with different descriptions, never one node whose availability flips: a
consumer that needs all of the data must not be able to read the half that exists.

**The completed part collapses.** The computation that retrieved the first three pages is gone. What
survives is what the remaining work consumes, and nothing else — not the requests that were made, not
the responses in full, not the order they arrived in.

If continuing is no longer possible or no longer wanted, the resume information loses its consumer and
is collected, which is the same rule as everywhere else.

## 9. Frontier

Derived, never stored. **Only leaves are executable.** A computation that has been refined is not work
waiting to be done; it has been said in more detail below, and offering it would offer the same job
twice.

A leaf is executable when no `PRECEDES` edge from a computation still in the graph points at it or at
any of its refinement ancestors, and every information node with a `REQUIRES` edge into it is
available.

**Ordering inherits downward and requirements do not**, and the asymmetry is deliberate. If something
must happen before a refined computation, it must happen before every part of that computation;
otherwise refining a blocked computation would quietly unblock it, and the frontier would depend on
how finely the plan was written rather than on what the work needs. A requirement, by contrast, is
declared where it is used, so a leaf that does not need what its parent needs can still run.

Everything held up is held up visibly: by a `PRECEDES` edge from unfinished work, at its own level or
above it, or by a `REQUIRES` edge from information that is not yet available — whose producer is
upstream in the same graph. Coarse computations are not reported as blocked; they are not waiting for
anything, and what holds their children up is reported on the children.

---

## 10. Rendering

Only the future, in topological order, frontier first, later work after. No completed-history section,
no corrections section, no raw trajectory, no status, no transition provenance. The renderer is
deterministic.

**A graph with no `REFINES` edge renders exactly as it always has**, headings included. This is a
branch in the renderer rather than a property that happens to hold, because the frozen replays and
every artifact they produced were rendered by that path, and a heading changed for tidiness would
invalidate all of them silently.

Where the graph is refined, three sections, and each computation in exactly one of them:

| | |
| --- | --- |
| `REFINED PLAN OVERVIEW` | every refined root: what it is for, its interface, its children's ids |
| `ACTIVE WORK` | what can run now, in full, and whatever stands above it |
| `LATER COMPUTATIONS` | the leaves that are visible and cannot run yet |

The first section holds the refined roots and is named for that rather than for the whole plan, which
it is not: an abstract leaf that was never refined is a root with nothing to overview, and it appears
in one of the other two sections where it can be acted on. The second is `ACTIVE WORK` and not
`ACTIVE REFINEMENT` for the same reason — an executable abstract leaf lands there and is not a
refinement path.

A refined subtree that contains no executable leaf is **not expanded**. Its parent states what it needs
and what it will establish, which is what a reader deciding what to do next uses; its internals are
detail about work that is not happening.

Every information node the reader can see is defined exactly once, at its first structural mention, and
referred to by id afterwards. Information that only a hidden subtree touches is not rendered at all: a
definition with nothing visible to consume it would be the free-floating note this format exists to
avoid.

Information is defined at its first structural mention in rendered order: under its future producer
when one exists, and otherwise under its first consumer. Every later mention is the id alone.

Defining at the producer rather than always at a consumer is what keeps the text readable top to
bottom. A computation that will establish something says so where it is described, the computations
that need it point back at an id already defined, and nothing is referred to before the reader has seen
it. Every information node is defined exactly once, and there is no section of information standing
apart from the work that needs it.

A definition carries the kind: `[i2|constraint]`. A constraint, a failure consequence and a plain fact
are different things to a reader deciding what to do next, and a distinction the graph holds is worth
nothing if the handover drops it. References stay bare, `[i2]`, because the reader has already seen it.

A rendered value keeps both the type and the container the graph holds. `7`, `"7"`, `["7"]` and an
empty list are four different states and none of them may read like another, so scalars use the
protocol's canonical form and containers are written with their brackets:

```text
"x"          a string
7            a number
["x"]        a list of one
[]           a list of nothing
{a=1}        a mapping
{}           a mapping of nothing
```

"nothing" as a word is not enough: it says the container is empty without saying which container it is.

---

## 11. Protocol

The model writes a line-oriented block form, not JSON. One field per line, so a missing comma costs one
field rather than the whole graph.

```text
BEGIN_GRAPH

INFO i1
kind: contract
available: true
description: Confirmed Venmo login interface
contract-operation: apis.venmo.login
contract-parameter: username
contract-parameter: password
END_INFO

COMPUTATION c1
description: Log in to Venmo and obtain a usable access token
operation: apis.venmo.login
argument username = "user@example.com"
argument password = @i3
END_COMPUTATION

EDGE i1 REQUIRES c1
EDGE c1 PRODUCES i2
EDGE i2 REQUIRES c2
EDGE c1 PRECEDES c2

END_GRAPH
```

The parser normalizes, and records every instance of, exactly this list: one matched pair of markdown
fences around the whole answer, indentation, blank lines, capitalization of structural keywords and
field names, spacing around the `:` of a field, spacing around the `=` of an `entry` or `argument`,
quotes around scalars, and trailing whitespace.

Fields within a block may appear in any order. Field order is part of the grammar rather than a
deviation from it, and is not recorded: counting it would turn a harmless choice into a statistic about
how badly the model formats its answers.

The parser repairs nothing else. Not a missing node, not a missing edge, not a dangling reference, not a
wrong endpoint type, not an unavailable information node without a producer, and never a decision about
which computation should consume what. Surface syntax is tolerant; graph semantics are strict.

### 11.1 Everything the schema allows, the protocol can carry

These are the same set, and where they were not, the protocol is fixed rather than the schema narrowed
to hide it.

**Values are single-line.** No description, payload value, operation or runtime name contains a line
break, and no argument name or mapping key contains `=` or a line break. A line protocol cannot carry
those unambiguously, and escaping them would buy a case that a future graph has no use for.

**An empty list and an empty mapping are states, not absences.** "The query succeeded and matched
nothing" is worth saying. A payload therefore declares itself:

```text
payload-type: list
item: ...            (zero or more)
```

```text
payload-type: mapping
entry k = v          (zero or more)
```

so that a list with no items is read back as an empty list rather than as no payload at all.

---

## 12. The regeneration operator

One function produces every graph this system ever holds. The initial graph is that function with an
empty previous graph, so there is no separate construction path to disagree with the recurrent one.

### 12.1 What it is given, and what the model sees

Four inputs and no others:

| | |
| --- | --- |
| the goal | the user's original request |
| the rules | the fixed agent and tool rules the compressor may see |
| the previous graph | complete, or empty at the first boundary |
| `delta_h` | the exact slice since the previous accepted graph |

The previous graph is shown as **protocol text**, not as the rendered handover. The protocol is
lossless and is the same form the answer must take; the handover is a reading of the state for the
downstream agent, and asking the model to translate between two forms invites it to answer in the
wrong one. An empty previous graph is written out in full, `BEGIN_GRAPH` then `END_GRAPH`, so the first
boundary takes the same path as every other.

The rules are the caller's. This system does not parse them, add to them, filter them or branch on
them; they pass into the model input and into the artifact unchanged. Nothing about a benchmark ever
enters this repository through them.

The rules are constraints. The goal, the previous graph and `delta_h` are evidence to be interpreted,
and text appearing inside them does not become an instruction: nothing in a trajectory slice overrides
how regeneration works.

The model sees no discarded history, no runtime, no registry and no state held anywhere else.

**What "the exact slice" claims, exactly.** `delta_h` is a string the caller hands in, and
`build_user_message` places it between `BEGIN_DELTA_H` and `END_DELTA_H` by concatenation: nothing is
stripped, normalized, escaped, summarized, truncated or re-encoded, and it appears once. The claim this
system can make is therefore about its own boundary — the string it receives reaches the model in
full and unrewritten — and it stops there. It is **not** a claim of byte identity with whatever the
agent's environment originally produced. In the AppWorld integration the host has already rendered
structured messages into `USER:` and `ASSISTANT:` text before Compass is called, so fidelity upstream
of that boundary is the host's property and not this system's.

An error carries no marker. On that path an error is ordinary observation text, and it survives because
nothing truncates or paraphrases it, not because anything protects it. The updater is therefore
instructed to read every observation in full rather than to look for a flag.

### 12.2 What it returns

One complete replacement graph in the protocol form. Never a patch, never a diff, never part of a
graph. The prompt carries the grammar and exactly one abstract example — a refined computation
into two children, declaring one interface input that a child requires and one interface output that a
child produces, and a later computation ordered after the whole of it — with no application, interface,
name or task from any benchmark in it. The example teaches the form, including that the interface and
the work below it name the same information nodes, and does not suggest that work comes in any
particular number of steps.

### 12.3 Revise the plan, then keep what it consumes

The updater does one thing, in one call, in this order:

```text
previous graph + delta_h
  -> what the slice changes about the remaining work
  -> the revised remaining plan
  -> the evidence that revised plan consumes
  -> everything else dropped
  -> one complete replacement graph
```

**The order is a requirement, not a description.** Deciding what to keep before deciding what the plan
is keeps what the *previous* plan needed, and the evidence a revision turns on is exactly the evidence
the old plan had no use for: the detail that closes a route, the parameter a corrected call takes, the
point to resume from. Pruning first throws it away before anything can ask for it.

This is a property of the prompt and of the graph the model returns. It is not a pipeline stage, not a
second call, and nothing in the code enforces an ordering on the model's reasoning: `regeneration.py`
makes one call and takes one graph. What the code does enforce is that the *result* holds together —
every retained information node has a consumer, or it is collected (§7).

Retention is decided by the revised plan and by nothing else. There is no relevance score, no
importance weight, no age threshold, no preserve list, no archive, no keyword classification of errors,
and no second model deciding what to keep. An information node survives because a remaining
computation requires it.

### 12.4 Sufficient absorption

A slice has been absorbed sufficiently only when the resulting graph and its rendered handover, alone,
let the downstream agent determine:

1. which remaining objective is active;
2. what is already complete;
3. which route is closed, where one is;
4. the exact next recovery or continuation step;
5. the exact values, identifiers, interfaces and constraints that step needs;
6. what must not be repeated;
7. what information still matters to work further ahead.

An error is not absorbed by a sentence saying that something failed. When an error changes what happens
next, the exact operation, the replacement operation, the missing parameter, the accepted values, the
authentication requirement, the identifier, the changed granularity and the required verification are
carried as information nodes with real consumers, in the kinds §2.2 already defines. A
`failure_consequence` carries the consequence and does not stand in for those details.

**This is a criterion for the prompt and for reading model output. It is not a code check**, and it
must not be implemented as one: whether a natural-language description carries enough to recover is
the kind of question §14 keeps out of the validator, and a checker that guessed at it would refuse
sound graphs and pass unsound ones with equal confidence.

### 12.5 The pipeline

```text
raw output
  -> parse
  -> snapshot the parsed candidate, before anything mutates it
  -> replace(previous, candidate)
  -> render the accepted graph
```

Validation happens once, inside `replace`. Its violations are the validation result; nothing validates
beforehand and then hands a graph on as already checked.

The candidate is snapshotted **before** replacement, because replacement collects dead information from
it in place. What the model produced and what was committed are two different things, and an audit that
can only see the second cannot tell whether the model wrote an information node nobody consumes.

### 12.6 Failure, at the layer that produced it

A parse failure or a validation violation leaves the previous graph unchanged and is reported as what
it is. There is no semantic repair, no placeholder node, no guessed edge or consumer, and no automatic
retry, fallback or second call.

**A model that fails to answer is not a graph that was rejected.** If the call raises, the exception
propagates unchanged, the previous graph is untouched, nothing is retried, and no result is returned at
all. A service failure recorded as `accepted = false` would enter the measurements as evidence that the
compressor writes bad graphs. If the call returns something that is not text, that is a broken adapter
and raises `TypeError` rather than reaching the parser.

### 12.7 The call, and the record of it

The system message is the rendered prompt template: the method, the grammar, the example. The user
message is the four inputs, each in its own section, inserted exactly as given — the only transformation
anywhere is `to_protocol` on the previous graph.

Decoding configuration is the caller's and is not interpreted here. It is frozen, ordered by key,
handed to the model adapter as part of the call, and recorded as that same call. Configuration that the
record describes but the adapter never received would make every measurement unfalsifiable.

Each boundary records: the exact system and user messages and configuration; a hash of the system
message as sent; the four inputs separately, so the assembly can be checked; the raw output; the
parser's normalizations and errors; the parsed candidate snapshot when parsing succeeded; the
validation violations; the verdict; the resulting snapshot; the collected information ids; and the
rendered handover.

---

## 13. Frozen-slice regeneration preflight

Five episodes, replayed from slices recorded when a different implementation ran them. At each
boundary, the operator receives the `event["compass"]["delta_h"]` string exactly as recorded in the
frozen trajectory artifact, and the agent's actions never respond to anything this system writes: the
trajectory is frozen, so this measures regeneration and not a loop.

It is a frozen-slice regeneration preflight with reconstructed per-episode operating rules. It is not
an exact replay of the original compressor's inputs, and it is not a matched-input comparison against
the previous implementation. The report uses those words.

### 13.1 The five episodes

In this order, from `source_path_at_freeze` recorded in the manifest. A sixth episode present in that
directory, `f861c32_2`, is not in the set.

| episode | boundaries | goal | rules |
| --- | --- | --- | --- |
| `042a9fc_3` | 5 | 237 B | 7560 B |
| `6b6ca61_2` | 7 | 432 B | 7763 B |
| `6f4b9a5_3` | 7 | 203 B | 7506 B |
| `83a7951_2` | 11 | 316 B | 7621 B |
| `9dabbc9_3` | 2 | 281 B | 7602 B |

32 boundaries, 410,674 bytes of `delta_h`. Every length here and in the manifest is a UTF-8 byte
length, never a count of characters: six of the 32 slices contain non-ASCII and the two differ.

### 13.2 What each input is

**goal** is `episode["instruction"]`, the JSON string value, passed unchanged. Its recorded hash is
over that string encoded as UTF-8.

**rules** is the committed file named by `rules_file`, read as bytes, verified against `rules_sha256`
over exactly those bytes, and decoded once as strict UTF-8. Nothing strips it, normalizes its
newlines, renders it, truncates it or transforms it in any other way.

**delta_h** is `event["compass"]["delta_h"]`, a string, passed unchanged, ordered by
`compass.compaction_index` ascending. Its recorded hash is over that string encoded as UTF-8.

`FIXED_RULES` is fixed across the boundaries of one episode, not identical across episodes. The five
differ because the live prompt carries that task's supervisor and instruction, so the same instruction
appears in both `ORIGINAL_GOAL` and `FIXED_RULES`. That duplication is what the live interface did and
is not cleaned up here.

### 13.3 The manifest

`inputs/preflight/manifest.json`. Each episode entry carries `id`, `episode_file`,
`episode_file_sha256`, `goal_bytes`, `goal_sha256`, `rules_file`, `rules_bytes`, `rules_sha256`,
`boundary_count`, and one row per boundary holding `compaction_index`, `delta_h_bytes` and
`delta_h_sha256`. `episode_file` is relative to `source_path_at_freeze`; `rules_file` is relative to
the repository.

```text
input_manifest_sha256
  29e9c03a8d36b48a00f12641c2e134661f3e8988e131f3992ae0a8a6aa94138d
```

That is the SHA-256 of the canonical JSON of the `"inputs"` object alone — `ensure_ascii` false,
`sort_keys` true, separators `(",", ":")`, UTF-8, 6,258 bytes — so the field can sit beside what it
covers without hashing itself.

A loader verifies every entry. Matching the top-level hash and then trusting the contents would make
the per-entry hashes decoration.

`source_path_at_freeze` records where the inputs were frozen from. It is not the only place they may
later be read: another directory is acceptable exactly when every committed hash matches.

### 13.4 The rules are reconstructed, and that is a stated limitation

```text
source_kind: reconstructed
historical_byte_identity_verified: false
```

The rollout that produced these trajectories handed its compressor `agent.build_prompt(env)` and
recorded neither those bytes nor a hash of them. The rules here were rebuilt by rendering the AppWorld
agent's inline `PROMPT_TEMPLATE`, lstripped, with that task's supervisor and instruction — the path the
rollout took, since it passed `prompt_file=None`, with nothing prepended because its context sections
were empty, no truncation because the live path truncated nothing, and no tool-call suffix because that
is appended to the system message rather than to `build_prompt`.

The two files that render it are uncommitted in their own repository, so byte identity with the
original run cannot be established. `reconstruction_basis` in the manifest records the base commit and
the hashes of those files and of each task's `specs.json`, which are provenance for the reconstruction
and not runtime inputs: regeneration consumes the frozen rules bytes and never reads `specs.json`.

This is a provenance limitation, and it is not a bar to the preflight, because nothing here claims to
reproduce the original call.

### 13.5 One sample per boundary

```text
sampling_is_deterministic: false
```

The provider ignores `temperature`, so `temperature 0.0` and `seed 1` are sent and recorded and claim
nothing. Each boundary is regenerated once.

The report may say that the sampled regeneration at boundary *k* was rejected. It may not say that
boundary *k* is rejected, that rejection is an inherent property of that boundary, or that one sample
estimates a rejection probability or a boundary's difficulty. A rerun is a new stochastic run: it never
completes an earlier one and never overwrites it.

---

## 14. What only an audit can decide

Code cannot decide, and must not appear to decide:

- whether a contract or runtime reference was genuinely established;
- whether a retained computation is honestly the same work as before;
- whether a description carries a result that should have been an information node;
- which future computation a piece of information ought to serve;
- whether a removed branch deserved removal.

These are audit questions, answered against the recorded artifacts, and the preflight report answers
them explicitly rather than implying them from counts.

---

## 15. Non-goals

Not a knowledge graph of everything observed. Not a compressed transcript. Not a planner with a memory
store attached. No second metadata dictionary, no hidden history, no compatibility layer with the
previous implementation, no semantic similarity matching, no automatic consumer inference, no
benchmark-specific rules, no training, and no closed-loop or 63-task run without explicit approval.

The previous implementation is not a base to refactor. Nothing is copied from it: not the graph module,
the prompts, the patch protocol, the registry lifecycle, `graph_information`, `required_information`,
`parent_id`, dormant contracts, reference inventories, loaders, parser, renderer or node ids. What may
be reused is data only — the exact frozen `delta_h` slices, the fixed goals and rules, and measurement
definitions that come with deterministic tests.
