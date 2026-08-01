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
    PRECEDES = "precedes"     # Computation -> Computation
    REQUIRES = "requires"     # Information -> Computation
    PRODUCES = "produces"     # Computation -> Information
```

Those three endpoint pairings are the only valid ones. `PRECEDES` expresses an execution dependency
that is not already carried by produced information.

Refinement lineage is not in the persistent graph. That an old coarse computation was replaced by three
new ones is provenance about a transition; it belongs in the transition artifact, where analysis can
read it, and nowhere else. Keeping it in the state produced a second relation living outside the graph
and disagreeing with it.

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

Derived, never stored. A computation is executable when it has no incoming `PRECEDES` edge from another
computation in the graph and every information node with a `REQUIRES` edge into it is available.

Everything held up is held up visibly: by a `PRECEDES` edge from unfinished work, or by a `REQUIRES`
edge from information that is not yet available — whose producer is upstream in the same graph.

---

## 10. Rendering

Only the future, in topological order, frontier first, later work after. No completed-history section,
no corrections section, no raw trajectory, no status, no transition provenance. The renderer is
deterministic.

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

### 12.2 What it returns

One complete replacement graph in the protocol form. Never a patch, never a diff, never part of a
graph. The prompt carries the grammar and exactly one abstract example — an available information node
required by a computation, that computation producing an unavailable result, that result required by a
second computation, and a `PRECEDES` edge — with no application, interface, name or task from any
benchmark in it. The example teaches the form, and does not suggest that work comes in two steps.

### 12.3 The pipeline

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

### 12.4 Failure, at the layer that produced it

A parse failure or a validation violation leaves the previous graph unchanged and is reported as what
it is. There is no semantic repair, no placeholder node, no guessed edge or consumer, and no automatic
retry, fallback or second call.

**A model that fails to answer is not a graph that was rejected.** If the call raises, the exception
propagates unchanged, the previous graph is untouched, nothing is retried, and no result is returned at
all. A service failure recorded as `accepted = false` would enter the measurements as evidence that the
compressor writes bad graphs. If the call returns something that is not text, that is a broken adapter
and raises `TypeError` rather than reaching the parser.

### 12.5 The call, and the record of it

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

## 13. What only an audit can decide

Code cannot decide, and must not appear to decide:

- whether a contract or runtime reference was genuinely established;
- whether a retained computation is honestly the same work as before;
- whether a description carries a result that should have been an information node;
- which future computation a piece of information ought to serve;
- whether a removed branch deserved removal.

These are audit questions, answered against the recorded artifacts, and the preflight report answers
them explicitly rather than implying them from counts.

---

## 14. Non-goals

Not a knowledge graph of everything observed. Not a compressed transcript. Not a planner with a memory
store attached. No second metadata dictionary, no hidden history, no compatibility layer with the
previous implementation, no semantic similarity matching, no automatic consumer inference, no
benchmark-specific rules, no training, and no closed-loop or 63-task run without explicit approval.

The previous implementation is not a base to refactor. Nothing is copied from it: not the graph module,
the prompts, the patch protocol, the registry lifecycle, `graph_information`, `required_information`,
`parent_id`, dormant contracts, reference inventories, loaders, parser, renderer or node ids. What may
be reused is data only — the exact frozen `delta_h` slices, the fixed goals and rules, and measurement
definitions that come with deterministic tests.
