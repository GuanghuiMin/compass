# future-graph

A compaction boundary keeps one thing: the computation that remains, together with the information that
remaining computation requires or will produce. Both are nodes in the same graph, and "this contract
serves that computation" is an edge rather than a string repeated in a metadata field.

At each boundary the model regenerates the whole remaining graph from the previous graph and the new
trajectory slice. The candidate is parsed, validated whole, and swapped in whole; if anything is wrong
the previous graph survives untouched.

`SPEC.md` is the contract. This file is the map.

## Why this exists

The previous implementation held future computations in a graph and everything that decided whether
those computations could run — contracts, runtime references, results — in dictionaries and string
lists beside it. Collapsing the resolved past worked well there and is kept. What did not work was
carrying the information forward: whether a contract reached its consumer was a question about list
membership and prompt compliance, so it could only be asked indirectly, and it failed quietly.

Here, information is a node, consumption is an edge, and information with no consumer is removed by a
graph operation rather than by remembering to.

## Layout

```text
src/future_graph/
    schema.py          nodes, payloads, relations
    state_graph.py     the typed graph and its accessors
    validation.py      every invariant, all violations reported together
    lifecycle.py       dead-information collection, atomic replacement
    frontier.py        what is executable, derived from structure
    protocol.py        the block grammar
    parser.py          tolerant surface parsing into a candidate
    rendering.py       the deterministic handover
    regeneration.py    one entry point; the initial graph is the empty-previous case
    artifacts.py       what every boundary records
    metrics.py         measurement definitions, with tests

tests/                 one file per module, plus fixtures
scripts/               frozen replay and its analysis
reports/               preflight findings
prompts/               the regeneration prompt
```

## State of the work

Commit 1 is this specification. Implementation follows in the order set out in `SPEC.md`, each commit
leaving the suite green: schema and graph, then validation and lifecycle, then frontier and rendering,
then protocol and parser, then the regeneration operator, and only then the frozen replay.

No closed-loop run and no 63-task evaluation happen without explicit approval.
