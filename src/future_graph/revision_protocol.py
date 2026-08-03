"""The block form a local revision is written in.

Relations are attached to the entity whose meaning they describe -- a computation says what it
requires, produces, is refined into and comes after -- so the code builds the directed edge and the
model never encodes a direction. That is not a convenience: asked to write free `EDGE` lines against
a real trajectory, the model wrote eleven of thirteen requirement edges backwards, and every one of
them cost the whole update.

Each relation has exactly one surface. There is no inverse spelling, because a relation with two
ways to write it is the direction problem returning under another name.
"""

from __future__ import annotations

from .schema import InformationKind

BEGIN_REVISION = "BEGIN_REVISION"
END_REVISION = "END_REVISION"

ADD, END_ADD = "ADD", "END_ADD"
REPLACE, END_REPLACE = "REPLACE", "END_REPLACE"
COMPLETE, END_COMPLETE = "COMPLETE", "END_COMPLETE"
INVALIDATE, END_INVALIDATE = "INVALIDATE", "END_INVALIDATE"
REVISE, END_REVISE = "REVISE", "END_REVISE"
REVISE_INFO, END_REVISE_INFO = "REVISE_INFO", "END_REVISE_INFO"
INVALIDATE_INFO, END_INVALIDATE_INFO = "INVALIDATE_INFO", "END_INVALIDATE_INFO"

COMPUTATION, END_COMPUTATION = "COMPUTATION", "END_COMPUTATION"
INFORMATION, END_INFORMATION = "INFORMATION", "END_INFORMATION"
NOW_AVAILABLE, END_NOW_AVAILABLE = "NOW_AVAILABLE", "END_NOW_AVAILABLE"

OPERATIONS = {
    ADD: END_ADD, REPLACE: END_REPLACE, COMPLETE: END_COMPLETE, INVALIDATE: END_INVALIDATE,
    REVISE: END_REVISE, REVISE_INFO: END_REVISE_INFO,
    INVALIDATE_INFO: END_INVALIDATE_INFO,
}

# operations that name an existing node, and how many words their header takes
HEADER_ARITY = {ADD: 1, REPLACE: 2, COMPLETE: 2, INVALIDATE: 2, REVISE: 2, REVISE_INFO: 2,
                INVALIDATE_INFO: 2}

ENTITIES = {COMPUTATION: END_COMPUTATION, INFORMATION: END_INFORMATION,
            NOW_AVAILABLE: END_NOW_AVAILABLE}

COMPUTATION_FIELDS = frozenset({"description", "operation", "argument",
                                "requires", "produces", "refined-into", "after"})
INFORMATION_FIELDS = frozenset({
    "kind", "available", "description",
    "payload-type", "value", "item", "entry", "runtime-name",
    "contract-operation", "contract-parameter", "contract-constraint",
})
NOW_AVAILABLE_FIELDS = INFORMATION_FIELDS - {"available"}
REPLACE_FIELDS = frozenset({"reason-for-replacement", "no-longer-requires", "no-longer-after"})
REVISE_FIELDS = frozenset({"add-requires", "remove-requires", "add-after", "remove-after"})

LIST_FIELDS = frozenset({"requires", "produces", "refined-into", "after",
                         "no-longer-requires", "no-longer-after",
                         "add-requires", "remove-requires", "add-after", "remove-after"})


GRAMMAR = f"""\
Structural words and field names are case-insensitive. Everything else keeps the case it was
written in.

A name with a leading `+` introduces something new. A name without one refers to a node of the
graph you were shown, and must already be there. Labels are renumbered when the revision is read,
so a `+name` may be anything readable: letters, digits and underscores, starting with a letter or
an underscore.

A list field is written on one line, separated by commas: `requires: i1, i4`.

{BEGIN_REVISION}

{ADD}                                  new top-level work
  <computation and information blocks>
{END_ADD}

{REPLACE} <computation>                removes it and everything it is refined into
reason-for-replacement: <text>
no-longer-requires: <information>      information the replaced work needed and nothing needs now
no-longer-after: <computation>         an order the replaced work was under and nothing is now
  <computation and information blocks>
{END_REPLACE}

{COMPLETE} <computation>               the work is done; it and its refinement are gone
  {NOW_AVAILABLE} <information>        something it was going to produce, which now exists
  kind: <kind>                         optional: what it turned out to be
  description: <text>                  optional
  <payload>                            optional
  {END_NOW_AVAILABLE}
{END_COMPLETE}

{INVALIDATE} <computation>             the work will not happen; it establishes nothing
{END_INVALIDATE}

{REVISE} <computation>                 relations only, on a computation that stays
add-requires: <information>
remove-requires: <information>
add-after: <computation>
remove-after: <computation>
{END_REVISE}

{REVISE_INFO} <information>            what it says, on information that stays
kind: <kind>
description: <text>
<payload>
{END_REVISE_INFO}

{INVALIDATE_INFO} <information>        removes it
{END_INVALIDATE_INFO}

{END_REVISION}

A computation block:

{COMPUTATION} <name>
description: <text>
operation: <function>                  a leaf that is ready to run
argument <name> = <scalar or @information>
requires: <information>, ...
produces: <information>, ...
refined-into: <computation>, ...       makes this a refined computation
after: <computation>, ...              this runs after those
{END_COMPUTATION}

An information block:

{INFORMATION} <name>
kind: {' | '.join(k.value for k in InformationKind)}
available: true | false
description: <text>
  and at most one payload, written as one of:
    value: <scalar>
    payload-type: list
    item: <scalar>                     (repeat, zero or more)
    payload-type: mapping
    entry <key> = <scalar>             (repeat, zero or more)
    runtime-name: <name the agent bound>
    contract-operation: <operation>
    contract-parameter: <name>         (repeat)
    contract-constraint: <text>        (repeat)
{END_INFORMATION}

An empty revision -- {BEGIN_REVISION} then {END_REVISION} with nothing between -- says the slice
changed nothing the state has to carry.
"""
