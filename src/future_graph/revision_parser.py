"""Reading a local revision back from the block form.

Surface tolerance is the same as the graph parser's and for the same reasons: one matched pair of
fences, indentation, blank lines, keyword and field-name case, separator spacing, scalar quotes,
trailing whitespace -- and a missing block terminator, because the next top-level statement proves
the block before it ended and losing a whole revision to an absent `END_` line taught nothing.

Nothing is repaired beyond that. A field the grammar does not know, a name that is neither an
anchor nor a `+label`, an operation that does not close: each is an error and the revision produces
nothing, because half a revision is not a smaller revision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import ParseError, _FAILED, _payload, _scalar, _strip_fences, _Reader
from .revision_protocol import (
    ADD, BEGIN_REVISION, COMPLETE, COMPUTATION, END_REVISION, ENTITIES, HEADER_ARITY,
    INFORMATION, INFORMATION_FIELDS, INVALIDATE, INVALIDATE_INFO, LIST_FIELDS, NOW_AVAILABLE,
    NOW_AVAILABLE_FIELDS, OPERATIONS, REPLACE, REPLACE_FIELDS, REVISE, REVISE_FIELDS,
    REVISE_INFO, COMPUTATION_FIELDS,
)
from .revision import (
    Add, Complete, Invalidate, InvalidateInformation, NewComputation, NewInformation,
    NowAvailable, Replace, Revision, ReviseComputation, ReviseInformation,
)
from .schema import InformationKind, InformationNode, InformationReference, SchemaError

LABEL = re.compile(r"\+?[A-Za-z_][A-Za-z0-9_]*")
SINGLE = frozenset({"description", "operation", "kind", "available", "reason-for-replacement",
                    "payload-type", "value", "runtime-name", "contract-operation"})


@dataclass(frozen=True)
class RevisionOutcome:
    revision: Revision | None
    normalizations: tuple[str, ...] = ()
    errors: tuple[ParseError, ...] = ()

    @property
    def ok(self) -> bool:
        return self.revision is not None


@dataclass
class _Entity:
    kind: str                       # COMPUTATION | INFORMATION | NOW_AVAILABLE
    name: str
    line: int
    usable: bool = True
    singles: dict = field(default_factory=dict)
    lists: dict = field(default_factory=dict)
    items: list = field(default_factory=list)
    entries: list = field(default_factory=list)
    parameters: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    arguments: list = field(default_factory=list)


@dataclass
class _Op:
    kind: str
    anchor: str | None
    line: int
    usable: bool = True
    singles: dict = field(default_factory=dict)
    lists: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)
    # REVISE_INFO carries information fields directly, so an operation reads a payload the same
    # way a block does.
    items: list = field(default_factory=list)
    entries: list = field(default_factory=list)
    parameters: list = field(default_factory=list)
    constraints: list = field(default_factory=list)


def parse_revision(text: str) -> RevisionOutcome:
    reader = _Reader()
    lines = _strip_fences(text, reader)
    ops = _read(lines, reader)
    revision = _materialize(ops, reader)
    return RevisionOutcome(revision=revision, normalizations=tuple(reader.normalizations),
                           errors=tuple(reader.errors))


def _word(text: str) -> str:
    return text.split()[0]


def _read(lines, reader: _Reader) -> list[_Op]:
    ops: list[_Op] = []
    seen_begin = seen_end = False
    op: _Op | None = None
    entity: _Entity | None = None

    for number, text in lines:
        raw_head = _word(text)
        head = raw_head.upper()
        if head != raw_head and (head in OPERATIONS or head in ENTITIES
                                 or head in {BEGIN_REVISION, END_REVISION}
                                 or head in set(OPERATIONS.values())
                                 or head in set(ENTITIES.values())):
            reader.note("structural keyword case", number, raw_head)

        if entity is not None:
            if head == ENTITIES[entity.kind]:
                op.entities.append(entity)
                entity = None
                continue
            if head in ENTITIES or head in OPERATIONS or head in set(OPERATIONS.values()) \
                    or head == END_REVISION:
                reader.note("block terminator", number, f"{entity.name} closed by {head}")
                op.entities.append(entity)
                entity = None
            else:
                _field(entity, _allowed_fields(entity.kind), text, number, reader)
                continue

        if op is not None:
            if head == OPERATIONS.get(op.kind):
                ops.append(op)
                op = None
                continue
            if head in ENTITIES:
                allowed = {ADD: (COMPUTATION, INFORMATION), REPLACE: (COMPUTATION, INFORMATION),
                           COMPLETE: (NOW_AVAILABLE,)}.get(op.kind, ())
                if head in allowed:
                    entity = _open_entity(head, text, number, reader)
                else:
                    reader.fail(number, f"{head} is not part of a {op.kind} operation")
                continue
            if head in OPERATIONS or head == END_REVISION:
                reader.note("block terminator", number, f"{op.kind} closed by {head}")
                ops.append(op)
                op = None
            else:
                _field(op, _operation_fields(op.kind), text, number, reader)
                continue

        if head == BEGIN_REVISION:
            if seen_begin:
                reader.fail(number, f"a second {BEGIN_REVISION}")
            seen_begin = True
            continue
        if head == END_REVISION:
            if seen_end:
                reader.fail(number, f"a second {END_REVISION}")
            seen_end = True
            continue
        if not seen_begin:
            reader.fail(number, f"text before {BEGIN_REVISION}: {text!r}")
            continue
        if seen_end:
            reader.fail(number, f"text after {END_REVISION}: {text!r}")
            continue
        if head in OPERATIONS:
            op = _open_operation(head, text, number, reader)
            continue
        reader.fail(number, f"{raw_head!r} is not a known statement")

    if entity is not None:
        reader.note("block terminator", entity.line, f"{entity.name} closed at the end")
        op.entities.append(entity)
    if op is not None:
        reader.note("block terminator", op.line, f"{op.kind} closed at the end")
        ops.append(op)
    if not seen_begin:
        reader.fail(0, f"no {BEGIN_REVISION}")
    if not seen_end:
        reader.fail(0, f"no {END_REVISION}")
    return ops


def _allowed_fields(kind: str) -> frozenset:
    if kind == COMPUTATION:
        return COMPUTATION_FIELDS
    if kind == INFORMATION:
        return INFORMATION_FIELDS
    return NOW_AVAILABLE_FIELDS


def _operation_fields(kind: str) -> frozenset:
    if kind == REPLACE:
        return REPLACE_FIELDS
    if kind == REVISE:
        return REVISE_FIELDS
    if kind == REVISE_INFO:
        return INFORMATION_FIELDS - {"available"}
    return frozenset()


def _open_operation(head: str, text: str, number: int, reader: _Reader) -> _Op:
    parts = text.split()
    want = HEADER_ARITY[head]
    if len(parts) != want:
        reader.fail(number, f"{head} takes {want} word{'s' if want > 1 else ''}, got {text!r}")
        return _Op(head, parts[1] if len(parts) > 1 else None, number, usable=False)
    anchor = parts[1] if want == 2 else None
    if anchor is not None and not LABEL.fullmatch(anchor):
        reader.fail(number, f"{anchor!r} is not a name")
        return _Op(head, anchor, number, usable=False)
    if anchor is not None and anchor.startswith("+"):
        reader.fail(number, f"{head} names a node of the graph you were shown, and {anchor!r} "
                            "declares a new one")
        return _Op(head, anchor, number, usable=False)
    return _Op(head, anchor, number)


def _open_entity(head: str, text: str, number: int, reader: _Reader) -> _Entity:
    parts = text.split()
    if len(parts) != 2:
        reader.fail(number, f"{head} takes 2 words, got {text!r}")
        return _Entity(head, parts[1] if len(parts) > 1 else "?", number, usable=False)
    name = parts[1]
    if not LABEL.fullmatch(name):
        reader.fail(number, f"{name!r} is not a name")
        return _Entity(head, name, number, usable=False)
    if head in (COMPUTATION, INFORMATION) and not name.startswith("+"):
        reader.fail(number, f"{head} declares something new, and {name!r} has no leading +")
        return _Entity(head, name, number, usable=False)
    if head == NOW_AVAILABLE and name.startswith("+"):
        reader.fail(number, f"{head} names information already in the graph, and {name!r} "
                            "declares a new one")
        return _Entity(head, name, number, usable=False)
    return _Entity(head, name, number)


def _field(target, allowed: frozenset, text: str, number: int, reader: _Reader) -> None:
    head = text.split(maxsplit=1)[0]
    if head.lower() in ("argument", "entry"):
        name = head.lower()
        if name not in allowed:
            reader.fail(number, f"{head!r} is not a field here")
            return
        _pair(target, name, text, number, reader)
        return
    if ":" not in text:
        reader.fail(number, f"{head!r} is not a field here")
        return
    raw_name, raw_value = text.split(":", 1)
    if raw_name != raw_name.rstrip() or raw_value != " " + raw_value.strip():
        reader.note("field separator spacing", number, text[:20])
    name = raw_name.strip().lower()
    if raw_name.strip() != name:
        reader.note("field name case", number, raw_name.strip())
    if name not in allowed:
        reader.fail(number, f"{raw_name.strip()!r} is not a field here")
        return
    value = raw_value.strip()
    if not value:
        reader.fail(number, f"{name!r} has no value")
        return
    if name in LIST_FIELDS:
        names = [part.strip() for part in value.split(",") if part.strip()]
        bad = [n for n in names if not LABEL.fullmatch(n)]
        for n in bad:
            reader.fail(number, f"{n!r} is not a name")
        target.lists.setdefault(name, []).extend(n for n in names if n not in bad)
        return
    if name in SINGLE:
        if name in target.singles:
            reader.fail(number, f"{name!r} is given twice")
            return
        target.singles[name] = (value, number)
    elif name == "item":
        target.items.append((value, number))
    elif name == "contract-parameter":
        target.parameters.append(value)
    elif name == "contract-constraint":
        target.constraints.append(value)


def _pair(target, name: str, text: str, number: int, reader: _Reader) -> None:
    body = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
    if "=" not in body:
        reader.fail(number, f"{name} reads '{name} <name> = <value>', got {text!r}")
        return
    key, value = (part.strip() for part in body.split("=", 1))
    if not key or not value:
        reader.fail(number, f"{name} reads '{name} <name> = <value>', got {text!r}")
        return
    if body != f"{key} = {value}":
        reader.note("pair separator spacing", number, text[:20])
    into = target.entries if name == "entry" else target.arguments
    if any(existing == key for existing, _, _ in into):
        reader.fail(number, f"{name} {key!r} is given twice")
        return
    into.append((key, value, number))


# --------------------------------------------------------------------------- construction

def _materialize(ops: list[_Op], reader: _Reader) -> Revision | None:
    operations = []
    for op in ops:
        if not op.usable:
            continue
        built = _operation(op, reader)
        if built is not None:
            operations.append(built)
    if reader.errors:
        return None
    return Revision(tuple(operations))


def _operation(op: _Op, reader: _Reader):
    computations, information, now_available = [], [], []
    for entity in op.entities:
        if not entity.usable:
            continue
        if entity.kind == COMPUTATION:
            built = _computation(entity, reader)
            if built is not None:
                computations.append(built)
        elif entity.kind == INFORMATION:
            built = _information(entity, reader)
            if built is not None:
                information.append(built)
        else:
            built = _now_available(entity, reader)
            if built is not None:
                now_available.append(built)

    if op.kind == ADD:
        return Add(tuple(computations), tuple(information))
    if op.kind == REPLACE:
        reason = op.singles.get("reason-for-replacement", ("", 0))[0]
        if not reason:
            reader.fail(op.line, f"{REPLACE} needs a reason-for-replacement")
            return None
        return Replace(op.anchor, reason, tuple(computations), tuple(information),
                       tuple(op.lists.get("no-longer-requires", ())),
                       tuple(op.lists.get("no-longer-after", ())))
    if op.kind == COMPLETE:
        return Complete(op.anchor, tuple(now_available))
    if op.kind == INVALIDATE:
        return Invalidate(op.anchor)
    if op.kind == REVISE:
        return ReviseComputation(op.anchor,
                                 tuple(op.lists.get("add-requires", ())),
                                 tuple(op.lists.get("remove-requires", ())),
                                 tuple(op.lists.get("add-after", ())),
                                 tuple(op.lists.get("remove-after", ())))
    if op.kind == REVISE_INFO:
        kind = _kind(op.singles, reader, op.line)
        if kind is _FAILED:
            return None
        payload = _payload(op, reader)
        if payload is _FAILED:
            return None
        given = payload is not None
        return ReviseInformation(op.anchor, kind,
                                 op.singles.get("description", (None,))[0],
                                 payload, payload_given=given)
    return InvalidateInformation(op.anchor)


def _kind(singles, reader, line):
    if "kind" not in singles:
        return None
    text, where = singles["kind"]
    try:
        return InformationKind(text)
    except ValueError:
        reader.fail(where, f"{text!r} is not an information kind")
        return _FAILED


def _computation(entity: _Entity, reader: _Reader) -> NewComputation | None:
    if "description" not in entity.singles:
        reader.fail(entity.line, f"{entity.name} has no description")
        return None
    arguments = {}
    failed = False
    for key, value, line in entity.arguments:
        if value.startswith("@"):
            name = value[1:]
            if name != name.strip() or not LABEL.fullmatch(name):
                reader.fail(line, f"{value!r} is not a reference")
                failed = True
                continue
            arguments[key] = InformationReference.__new__(InformationReference)
            object.__setattr__(arguments[key], "information_id", name)
        else:
            arguments[key] = _scalar(value, reader, line)
    if failed:
        return None
    return NewComputation(
        label=entity.name, description=entity.singles["description"][0],
        operation=entity.singles.get("operation", (None,))[0], arguments=arguments,
        requires=tuple(entity.lists.get("requires", ())),
        produces=tuple(entity.lists.get("produces", ())),
        refined_into=tuple(entity.lists.get("refined-into", ())),
        after=tuple(entity.lists.get("after", ())))


def _information(entity: _Entity, reader: _Reader) -> NewInformation | None:
    missing = [f for f in ("kind", "available", "description") if f not in entity.singles]
    if missing:
        reader.fail(entity.line, f"{entity.name} has no {', '.join(missing)}")
        return None
    kind = _kind(entity.singles, reader, entity.line)
    if kind is _FAILED:
        return None
    available_text, where = entity.singles["available"]
    if available_text not in ("true", "false"):
        reader.fail(where, f"available reads true or false, got {available_text!r}")
        return None
    payload = _payload(entity, reader)
    if payload is _FAILED:
        return None
    available = available_text == "true"
    # Built as the node it will become, so a declaration the schema would never accept is refused
    # here rather than surviving as a revision that cannot be applied.
    try:
        InformationNode(id="i1", kind=kind, description=entity.singles["description"][0],
                        available=available, payload=payload)
    except SchemaError as err:
        reader.fail(entity.line, str(err))
        return None
    return NewInformation(entity.name, kind, entity.singles["description"][0], available, payload)


def _now_available(entity: _Entity, reader: _Reader) -> NowAvailable | None:
    kind = _kind(entity.singles, reader, entity.line)
    if kind is _FAILED:
        return None
    payload = _payload(entity, reader)
    if payload is _FAILED:
        return None
    return NowAvailable(entity.name, kind, entity.singles.get("description", (None,))[0], payload)
