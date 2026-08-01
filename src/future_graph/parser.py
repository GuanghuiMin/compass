"""Reading the block form back into a candidate graph.

Blocks are collected into local records first and the graph is built only once every id an edge names
is known to be declared. That order is the point: handing networkx an edge with an unknown endpoint
would invent the endpoint, and inventing a node is exactly what a parser must never do.

Errors accumulate. A model that got three things wrong should be told three things, and a parse that
found any error returns no graph at all -- half a graph is not a smaller version of a graph.

Surface variation is tolerated only from the list in `NORMALIZATIONS`, and every instance is recorded
so an artifact shows what the text looked like before. Nothing else is repaired: a missing consumer, a
wrong endpoint type or an omitted producer is a question about meaning, and meaning belongs to
validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import (
    BEGIN_COMPUTATION, BEGIN_GRAPH, BEGIN_INFO, COMPUTATION_FIELDS, DECLARED_PAYLOADS,
    END_COMPUTATION, END_GRAPH, END_INFO, EDGE, INFORMATION_FIELDS,
)
from .schema import (
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    ListPayload, MappingPayload, Relation, RuntimeReferencePayload, ScalarPayload, SchemaError,
)
from .state_graph import StateGraph

STRUCTURAL_WORDS = frozenset({BEGIN_GRAPH, END_GRAPH, BEGIN_INFO, END_INFO, BEGIN_COMPUTATION,
                              END_COMPUTATION, EDGE} | {r.name for r in Relation})

SINGLETON_FIELDS = frozenset({"kind", "available", "description", "operation", "payload-type",
                              "value", "runtime-name", "contract-operation"})

NORMALIZATIONS = (
    "markdown fence",
    "indentation",
    "blank line",
    "structural keyword case",
    "field name case",
    "field separator spacing",
    "pair separator spacing",
    "scalar quotes",
    "trailing whitespace",
)

PAIR_FIELDS = frozenset({"entry", "argument"})

# Structural lines carry a fixed number of words. Anything else is a form the grammar does not
# describe, and a trailing word is as likely to be a lost field as it is to be noise.
STRUCTURAL_ARITY = {BEGIN_GRAPH: 1, END_GRAPH: 1, END_INFO: 1, END_COMPUTATION: 1,
                    BEGIN_INFO: 2, BEGIN_COMPUTATION: 2, EDGE: 4}


@dataclass(frozen=True)
class ParseError:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


@dataclass(frozen=True)
class ParseOutcome:
    """A candidate graph, or nothing at all, plus what was tolerated and what was not."""
    graph: StateGraph | None
    normalizations: tuple[str, ...] = ()
    errors: tuple[ParseError, ...] = ()

    @property
    def ok(self) -> bool:
        return self.graph is not None


@dataclass
class _Block:
    kind: str                       # "info" or "computation"
    node_id: str
    line: int
    usable: bool = True             # False once its header was already reported as malformed
    singles: dict[str, tuple[str, int]] = field(default_factory=dict)
    items: list[tuple[str, int]] = field(default_factory=list)
    entries: list[tuple[str, str, int]] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    arguments: list[tuple[str, str, int]] = field(default_factory=list)


class _Reader:
    def __init__(self) -> None:
        self.errors: list[ParseError] = []
        self.normalizations: list[str] = []
        self.blocks: list[_Block] = []
        self.edges: list[tuple[str, Relation, str, int]] = []

    def note(self, kind: str, line: int, detail: str) -> None:
        self.normalizations.append(f"line {line}: {kind} ({detail})")

    def fail(self, line: int, message: str) -> None:
        self.errors.append(ParseError(line, message))


def parse(text: str) -> ParseOutcome:
    reader = _Reader()
    lines = _strip_fences(text, reader)
    _read_blocks(lines, reader)
    # Blocks are turned into nodes even when the scan already found faults, so a bad relation word
    # does not hide a bad information kind three lines above it.
    graph = _materialize(reader)
    return ParseOutcome(graph=graph, normalizations=tuple(reader.normalizations),
                        errors=tuple(reader.errors))


# --------------------------------------------------------------------------- surface

def _strip_fences(text: str, reader: _Reader) -> list[tuple[int, str]]:
    """Drop one matched pair of fences around the whole answer, and nothing else.

    Skipping every fence-like line would silently swallow one in the middle of a graph, which is a
    structural form the grammar does not describe and should be refused rather than absorbed.
    """
    raw_lines = list(enumerate(text.splitlines(), start=1))
    content = [i for i, (_, raw) in enumerate(raw_lines) if raw.strip()]
    skip: set[int] = set()
    if len(content) >= 2:
        first, last = content[0], content[-1]
        if raw_lines[first][1].strip().startswith("```") \
                and raw_lines[last][1].strip().startswith("```"):
            skip = {first, last}
            reader.note("markdown fence", raw_lines[first][0], "one surrounding pair")

    out: list[tuple[int, str]] = []
    for index, (number, raw) in enumerate(raw_lines):
        if index in skip:
            continue
        stripped = raw.strip()
        if not stripped:
            reader.note("blank line", number, "")
            continue
        if raw != stripped:
            detail = "indentation" if raw.lstrip() != raw else "trailing whitespace"
            reader.note(detail, number, repr(raw[:12]))
        out.append((number, stripped))
    return out


def _keyword(word: str, reader: _Reader, line: int) -> str:
    upper = word.upper()
    if upper in STRUCTURAL_WORDS and word != upper:
        reader.note("structural keyword case", line, word)
    return upper


def _field_name(word: str, reader: _Reader, line: int) -> str:
    lower = word.lower()
    if word != lower:
        reader.note("field name case", line, word)
    return lower


# --------------------------------------------------------------------------- blocks

def _arity(head: str, text: str, number: int, reader: _Reader) -> bool:
    want = STRUCTURAL_ARITY[head]
    got = len(text.split())
    if got != want:
        reader.fail(number, f"{head} takes {want} word{'s' if want > 1 else ''}, got {text!r}")
        return False
    return True


def _read_blocks(lines: list[tuple[int, str]], reader: _Reader) -> None:
    seen_begin = False
    seen_end = False
    current: _Block | None = None

    for number, text in lines:
        head = _keyword(text.split()[0], reader, number)

        if current is None:
            if head == BEGIN_GRAPH:
                _arity(head, text, number, reader)
                if seen_begin:
                    reader.fail(number, f"a second {BEGIN_GRAPH}")
                seen_begin = True
                continue
            if head == END_GRAPH:
                _arity(head, text, number, reader)
                if seen_end:
                    reader.fail(number, f"a second {END_GRAPH}")
                seen_end = True
                continue
            if not seen_begin:
                reader.fail(number, f"text before {BEGIN_GRAPH}: {text!r}")
                continue
            if seen_end:
                reader.fail(number, f"text after {END_GRAPH}: {text!r}")
                continue
            if head in (BEGIN_INFO, BEGIN_COMPUTATION):
                current = _open_block(head, text, number, reader)
                continue
            if head == EDGE:
                _read_edge(text, number, reader)
                continue
            reader.fail(number, f"{text.split()[0]!r} is not a known statement")
            continue

        if head in (END_INFO, END_COMPUTATION):
            expected = END_INFO if current.kind == "info" else END_COMPUTATION
            if head != expected:
                reader.fail(number, f"{current.node_id} is closed by {head}, expected {expected}")
            _arity(head, text, number, reader)
            reader.blocks.append(current)
            current = None
            continue
        if head in (BEGIN_INFO, BEGIN_COMPUTATION, EDGE, END_GRAPH, BEGIN_GRAPH):
            reader.fail(number, f"{current.node_id} was never closed before {head}")
            reader.blocks.append(current)
            current = None
            continue
        _read_field(current, text, number, reader)

    if current is not None:
        reader.fail(current.line, f"{current.node_id} is never closed")
    if not seen_begin:
        reader.fail(0, f"no {BEGIN_GRAPH}")
    if not seen_end:
        reader.fail(0, f"no {END_GRAPH}")


def _open_block(head: str, text: str, number: int, reader: _Reader) -> _Block:
    kind = "info" if head == BEGIN_INFO else "computation"
    parts = text.split()
    if not _arity(head, text, number, reader):
        # Keep reading its fields so their faults are seen too, but do not build a node from a
        # header that was never understood.
        return _Block(kind, parts[1] if len(parts) > 1 else "?", number, usable=False)
    return _Block(kind, parts[1], number)


def _read_edge(text: str, number: int, reader: _Reader) -> None:
    parts = text.split()
    if not _arity(EDGE, text, number, reader):
        return
    _, source, word, target = parts
    name = _keyword(word, reader, number)
    try:
        relation = Relation[name]
    except KeyError:
        reader.fail(number, f"{word!r} is not a relation")
        return
    reader.edges.append((source, relation, target, number))


def _read_field(block: _Block, text: str, number: int, reader: _Reader) -> None:
    allowed = INFORMATION_FIELDS if block.kind == "info" else COMPUTATION_FIELDS
    head = text.split(maxsplit=1)[0]

    if head.lower() in PAIR_FIELDS:
        name = _field_name(head, reader, number)
        if name not in allowed:
            reader.fail(number, f"{head!r} is not a field of {block.kind} blocks")
            return
        _read_pair(block, name, text, number, reader)
        return

    if ":" not in text:
        reader.fail(number, f"{head!r} is not a field of {block.kind} blocks")
        return
    raw_name, raw_value = text.split(":", 1)
    if raw_name != raw_name.rstrip() or raw_value != " " + raw_value.strip():
        reader.note("field separator spacing", number, text[:20])
    name = _field_name(raw_name.strip(), reader, number)

    if name not in allowed:
        reader.fail(number, f"{raw_name.strip()!r} is not a field of {block.kind} blocks")
        return

    value = raw_value.strip()
    if not value:
        reader.fail(number, f"{name!r} has no value")
        return

    if name in SINGLETON_FIELDS:
        if name in block.singles:
            reader.fail(number, f"{name!r} is given twice in {block.node_id}")
            return
        block.singles[name] = (value, number)
    elif name == "item":
        block.items.append((value, number))
    elif name == "contract-parameter":
        block.parameters.append(value)
    elif name == "contract-constraint":
        block.constraints.append(value)


def _read_pair(block: _Block, name: str, text: str, number: int, reader: _Reader) -> None:
    body = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
    if "=" not in body:
        reader.fail(number, f"{name} reads '{name} <name> = <value>', got {text!r}")
        return
    key, value = body.split("=", 1)
    key, value = key.strip(), value.strip()
    if not key or not value:
        reader.fail(number, f"{name} reads '{name} <name> = <value>', got {text!r}")
        return
    if body != f"{key} = {value}":
        reader.note("pair separator spacing", number, text[:20])
    target = block.entries if name == "entry" else block.arguments
    if any(existing == key for existing, _, _ in target):
        reader.fail(number, f"{name} {key!r} is given twice in {block.node_id}")
        return
    target.append((key, value, number))


# --------------------------------------------------------------------------- values

def _scalar(raw: str, reader: _Reader, line: int):
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        reader.note("scalar quotes", line, raw[:12])
        return raw[1:-1]
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _argument_value(raw: str, reader: _Reader, line: int):
    """`@i3` is a reference; `"@i3"` is the text. Whether i3 exists never decides which.

    A reference whose name is not shaped like an information id is a parse error. No text a model can
    write may escape as an exception from this module.
    """
    if raw.startswith("@"):
        name = raw[1:]
        if name != name.strip():
            # The id check strips, so `@ i3` would quietly become a reference to i3. Whitespace
            # inside a reference is not on the list of tolerated surface.
            reader.fail(line, f"{raw!r} is not a reference: an id carries no spaces")
            return _FAILED
        try:
            return InformationReference(name)
        except SchemaError as err:
            reader.fail(line, f"{raw!r} is not a reference: {err}")
            return _FAILED
    return _scalar(raw, reader, line)


# --------------------------------------------------------------------------- construction

def _materialize(reader: _Reader) -> StateGraph | None:
    nodes = []
    declared: set[str] = set()
    for block in reader.blocks:
        if not block.usable:
            continue
        node = _information(block, reader) if block.kind == "info" else _computation(block, reader)
        if node is None:
            continue
        if node.id in declared:
            reader.fail(block.line, f"{node.id} is declared twice in one snapshot")
            continue
        declared.add(node.id)
        nodes.append(node)

    for source, relation, target, line in reader.edges:
        for endpoint in (source, target):
            if endpoint not in declared:
                reader.fail(line, f"edge names {endpoint!r}, which no block declares")

    if reader.errors:
        return None

    graph = StateGraph()
    for node in nodes:
        graph.add(node)
    for source, relation, target, _ in reader.edges:
        graph.add_edge(source, relation, target)
    return graph


def _information(block: _Block, reader: _Reader) -> InformationNode | None:
    """Report everything wrong with this block that can be decided on its own.

    A bad `kind` says nothing about whether `available` is a word this grammar knows, so finding the
    first is no reason to stop looking for the second.
    """
    failed = False
    missing = [f for f in ("kind", "available", "description") if f not in block.singles]
    if missing:
        reader.fail(block.line, f"{block.node_id} has no {', '.join(missing)}")
        failed = True

    kind = None
    if "kind" in block.singles:
        kind_text, kind_line = block.singles["kind"]
        try:
            kind = InformationKind(kind_text)
        except ValueError:
            reader.fail(kind_line, f"{kind_text!r} is not an information kind")
            failed = True

    available = None
    if "available" in block.singles:
        available_text, available_line = block.singles["available"]
        if available_text in ("true", "false"):
            available = available_text == "true"
        else:
            reader.fail(available_line, f"available reads true or false, got {available_text!r}")
            failed = True

    payload = _payload(block, reader)
    if payload is _FAILED or failed:
        return None
    try:
        return InformationNode(id=block.node_id, kind=kind,
                               description=block.singles["description"][0],
                               available=available, payload=payload)
    except SchemaError as err:
        reader.fail(block.line, str(err))
        return None


_FAILED = object()


def _payload(block: _Block, reader: _Reader):
    """One payload per block, and a list or mapping only when the block says which it is."""
    families = {
        "value": "value" in block.singles,
        "item": bool(block.items),
        "entry": bool(block.entries),
        "runtime-name": "runtime-name" in block.singles,
        "contract": "contract-operation" in block.singles or bool(block.parameters)
                    or bool(block.constraints),
    }
    declared = block.singles.get("payload-type")
    present = [name for name, yes in families.items() if yes]

    if declared is not None:
        return _declared_payload(block, reader, declared, present)
    if "item" in present or "entry" in present:
        wanted = "list" if "item" in present else "mapping"
        reader.fail(block.line,
                    f"{block.node_id} has {'item' if wanted == 'list' else 'entry'} lines without "
                    f"'payload-type: {wanted}'; nothing here decides that for you")
        return _FAILED
    if not present:
        return None
    if len(present) > 1:
        reader.fail(block.line,
                    f"{block.node_id} gives {' and '.join(sorted(present))}; a payload is one kind")
        return _FAILED
    try:
        if present == ["value"]:
            text, line = block.singles["value"]
            return ScalarPayload(_scalar(text, reader, line))
        if present == ["runtime-name"]:
            return RuntimeReferencePayload(block.singles["runtime-name"][0])
        if "contract-operation" not in block.singles:
            reader.fail(block.line, f"{block.node_id} has contract details but no "
                                    "contract-operation")
            return _FAILED
        return ContractPayload(block.singles["contract-operation"][0],
                               tuple(block.parameters), tuple(block.constraints))
    except SchemaError as err:
        reader.fail(block.line, str(err))
        return _FAILED


def _declared_payload(block: _Block, reader: _Reader, declared: tuple[str, int],
                      present: list[str]):
    text, line = declared
    if text not in DECLARED_PAYLOADS:
        reader.fail(line, f"payload-type reads {' or '.join(DECLARED_PAYLOADS)}, got {text!r}")
        return _FAILED
    wanted = "item" if text == "list" else "entry"
    intruders = sorted(set(present) - {wanted})
    if intruders:
        reader.fail(block.line, f"{block.node_id} declares a {text} and also gives "
                                f"{' and '.join(intruders)}")
        return _FAILED
    try:
        if text == "list":
            return ListPayload(tuple(_scalar(v, reader, line) for v, line in block.items))
        return MappingPayload(tuple((k, _scalar(v, reader, line)) for k, v, line in block.entries))
    except SchemaError as err:
        reader.fail(block.line, str(err))
        return _FAILED


def _computation(block: _Block, reader: _Reader) -> ComputationNode | None:
    failed = "description" not in block.singles
    if failed:
        reader.fail(block.line, f"{block.node_id} has no description")
    arguments = {}
    for key, value, line in block.arguments:
        parsed = _argument_value(value, reader, line)
        if parsed is _FAILED:
            failed = True
        else:
            arguments[key] = parsed
    if failed:
        return None
    try:
        return ComputationNode(id=block.node_id, description=block.singles["description"][0],
                               operation=block.singles.get("operation", (None,))[0],
                               arguments=arguments)
    except SchemaError as err:
        reader.fail(block.line, str(err))
        return None
