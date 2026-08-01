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
    BEGIN_COMPUTATION, BEGIN_GRAPH, BEGIN_INFO, COMPUTATION_FIELDS, END_COMPUTATION, END_GRAPH,
    END_INFO, EDGE, INFORMATION_FIELDS,
)
from .schema import (
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    ListPayload, MappingPayload, Relation, RuntimeReferencePayload, ScalarPayload, SchemaError,
)
from .state_graph import StateGraph

STRUCTURAL_WORDS = frozenset({BEGIN_GRAPH, END_GRAPH, BEGIN_INFO, END_INFO, BEGIN_COMPUTATION,
                              END_COMPUTATION, EDGE} | {r.name for r in Relation})

SINGLETON_FIELDS = frozenset({"kind", "available", "description", "operation", "value",
                              "runtime-name", "contract-operation"})

NORMALIZATIONS = (
    "markdown fence",
    "indentation",
    "blank line",
    "structural keyword case",
    "field name case",
    "scalar quotes",
    "trailing whitespace",
)


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
    singles: dict[str, tuple[str, int]] = field(default_factory=dict)
    items: list[str] = field(default_factory=list)
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
    out: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if raw.strip().startswith("```"):
            reader.note("markdown fence", number, raw.strip()[:20])
            continue
        stripped = raw.strip()
        if not stripped:
            reader.note("blank line", number, "")
            continue
        if raw != raw.strip():
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

def _read_blocks(lines: list[tuple[int, str]], reader: _Reader) -> None:
    seen_begin = False
    seen_end = False
    current: _Block | None = None

    for number, text in lines:
        head = _keyword(text.split()[0], reader, number)

        if current is None:
            if head == BEGIN_GRAPH:
                if seen_begin:
                    reader.fail(number, f"a second {BEGIN_GRAPH}")
                seen_begin = True
                continue
            if head == END_GRAPH:
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


def _open_block(head: str, text: str, number: int, reader: _Reader) -> _Block | None:
    parts = text.split()
    if len(parts) != 2:
        reader.fail(number, f"{head} takes exactly one id, got {text!r}")
        return _Block("info" if head == BEGIN_INFO else "computation", "?", number)
    return _Block("info" if head == BEGIN_INFO else "computation", parts[1], number)


def _read_edge(text: str, number: int, reader: _Reader) -> None:
    parts = text.split()
    if len(parts) != 4:
        reader.fail(number, f"an edge reads '{EDGE} <source> <relation> <target>', got {text!r}")
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
    first = text.split(maxsplit=1)[0].rstrip(":")
    name = _field_name(first, reader, number)

    if name not in allowed:
        reader.fail(number, f"{first!r} is not a field of {block.kind} blocks")
        return

    if name in ("entry", "argument"):
        _read_pair(block, name, text, number, reader)
        return

    if ":" not in text:
        reader.fail(number, f"{name!r} needs a colon and a value")
        return
    value = text.split(":", 1)[1].strip()
    if not value:
        reader.fail(number, f"{name!r} has no value")
        return

    if name in SINGLETON_FIELDS:
        if name in block.singles:
            reader.fail(number, f"{name!r} is given twice in {block.node_id}")
            return
        block.singles[name] = (value, number)
    elif name == "item":
        block.items.append(value)
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
    """`@i3` is a reference; `"@i3"` is the text. Whether i3 exists never decides which."""
    if raw.startswith("@"):
        return InformationReference(raw[1:])
    return _scalar(raw, reader, line)


# --------------------------------------------------------------------------- construction

def _materialize(reader: _Reader) -> StateGraph | None:
    nodes = []
    declared: set[str] = set()
    for block in reader.blocks:
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
    missing = [f for f in ("kind", "available", "description") if f not in block.singles]
    if missing:
        reader.fail(block.line, f"{block.node_id} has no {', '.join(missing)}")
        return None
    kind_text, kind_line = block.singles["kind"]
    try:
        kind = InformationKind(kind_text)
    except ValueError:
        reader.fail(kind_line, f"{kind_text!r} is not an information kind")
        return None
    available_text, available_line = block.singles["available"]
    if available_text not in ("true", "false"):
        reader.fail(available_line, f"available reads true or false, got {available_text!r}")
        return None

    payload = _payload(block, reader)
    if payload is _FAILED:
        return None
    try:
        return InformationNode(id=block.node_id, kind=kind,
                               description=block.singles["description"][0],
                               available=available_text == "true", payload=payload)
    except SchemaError as err:
        reader.fail(block.line, str(err))
        return None


_FAILED = object()


def _payload(block: _Block, reader: _Reader):
    families = {
        "value": "value" in block.singles,
        "item": bool(block.items),
        "entry": bool(block.entries),
        "runtime-name": "runtime-name" in block.singles,
        "contract": "contract-operation" in block.singles or bool(block.parameters)
                    or bool(block.constraints),
    }
    present = [name for name, yes in families.items() if yes]
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
        if present == ["item"]:
            return ListPayload(tuple(_scalar(v, reader, block.line) for v in block.items))
        if present == ["entry"]:
            return MappingPayload(tuple((k, _scalar(v, reader, line)) for k, v, line
                                        in block.entries))
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


def _computation(block: _Block, reader: _Reader) -> ComputationNode | None:
    if "description" not in block.singles:
        reader.fail(block.line, f"{block.node_id} has no description")
        return None
    arguments = {key: _argument_value(value, reader, line) for key, value, line in block.arguments}
    try:
        return ComputationNode(id=block.node_id, description=block.singles["description"][0],
                               operation=block.singles.get("operation", (None,))[0],
                               arguments=arguments)
    except SchemaError as err:
        reader.fail(block.line, str(err))
        return None
