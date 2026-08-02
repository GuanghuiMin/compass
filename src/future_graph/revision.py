"""A local revision: what the model says changed, and what the code does with it.

The model names the regions the new trajectory affected and writes the local plan that replaces
them. Everything it does not name is preserved by code, so the size of an answer tracks the size of
the change rather than the size of the graph. That is the whole reason this exists: under complete
rewriting the failure surface of a boundary grew with the state, and a single mechanical slip cost
the entire update.

Nothing here interprets a trajectory or infers a correspondence. It resolves anchors, computes the
closure of a removed region, checks that every relation crossing that region's boundary was
accounted for, and refuses when it was not -- because a relation the model forgot and a relation the
model meant to drop look identical, and only one of them is safe to act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union

from .schema import (
    ArgumentValue, ComputationNode, InformationKind, InformationNode, InformationPayload,
    InformationReference, Relation, SchemaError,
)
from .state_graph import Edge, StateGraph


class RevisionError(ValueError):
    """A revision that cannot be applied, reported the way a validation violation is."""


@dataclass(frozen=True)
class Fault:
    """One reason a revision was refused, in the same shape as a validation violation."""
    code: str
    message: str
    nodes: tuple[str, ...] = ()

    def __str__(self) -> str:
        where = f" [{', '.join(self.nodes)}]" if self.nodes else ""
        return f"{self.code}: {self.message}{where}"


# --------------------------------------------------------------------------- what the model writes

@dataclass(frozen=True)
class NewComputation:
    """A computation the revision introduces. `label` carries the `+`, so it cannot read as an
    anchor into the previous graph."""
    label: str
    description: str
    operation: str | None = None
    arguments: Mapping[str, ArgumentValue] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    refined_into: tuple[str, ...] = ()
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class NewInformation:
    label: str
    kind: InformationKind
    description: str
    available: bool
    payload: InformationPayload | None = None


@dataclass(frozen=True)
class NowAvailable:
    """An existing unavailable result the completed work established.

    The only way availability moves from false to true, and the only place the kind and payload of
    a promised result may become the confirmed thing it turned into.
    """
    anchor: str
    kind: InformationKind | None = None
    description: str | None = None
    payload: InformationPayload | None = None


@dataclass(frozen=True)
class Add:
    computations: tuple[NewComputation, ...] = ()
    information: tuple[NewInformation, ...] = ()


@dataclass(frozen=True)
class Replace:
    anchor: str
    reason: str
    computations: tuple[NewComputation, ...] = ()
    information: tuple[NewInformation, ...] = ()
    no_longer_requires: tuple[str, ...] = ()
    no_longer_after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Complete:
    anchor: str
    now_available: tuple[NowAvailable, ...] = ()


@dataclass(frozen=True)
class Invalidate:
    anchor: str


@dataclass(frozen=True)
class ReviseComputation:
    anchor: str
    add_requires: tuple[str, ...] = ()
    remove_requires: tuple[str, ...] = ()
    add_after: tuple[str, ...] = ()
    remove_after: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviseInformation:
    anchor: str
    kind: InformationKind | None = None
    description: str | None = None
    payload: InformationPayload | None = None
    payload_given: bool = False


@dataclass(frozen=True)
class InvalidateInformation:
    anchor: str


Operation = Union[Add, Replace, Complete, Invalidate, ReviseComputation, ReviseInformation,
                  InvalidateInformation]


@dataclass(frozen=True)
class Revision:
    operations: tuple[Operation, ...] = ()

    @property
    def is_empty(self) -> bool:
        """No operations at all: the slice changed nothing the state has to carry.

        A legal answer, and one worth being able to give -- without it a boundary that established
        nothing structural would force the model to invent a change.
        """
        return not self.operations


# --------------------------------------------------------------------------- what the code did

@dataclass(frozen=True)
class EdgeChange:
    action: str            # "removed" or "added"
    source: str
    relation: Relation
    target: str


@dataclass(frozen=True)
class NodeRemoval:
    node_id: str
    reason: str            # "affected_region" | "region_internal" | "invalidated_information"


@dataclass(frozen=True)
class CompletionChange:
    """What a `now-available` declaration turned into, kept apart from ordinary derivation.

    The semantic fact -- that the completed work established this -- came from the model. The code
    only carries it out, and the record has to show which of the two happened.
    """
    action: str            # "became_available" | "producer_removed" | "content_replaced"
                           # | "provenance_materialized"
    node_id: str
    detail: str = ""


@dataclass(frozen=True)
class RevisionChanges:
    affected_roots: tuple[str, ...] = ()
    touched_nodes: tuple[str, ...] = ()
    removed_nodes: tuple[NodeRemoval, ...] = ()
    removed_edges: tuple[EdgeChange, ...] = ()
    replacement_boundary_changes: tuple[EdgeChange, ...] = ()
    completion_changes: tuple[CompletionChange, ...] = ()
    id_map: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Applied:
    graph: StateGraph | None
    changes: RevisionChanges
    faults: tuple[Fault, ...] = ()


# --------------------------------------------------------------------------- application

def apply_revision(previous: StateGraph, revision: Revision) -> Applied:
    """Build the graph the revision describes, or report why it cannot be built.

    Faults are collected rather than raised on the first one, for the reason validation collects
    them: a refusal that names one problem teaches nothing about the others.
    """
    worker = _Application(previous, revision)
    return worker.run()


class _Application:
    def __init__(self, previous: StateGraph, revision: Revision) -> None:
        self.previous = previous
        self.revision = revision
        self.faults: list[Fault] = []
        self.computation_ids = {c.id for c in previous.computations}
        self.information_ids = {i.id for i in previous.information}

    def fail(self, code: str, message: str, *nodes: str) -> None:
        self.faults.append(Fault(code, message, tuple(nodes)))

    # ------------------------------------------------------------------ entry
    def run(self) -> Applied:
        ops = self.revision.operations
        if not ops:
            # Not "a revision that happens to change nothing" but the same graph, the same object,
            # untouched. Rebuilding it would re-derive its interfaces and renumber its nodes, and a
            # no-op that renumbers is not a no-op.
            return Applied(self.previous, RevisionChanges(), ())
        roots = self._region_roots(ops)
        self._check_operation_targets(ops)
        region = self._region_closure(roots)
        if self.faults:
            return Applied(None, RevisionChanges(), tuple(self.faults))

        invalidated = self._invalidated_information(ops, region)
        internal = self._region_internal_information(ops, region) - invalidated
        removed_information = internal | invalidated

        self._check_crossings(ops, region, removed_information)
        completion = self._plan_completions(ops, region)
        if self.faults:
            return Applied(None, RevisionChanges(), tuple(self.faults))

        return self._assemble(ops, roots, region, internal, invalidated, completion)

    # ------------------------------------------------------------------ regions
    def _region_roots(self, ops) -> dict[str, Operation]:
        roots: dict[str, Operation] = {}
        for op in ops:
            if not isinstance(op, (Replace, Complete, Invalidate)):
                continue
            if op.anchor not in self.computation_ids:
                self.fail("unknown_anchor",
                          f"{op.anchor!r} names no computation in the previous graph", op.anchor)
                continue
            if op.anchor in roots:
                self.fail("conflicting_operations",
                          f"{op.anchor!r} is the root of more than one operation", op.anchor)
                continue
            roots[op.anchor] = op
        for anchor in roots:
            for other in roots:
                if other != anchor and other in self.previous.refinement_ancestors_of(anchor):
                    self.fail("overlapping_affected_regions",
                              f"{other!r} is refined into {anchor!r}, and operations name the "
                              "highest disjoint roots", other, anchor)
        return roots

    def _check_operation_targets(self, ops) -> None:
        seen: dict[str, str] = {}
        for op in ops:
            anchor = getattr(op, "anchor", None)
            if anchor is None:
                continue
            kind = type(op).__name__
            if anchor in seen:
                self.fail("conflicting_operations",
                          f"{anchor!r} is named by {seen[anchor]} and {kind}", anchor)
            seen[anchor] = kind
            if isinstance(op, (ReviseInformation, InvalidateInformation)):
                if anchor not in self.information_ids:
                    self.fail("unknown_anchor",
                              f"{anchor!r} names no information in the previous graph", anchor)
            elif isinstance(op, ReviseComputation):
                if anchor not in self.computation_ids:
                    self.fail("unknown_anchor",
                              f"{anchor!r} names no computation in the previous graph", anchor)

    def _region_closure(self, roots) -> set[str]:
        inside: set[str] = set()
        for anchor in roots:
            inside.add(anchor)
            inside.update(self.previous.refinement_descendants_of(anchor))
        return inside

    def _region_internal_information(self, ops, region: set[str]) -> set[str]:
        """Intermediate state of a subtree that is going away, removed with it.

        A node qualifies only if the removed work was going to produce it: something the region
        merely consumed is an input, and the model still has to say whether the replacement needs
        it. Then, if every consumer is also gone -- inside the region, or having explicitly dropped
        the requirement -- nothing outside could ever ask for it again.

        This is code-owned and not the model's `INVALIDATE_INFO`. Without it, an unavailable node
        left behind by a removed producer fails the one-producer rule at validation, and a revision
        the model wrote correctly is refused over a node it had no reason to mention.
        """
        reused = self._reused_anchors(ops)
        dropped = self._dropped_requirements(ops)
        internal = set()
        for node in self.previous.information:
            producers = self.previous.producers_of(node.id)
            if not producers or any(c not in region for c in producers):
                continue
            consumers = self.previous.consumers_of(node.id)
            if any(c not in region and (c, node.id) not in dropped for c in consumers):
                continue
            if node.id in reused:
                continue
            internal.add(node.id)
        return internal

    def _reused_anchors(self, ops) -> set[str]:
        """Nodes the revision keeps using. A node it only drops or invalidates is not reused."""
        reused: set[str] = set()
        for op in ops:
            if isinstance(op, ReviseInformation):
                reused.add(op.anchor)
            for computation in getattr(op, "computations", ()):
                reused.update(computation.requires)
                reused.update(computation.produces)
                reused.update(v.information_id for v in computation.arguments.values()
                              if isinstance(v, InformationReference))
            for entry in getattr(op, "now_available", ()):
                reused.add(entry.anchor)
            reused.update(getattr(op, "add_requires", ()))
        return reused

    def _dropped_requirements(self, ops) -> set[tuple[str, str]]:
        return {(op.anchor, i) for op in ops if isinstance(op, ReviseComputation)
                for i in op.remove_requires}

    def _invalidated_information(self, ops, region: set[str]) -> set[str]:
        invalidated = set()
        for op in ops:
            if not isinstance(op, InvalidateInformation):
                continue
            if op.anchor not in self.information_ids:
                continue
            producers = self.previous.producers_of(op.anchor)
            surviving = [p for p in producers if p not in region]
            if surviving:
                self.fail("surviving_producer_of_invalidated_information",
                          "information cannot be invalidated while a surviving computation "
                          "produces it; replace that computation instead",
                          op.anchor, *surviving)
                continue
            invalidated.add(op.anchor)
        return invalidated

    # ------------------------------------------------------------------ crossings
    def _check_crossings(self, ops, region: set[str], removed_information: set[str]) -> None:
        roots = {op.anchor: op for op in ops
                 if isinstance(op, (Replace, Complete, Invalidate))}
        if not region:
            self._check_information_consumers(ops, region, removed_information)
            return

        re_required, re_produced, re_after = self._re_established(ops)
        dropped_requires = {i for op in ops if isinstance(op, Replace)
                            for i in op.no_longer_requires}
        dropped_after = {c for op in ops if isinstance(op, Replace) for c in op.no_longer_after}
        now_available = {e.anchor for op in ops if isinstance(op, Complete)
                         for e in op.now_available}
        # A survivor that waited on removed work is handled by a `REVISE` that says something about
        # this successor's ordering: `remove-after` naming what is going, or an `add-after` saying
        # what it waits on instead. An unrelated revision of the same computation does not count,
        # or any `REVISE` at all would launder the crossing. Roots cannot appear here: they and
        # their descendants are inside the region by construction.
        reordered = {op.anchor for op in ops if isinstance(op, ReviseComputation) and op.add_after}
        dropped_orderings = {(op.anchor, c) for op in ops if isinstance(op, ReviseComputation)
                             for c in op.remove_after}

        for edge in self.previous.edges:
            if edge.relation in (Relation.INTERFACE_INPUT, Relation.INTERFACE_OUTPUT):
                continue           # code-derived, recomputed from scratch
            inside_source = edge.source in region
            inside_target = edge.target in region
            if inside_source == inside_target:
                continue           # wholly inside, or wholly outside

            if edge.relation is Relation.REQUIRES and inside_target:
                if edge.source in removed_information:
                    continue
                operation = self._governing(edge.target, roots)
                if isinstance(operation, (Complete, Invalidate)):
                    continue
                if edge.source in re_required or edge.source in dropped_requires:
                    continue
                self.fail("unaccounted_crossing_relation",
                          f"{edge.source!r} was required by the replaced work and the revision "
                          "neither requires it again nor declares it no longer required",
                          edge.source, edge.target)

            elif edge.relation is Relation.PRODUCES and inside_source:
                if edge.target in removed_information:
                    continue
                operation = self._governing(edge.source, roots)
                if isinstance(operation, Complete) and edge.target in now_available:
                    continue
                if isinstance(operation, Replace) and edge.target in re_produced:
                    continue
                self.fail("unaccounted_crossing_relation",
                          f"{edge.target!r} was produced by the removed work and the revision "
                          "neither produces it again, nor establishes it, nor invalidates it",
                          edge.source, edge.target)

            elif edge.relation is Relation.PRECEDES:
                if inside_target:
                    # an outside predecessor of removed work
                    operation = self._governing(edge.target, roots)
                    if isinstance(operation, (Complete, Invalidate)):
                        continue
                    if edge.source in re_after or edge.source in dropped_after:
                        continue
                    self.fail("unaccounted_crossing_relation",
                              f"the replaced work came after {edge.source!r} and the revision "
                              "neither keeps that order nor declares it no longer required",
                              edge.source, edge.target)
                else:
                    # Removed work was a prerequisite of something that survives. Dataflow into the
                    # successor does not settle this, however suggestive it looks: needing what one
                    # part of the replacement produces orders the successor after that part, while
                    # the edge may have meant it waits for the whole obligation. Which of the two
                    # the old ordering meant is a semantic judgement about a graph that no longer
                    # exists, so the successor has to be named.
                    operation = self._governing(edge.source, roots)
                    if isinstance(operation, Complete):
                        continue          # the prerequisite really was satisfied
                    if edge.target in reordered \
                            or (edge.target, edge.source) in dropped_orderings:
                        continue
                    self.fail("unaccounted_crossing_relation",
                              f"{edge.target!r} waited on work that is gone, and nothing in the "
                              "revision says whether it still waits or what it waits on now",
                              edge.source, edge.target)

            elif edge.relation is Relation.REFINES:
                continue           # position inheritance, handled during assembly

        self._check_information_consumers(ops, region, removed_information)

    def _check_information_consumers(self, ops, region, removed_information) -> None:
        removed_by_hand = {op.anchor for op in ops if isinstance(op, InvalidateInformation)}
        dropped = {(op.anchor, i) for op in ops if isinstance(op, ReviseComputation)
                   for i in op.remove_requires}
        for information_id in removed_by_hand & removed_information:
            for consumer in self.previous.consumers_of(information_id):
                if consumer in region:
                    continue
                if (consumer, information_id) in dropped:
                    continue
                self.fail("unhandled_information_reference",
                          f"{consumer!r} still requires invalidated information, and only the "
                          "revision can say it no longer does",
                          information_id, consumer)

    def _governing(self, node_id: str, roots) -> Operation | None:
        if node_id in roots:
            return roots[node_id]
        for anchor in self.previous.refinement_ancestors_of(node_id):
            if anchor in roots:
                return roots[anchor]
        return None

    def _re_established(self, ops) -> tuple[set[str], set[str], set[str]]:
        required: set[str] = set()
        produced: set[str] = set()
        after: set[str] = set()
        for op in ops:
            for computation in getattr(op, "computations", ()):
                required.update(computation.requires)
                required.update(v.information_id for v in computation.arguments.values()
                                if isinstance(v, InformationReference))
                produced.update(computation.produces)
                after.update(computation.after)
            if isinstance(op, ReviseComputation):
                required.update(op.add_requires)
                after.update(op.add_after)
        return required, produced, after

    # ------------------------------------------------------------------ completion
    def _plan_completions(self, ops, region: set[str]) -> dict[str, NowAvailable]:
        planned: dict[str, NowAvailable] = {}
        for op in ops:
            if not isinstance(op, Complete):
                continue
            inside = {op.anchor, *self.previous.refinement_descendants_of(op.anchor)}
            for entry in op.now_available:
                if entry.anchor not in self.information_ids:
                    self.fail("unknown_anchor",
                              f"{entry.anchor!r} names no information in the previous graph",
                              entry.anchor)
                    continue
                node = self.previous.node(entry.anchor)
                if node.available:
                    self.fail("now_available_is_already_available",
                              "only information that does not exist yet can be established by "
                              "completing the work that produces it", entry.anchor)
                    continue
                producers = self.previous.producers_of(entry.anchor)
                if len(producers) != 1 or producers[0] not in inside:
                    self.fail("now_available_producer_outside_region",
                              "the completed work must be the one thing that was going to produce "
                              f"it, and {len(producers)} computation(s) did", entry.anchor,
                              *producers)
                    continue
                if entry.anchor in planned:
                    self.fail("conflicting_operations",
                              f"{entry.anchor!r} is established twice", entry.anchor)
                    continue
                planned[entry.anchor] = entry
        return planned

    # ------------------------------------------------------------------ assembly
    def _assemble(self, ops, roots, region, internal, invalidated, completion) -> Applied:
        removed_information = internal | invalidated
        surviving_computations = [c for c in self.previous.computations if c.id not in region]
        surviving_information = [i for i in self.previous.information
                                 if i.id not in removed_information]

        new_computations: list[tuple[str, NewComputation]] = []
        new_information: list[tuple[str, NewInformation]] = []
        for op in ops:
            for computation in getattr(op, "computations", ()):
                new_computations.append((computation.label, computation))
            for information in getattr(op, "information", ()):
                new_information.append((information.label, information))

        labels = {label for label, _ in new_computations} | {l for l, _ in new_information}
        duplicate = [label for label in labels
                     if [l for l, _ in new_computations + new_information].count(label) > 1]
        for label in sorted(set(duplicate)):
            self.fail("redeclared_label", f"{label!r} is declared more than once", label)

        identity = _allocate(surviving_computations, new_computations,
                             surviving_information, new_information)
        self._check_references(ops, identity, region, removed_information)
        if self.faults:
            return Applied(None, RevisionChanges(), tuple(self.faults))

        graph, boundary, completion_changes, removed_edges = self._build(
            ops, roots, region, removed_information, completion, identity,
            surviving_computations, surviving_information, new_computations, new_information)

        changes = RevisionChanges(
            affected_roots=tuple(sorted(roots)),
            touched_nodes=tuple(sorted(op.anchor for op in ops
                                       if isinstance(op, (ReviseComputation, ReviseInformation)))),
            removed_nodes=tuple(
                [NodeRemoval(n, "affected_region") for n in sorted(region)]
                + [NodeRemoval(n, "region_internal") for n in sorted(internal)]
                + [NodeRemoval(n, "invalidated_information") for n in sorted(invalidated)]),
            removed_edges=tuple(removed_edges),
            replacement_boundary_changes=tuple(boundary),
            completion_changes=tuple(completion_changes),
            id_map=tuple(sorted(identity.items())),
        )
        return Applied(graph, changes, ())

    def _check_references(self, ops, identity, region, removed_information) -> None:
        gone = region | removed_information
        for op in ops:
            for computation in getattr(op, "computations", ()):
                names = (list(computation.requires) + list(computation.produces)
                         + list(computation.refined_into) + list(computation.after)
                         + [v.information_id for v in computation.arguments.values()
                            if isinstance(v, InformationReference)])
                for name in names:
                    self._check_reference(name, identity, gone, computation.label)
            if isinstance(op, ReviseComputation):
                for name in (*op.add_requires, *op.add_after):
                    self._check_reference(name, identity, gone, op.anchor)
                # Dropping a relation to something this revision removed is the point of dropping
                # it, so these are checked against the graph the model was shown, not the result.
                for name in (*op.remove_requires, *op.remove_after):
                    if name not in self.computation_ids and name not in self.information_ids:
                        self.fail("unknown_anchor",
                                  f"{name!r} names nothing in the previous graph", op.anchor, name)

    def _check_reference(self, name, identity, gone, where) -> None:
        if name in gone:
            self.fail("reference_into_removed_region",
                      f"{name!r} was removed by this revision and cannot be referred to", where,
                      name)
        elif name not in identity:
            self.fail("unknown_anchor", f"{name!r} names nothing in the resulting graph", where,
                      name)

    def _build(self, ops, roots, region, removed_information, completion, identity,
               surviving_computations, surviving_information, new_computations, new_information):
        graph = StateGraph()
        boundary: list[EdgeChange] = []
        completion_changes: list[CompletionChange] = []

        revised_info = {op.anchor: op for op in ops if isinstance(op, ReviseInformation)}
        for node in surviving_information:
            graph.add(_revised_information(node, identity[node.id], revised_info.get(node.id),
                                           completion.get(node.id), completion_changes))
        for label, declared in new_information:
            graph.add(InformationNode(id=identity[label], kind=declared.kind,
                                      description=declared.description,
                                      available=declared.available, payload=declared.payload))
        for node in surviving_computations:
            graph.add(ComputationNode(id=identity[node.id], description=node.description,
                                      operation=node.operation,
                                      arguments=_renamed(node.arguments, identity)))
        for label, declared in new_computations:
            graph.add(ComputationNode(id=identity[label], description=declared.description,
                                      operation=declared.operation,
                                      arguments=_renamed(declared.arguments, identity)))

        removed_edges = self._carry_edges(graph, ops, region, removed_information, completion,
                                          identity, completion_changes)
        self._declared_edges(graph, ops, identity)
        self._inherit_positions(graph, ops, roots, identity, boundary)
        self._materialize_provenance(graph, ops, completion, identity, completion_changes)
        return graph, boundary, completion_changes, removed_edges

    def _carry_edges(self, graph, ops, region, removed_information, completion, identity,
                     completion_changes) -> list[EdgeChange]:
        removed: list[EdgeChange] = []
        dropped = {(op.anchor, i) for op in ops if isinstance(op, ReviseComputation)
                   for i in op.remove_requires}
        dropped |= {(op.anchor, c) for op in ops if isinstance(op, ReviseComputation)
                    for c in op.remove_after}
        for edge in self.previous.edges:
            if edge.relation in (Relation.INTERFACE_INPUT, Relation.INTERFACE_OUTPUT):
                continue
            gone = region | removed_information
            if edge.source in gone or edge.target in gone:
                removed.append(EdgeChange("removed", edge.source, edge.relation, edge.target))
                if (edge.relation is Relation.PRODUCES and edge.target in completion):
                    completion_changes.append(CompletionChange(
                        "producer_removed", edge.target,
                        f"produced by {edge.source}, which completed"))
                continue
            if edge.relation is Relation.REQUIRES and (edge.target, edge.source) in dropped:
                removed.append(EdgeChange("removed", edge.source, edge.relation, edge.target))
                continue
            if edge.relation is Relation.PRECEDES and (edge.target, edge.source) in dropped:
                removed.append(EdgeChange("removed", edge.source, edge.relation, edge.target))
                continue
            graph.add_edge(identity[edge.source], edge.relation, identity[edge.target])
        return removed

    def _declared_edges(self, graph, ops, identity) -> None:
        for op in ops:
            for computation in getattr(op, "computations", ()):
                here = identity[computation.label]
                for name in computation.requires:
                    graph.add_edge(identity[name], Relation.REQUIRES, here)
                for name in computation.produces:
                    graph.add_edge(here, Relation.PRODUCES, identity[name])
                for name in computation.refined_into:
                    graph.add_edge(here, Relation.REFINES, identity[name])
                for name in computation.after:
                    graph.add_edge(identity[name], Relation.PRECEDES, here)
            if isinstance(op, ReviseComputation):
                here = identity[op.anchor]
                for name in op.add_requires:
                    graph.add_edge(identity[name], Relation.REQUIRES, here)
                for name in op.add_after:
                    graph.add_edge(identity[name], Relation.PRECEDES, here)

    def _inherit_positions(self, graph, ops, roots, identity, boundary) -> None:
        """A replacement takes the place of what it replaced.

        `REPLACE` means replace in place, so the replacement roots become children of whatever the
        removed root was a child of. The parent's own fields are untouched and it wrote nothing, but
        its edge now points somewhere else, which is why this is recorded rather than called
        preservation.
        """
        for op in ops:
            if not isinstance(op, Replace):
                continue
            parents = [p for p in self.previous.refinement_parents_of(op.anchor)
                       if p in identity]
            if not parents:
                continue
            named_as_child = {name for c in op.computations for name in c.refined_into}
            tops = [c.label for c in op.computations if c.label not in named_as_child]
            for parent in parents:
                for label in tops:
                    graph.add_edge(identity[parent], Relation.REFINES, identity[label])
                    boundary.append(EdgeChange("added", identity[parent], Relation.REFINES,
                                               identity[label]))


    def _materialize_provenance(self, graph, ops, completion, identity,
                                completion_changes) -> None:
        """Say that the surviving refinement established what the completed work established.

        Once the producing child is gone the structure cannot tell this apart from a value
        established somewhere else that the refinement happens to use, because an available node has
        no producer either way. That is why `INTERFACE_OUTPUT` on available information is the one
        interface edge derivation cannot recover -- and why it can be recovered now: the model's
        `COMPLETE root` plus `now-available i` is exactly the provenance the graph was about to
        lose, and the code is only writing it down before it goes.
        """
        for op in ops:
            if not isinstance(op, Complete):
                continue
            ancestors = [a for a in self.previous.refinement_ancestors_of(op.anchor)
                         if a in identity]
            for entry in op.now_available:
                if entry.anchor not in completion:
                    continue
                information = identity[entry.anchor]
                for ancestor in ancestors:
                    below = {ancestor, *self.previous.refinement_descendants_of(ancestor)}
                    consumers = [c for c in graph.consumers_of(information)
                                 if c not in {identity[n] for n in below if n in identity}]
                    if not consumers:
                        continue      # it never leaves this boundary, so it does not cross it
                    graph.add_edge(identity[ancestor], Relation.INTERFACE_OUTPUT, information)
                    completion_changes.append(CompletionChange(
                        "provenance_materialized", information,
                        f"established by {identity[ancestor]}, whose producing child completed"))


def _renamed(arguments: Mapping[str, ArgumentValue], identity) -> dict:
    """The same arguments, with every reference pointing at the id its target ended up with."""
    return {key: (InformationReference(identity[value.information_id])
                  if isinstance(value, InformationReference) else value)
            for key, value in arguments.items()}


def _revised_information(node: InformationNode, new_id: str,
                         revision: ReviseInformation | None,
                         completion: NowAvailable | None,
                         completion_changes: list[CompletionChange]) -> InformationNode:
    kind, description, payload = node.kind, node.description, node.payload
    available = node.available
    if revision is not None:
        kind = revision.kind if revision.kind is not None else kind
        description = revision.description if revision.description is not None else description
        if revision.payload_given:
            payload = revision.payload
    if completion is not None:
        available = True
        completion_changes.append(CompletionChange("became_available", new_id))
        if completion.kind is not None or completion.description is not None \
                or completion.payload is not None:
            kind = completion.kind if completion.kind is not None else kind
            description = (completion.description if completion.description is not None
                           else description)
            payload = completion.payload if completion.payload is not None else payload
            completion_changes.append(CompletionChange("content_replaced", new_id,
                                                       f"now {kind.value}"))
    try:
        return InformationNode(id=new_id, kind=kind, description=description,
                               available=available, payload=payload)
    except SchemaError as err:
        raise RevisionError(str(err)) from err


def _allocate(surviving_computations, new_computations,
              surviving_information, new_information) -> dict[str, str]:
    """Survivors in previous canonical order, then new entities in declaration order.

    Fixed so that the same revision against the same graph always produces the same ids, and so
    that nothing depends on the order a traversal happened to visit nodes in.
    """
    identity: dict[str, str] = {}
    position = 1
    for node in surviving_computations:
        identity[node.id] = f"c{position}"
        position += 1
    for label, _ in new_computations:
        identity[label] = f"c{position}"
        position += 1
    position = 1
    for node in surviving_information:
        identity[node.id] = f"i{position}"
        position += 1
    for label, _ in new_information:
        identity[label] = f"i{position}"
        position += 1
    return identity
