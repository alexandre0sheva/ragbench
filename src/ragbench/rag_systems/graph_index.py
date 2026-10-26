"""The knowledge graph of `graph_rag`: what an extraction reply says, the graph the replies merge into, and its communities (Louvain)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ragbench.documents.schema import TextChunk
from ragbench.utils.query_planning import loads_lenient

UNKNOWN_TYPE = "unknown"


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    target: str
    description: str


@dataclass
class Extraction:
    """What one chunk's extraction reply says: the entities in it and the relations between them."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


@dataclass
class Entity:
    name: str  # the first spelling seen
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    types: Counter[str] = field(default_factory=Counter)

    @property
    def type(self) -> str:
        """The most frequent type given to it (the first one seen on a tie); `unknown` only when no extraction typed it."""
        typed = [(kind, count) for kind, count in self.types.most_common() if kind != UNKNOWN_TYPE]
        return typed[0][0] if typed else UNKNOWN_TYPE


@dataclass
class Relation:
    source: str  # normalised names, in sorted order: the graph is undirected
    target: str
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    weight: float = 0.0  # how many times it was extracted


def normalize_name(name: str) -> str:
    """The key an entity is merged under: case, inner spacing, surrounding quotes and punctuation, and a leading "the" do not count."""
    key = " ".join(name.split()).casefold().strip("\"'`.,;:()[]")
    return key[4:] if key.startswith("the ") else key


def parse_extraction(text: str) -> Extraction | None:
    """The entities and relations of an extraction reply (JSON, fenced or with prose around it); malformed items are skipped.

    None when the reply is not a JSON object at all (prose, or JSON cut off at the token limit): a failed call, not a chunk without entities.
    """
    data = loads_lenient(text)
    if not isinstance(data, dict):
        return None
    entities = [
        ExtractedEntity(name=item["name"].strip(), type=_text(item.get("type")).lower() or UNKNOWN_TYPE, description=_text(item.get("description")))
        for item in _items(data.get("entities"))
        if _named(item, "name")
    ]
    relations = [
        ExtractedRelation(source=item["source"].strip(), target=item["target"].strip(), description=_text(item.get("description")))
        for item in _items(data.get("relations"))
        if _named(item, "source") and _named(item, "target")
    ]
    return Extraction(entities=entities, relations=relations)


def _items(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _named(item: dict, key: str) -> bool:
    return isinstance(item.get(key), str) and bool(normalize_name(item[key]))


def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


class KnowledgeGraph:
    """Entities merged by normalised name across chunks, and the undirected, weighted relations between them."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple[str, str], Relation] = {}
        self.chunk_docs: dict[str, str] = {}
        self._adjacent: dict[str, dict[str, float]] = {}

    def add(self, chunk: TextChunk, extraction: Extraction) -> None:
        self.chunk_docs[chunk.chunk_id] = chunk.doc_id
        for entity in extraction.entities:
            self._mention(entity.name, entity.type, entity.description, chunk.chunk_id)
        for relation in extraction.relations:
            source, target = normalize_name(relation.source), normalize_name(relation.target)
            if source == target:
                continue
            self._mention(relation.source, UNKNOWN_TYPE, "", chunk.chunk_id)
            self._mention(relation.target, UNKNOWN_TYPE, "", chunk.chunk_id)
            key = (min(source, target), max(source, target))
            edge = self.relations.setdefault(key, Relation(source=key[0], target=key[1]))
            edge.weight += 1
            _append_once(edge.descriptions, relation.description)
            _append_once(edge.chunk_ids, chunk.chunk_id)
            for a, b in (key, key[::-1]):
                self._adjacent.setdefault(a, {})[b] = edge.weight

    def _mention(self, name: str, kind: str, description: str, chunk_id: str) -> None:
        entity = self.entities.setdefault(normalize_name(name), Entity(name=" ".join(name.split())))
        entity.types[kind] += 1
        _append_once(entity.descriptions, description)
        _append_once(entity.chunk_ids, chunk_id)

    def neighbours(self, key: str) -> list[tuple[str, float]]:
        """The entities related to `key`, heaviest relation first (then by name)."""
        return sorted(self._adjacent.get(key, {}).items(), key=lambda item: (-item[1], item[0]))

    def degree(self, key: str) -> float:
        return sum(self._adjacent.get(key, {}).values())

    def doc_ids(self, *keys: str) -> list[str]:
        """The documents that mention the entities, by how many of their chunks do (then in the order first seen)."""
        counts = Counter(self.chunk_docs[chunk_id] for key in keys for chunk_id in self.entities[key].chunk_ids)
        return [doc_id for doc_id, _ in counts.most_common()]

    def edges(self) -> dict[tuple[str, str], float]:
        return {key: relation.weight for key, relation in self.relations.items()}


def _append_once(items: list[str], item: str) -> None:
    if item and item not in items:
        items.append(item)


def detect_communities(nodes: list[str], edges: dict[tuple[str, str], float], *, resolution: float = 1.0) -> list[list[str]]:
    """Louvain modularity communities of an undirected weighted graph: local moves, then aggregation, until nothing moves.

    Deterministic: nodes are visited in sorted order and ties keep the current community, so the input order does not matter. Each
    community's members are sorted; communities come largest first (then by first member). A node without edges is its own community.
    """
    names = sorted(set(nodes) | {name for edge in edges for name in edge})
    index = {name: i for i, name in enumerate(names)}
    adjacency: list[dict[int, float]] = [{} for _ in names]
    loops = [0.0] * len(names)  # weight inside an aggregated node, each edge once
    for (a, b), weight in sorted(edges.items()):
        i, j = index[a], index[b]
        if weight <= 0:
            continue
        if i == j:
            loops[i] += weight
            continue
        adjacency[i][j] = adjacency[i].get(j, 0.0) + weight
        adjacency[j][i] = adjacency[j].get(i, 0.0) + weight
    members = [[name] for name in names]
    while True:
        community, moved = _local_moves(adjacency, loops, resolution)
        if not moved:
            break
        adjacency, loops, merged = _aggregate(adjacency, loops, community, members)
        if len(merged) == len(members):
            break
        members = merged
    return sorted((sorted(group) for group in members), key=lambda group: (-len(group), group[0]))


def _local_moves(adjacency: list[dict[int, float]], loops: list[float], resolution: float) -> tuple[list[int], bool]:
    """Louvain's first phase: move each node to the neighbouring community with the largest modularity gain until none improves."""
    degree = [sum(links.values()) + 2 * loop for links, loop in zip(adjacency, loops, strict=True)]
    twice_m = sum(degree)
    community = list(range(len(adjacency)))
    if twice_m == 0:
        return community, False
    total = degree[:]
    moved, changed = False, True
    while changed:
        changed = False
        for node, links in enumerate(adjacency):
            current = community[node]
            to_community: dict[int, float] = {}
            for other, weight in links.items():
                to_community[community[other]] = to_community.get(community[other], 0.0) + weight
            total[current] -= degree[node]
            best = current
            best_gain = to_community.get(current, 0.0) - resolution * total[current] * degree[node] / twice_m
            for candidate in sorted(to_community):
                gain = to_community[candidate] - resolution * total[candidate] * degree[node] / twice_m
                if gain > best_gain + 1e-12:
                    best, best_gain = candidate, gain
            total[best] += degree[node]
            if best != current:
                community[node] = best
                moved = changed = True
    return community, moved


def _aggregate(
    adjacency: list[dict[int, float]], loops: list[float], community: list[int], members: list[list[str]]
) -> tuple[list[dict[int, float]], list[float], list[list[str]]]:
    """Louvain's second phase: one node per community, edges between communities summed, edges inside them kept as its loop."""
    renumber = {old: new for new, old in enumerate(sorted(set(community)))}
    merged: list[list[str]] = [[] for _ in renumber]
    new_loops = [0.0] * len(renumber)
    new_adjacency: list[dict[int, float]] = [{} for _ in renumber]
    for node, links in enumerate(adjacency):
        mine = renumber[community[node]]
        merged[mine].extend(members[node])
        new_loops[mine] += loops[node]
        for other, weight in links.items():
            theirs = renumber[community[other]]
            if theirs == mine:
                if node < other:
                    new_loops[mine] += weight
            else:
                new_adjacency[mine][theirs] = new_adjacency[mine].get(theirs, 0.0) + weight
    return new_adjacency, new_loops, merged
