"""The knowledge graph behind `graph_rag`: parsing an extraction reply, merging it into one graph, and finding communities."""

from __future__ import annotations

import json

from ragbench.documents.schema import TextChunk
from ragbench.rag_systems.graph_index import KnowledgeGraph, detect_communities, normalize_name, parse_extraction


def _chunk(chunk_id: str, doc_id: str) -> TextChunk:
    return TextChunk(chunk_id=chunk_id, doc_id=doc_id, text="")


def _reply(entities: list[tuple[str, str, str]], relations: list[tuple[str, str, str]]) -> str:
    return json.dumps(
        {
            "entities": [{"name": n, "type": t, "description": d} for n, t, d in entities],
            "relations": [{"source": s, "target": t, "description": d} for s, t, d in relations],
        }
    )


# --- parsing ----------------------------------------------------------------------------------------------------------------


def test_an_extraction_reply_in_a_code_fence_with_prose_around_it_still_parses():
    reply = "Here you go:\n```json\n" + _reply([("Acme Corp", "organization", "A maker of anvils.")], []) + "\n```"

    extraction = parse_extraction(reply)

    assert [(e.name, e.type, e.description) for e in extraction.entities] == [("Acme Corp", "organization", "A maker of anvils.")]


def test_malformed_items_are_skipped_and_the_rest_is_kept():
    reply = json.dumps(
        {
            "entities": [{"name": ""}, "not an object", {"name": "Oslo"}, {"type": "person"}],
            "relations": [{"source": "Oslo"}, {"source": "Oslo", "target": "Norway", "description": 7}, {"source": "Oslo", "target": "Norway"}],
        }
    )

    extraction = parse_extraction(reply)

    assert [e.name for e in extraction.entities] == ["Oslo"] and extraction.entities[0].type == "unknown"
    assert [(r.source, r.target, r.description) for r in extraction.relations] == [("Oslo", "Norway", "7"), ("Oslo", "Norway", "")]


def test_an_unusable_reply_is_no_extraction_at_all_but_valid_empty_json_is_an_empty_one():
    assert parse_extraction("I cannot help with that.") is None
    assert parse_extraction('["a", "b"]') is None
    assert parse_extraction('{"entities": [{"name": "Acme", "type": "org"') is None, "a reply cut off at max_tokens"
    empty = parse_extraction('{"entities": [], "relations": []}')
    assert empty is not None and empty.entities == [] and empty.relations == []


def test_names_are_normalised_by_case_spacing_and_a_leading_article():
    assert normalize_name("  The  ACME   corp ") == normalize_name("acme corp") == "acme corp"
    assert normalize_name("Theodore Roosevelt") == "theodore roosevelt"


# --- merging ----------------------------------------------------------------------------------------------------------------


def test_one_entity_found_in_two_chunks_is_one_node_with_both_chunks_and_both_descriptions():
    graph = KnowledgeGraph()
    graph.add(_chunk("c1", "doc_a"), parse_extraction(_reply([("Acme Corp", "organization", "Makes anvils.")], [])))
    graph.add(_chunk("c2", "doc_b"), parse_extraction(_reply([("ACME corp", "company", "Based in Oslo."), ("Oslo", "place", "A city.")], [])))
    graph.add(_chunk("c3", "doc_b"), parse_extraction(_reply([("acme corp", "organization", "Makes anvils.")], [])))

    acme = graph.entities["acme corp"]
    assert acme.name == "Acme Corp", "the first spelling seen is the display name"
    assert acme.type == "organization", "the most frequent type wins"
    assert acme.descriptions == ["Makes anvils.", "Based in Oslo."], "repeated descriptions are kept once"
    assert acme.chunk_ids == ["c1", "c2", "c3"]
    assert graph.doc_ids("acme corp") == ["doc_b", "doc_a"], "documents by how many chunks mention the entity"


def test_a_relation_adds_its_unknown_endpoints_counts_repeats_and_ignores_self_loops():
    graph = KnowledgeGraph()
    graph.add(_chunk("c1", "doc_a"), parse_extraction(_reply([("Alice", "person", "")], [("Alice", "Acme Corp", "Alice founded Acme."), ("Alice", "alice", "x")])))
    graph.add(_chunk("c2", "doc_b"), parse_extraction(_reply([], [("Acme Corp", "Alice", "Acme was founded by Alice.")])))

    assert set(graph.entities) == {"alice", "acme corp"}
    assert graph.entities["acme corp"].type == "unknown" and graph.entities["acme corp"].chunk_ids == ["c1", "c2"]
    [relation] = graph.relations.values()
    assert relation.weight == 2 and relation.descriptions == ["Alice founded Acme.", "Acme was founded by Alice."]
    assert graph.neighbours("alice") == [("acme corp", 2.0)]


def test_the_documents_of_several_entities_are_counted_together():
    graph = KnowledgeGraph()
    graph.add(_chunk("c1", "doc_a"), parse_extraction(_reply([("Ann", "person", "")], [])))
    graph.add(_chunk("c2", "doc_b"), parse_extraction(_reply([("Ben", "person", "")], [])))
    graph.add(_chunk("c3", "doc_b"), parse_extraction(_reply([("Ann", "person", "")], [])))

    assert graph.doc_ids("ann", "ben") == ["doc_b", "doc_a"]


# --- communities --------------------------------------------------------------------------------------------------------------


def _clique(names: list[str]) -> dict[tuple[str, str], float]:
    return {(a, b): 1.0 for i, a in enumerate(names) for b in names[i + 1 :]}


def test_two_dense_groups_joined_by_one_edge_are_two_communities_and_a_loner_is_its_own():
    left, right = ["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]
    edges = {**_clique(left), **_clique(right), ("a4", "b1"): 1.0}

    communities = detect_communities([*left, *right, "loner"], edges)

    assert sorted(sorted(c) for c in communities) == [left, right, ["loner"]]


def test_communities_do_not_depend_on_the_order_nodes_and_edges_are_given_in():
    left, middle, right = ["a1", "a2", "a3"], ["m1", "m2", "m3"], ["b1", "b2", "b3"]
    edges = {**_clique(left), **_clique(middle), **_clique(right), ("a3", "m1"): 1.0, ("m3", "b1"): 1.0}
    nodes = [*left, *middle, *right]

    forward = detect_communities(nodes, edges)
    backward = detect_communities(list(reversed(nodes)), dict(reversed(list(edges.items()))))

    assert forward == backward
    assert sorted(sorted(c) for c in forward) == [left, right, middle]


def test_a_heavy_edge_pulls_its_ends_together():
    edges = {("x", "y"): 10.0, ("y", "z"): 0.1, ("z", "w"): 10.0}

    assert sorted(sorted(c) for c in detect_communities(["w", "x", "y", "z"], edges)) == [["w", "z"], ["x", "y"]]
