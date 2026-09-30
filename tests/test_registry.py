from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragbench import registry as registry_module
from ragbench.registry import Registry, UnknownComponentError


class Widget:
    pass


def test_register_get_names_and_aliases():
    reg: Registry[type] = Registry("widget")

    @reg.register("basic", aliases=("plain", "b"))
    class Basic(Widget):
        pass

    assert reg.get("basic") is Basic and reg.get("plain") is Basic and reg.get("b") is Basic
    assert reg.names() == ["basic"]  # aliases are not listed as separate components
    assert "plain" in reg and "basic" in reg and "nope" not in reg
    assert dict(reg.items()) == {"basic": Basic}


def test_unknown_name_error_suggests_close_matches_and_lists_options():
    reg: Registry[type] = Registry("RAG system")
    reg.add("hybrid", Widget)
    reg.add("hybrid_rerank", Widget)
    reg.add("bm25", Widget)

    with pytest.raises(UnknownComponentError) as excinfo:
        reg.get("hybrd")

    message = str(excinfo.value)
    assert "Unknown RAG system 'hybrd'" in message and "Did you mean 'hybrid'" in message
    assert "bm25" in message and "hybrid_rerank" in message
    assert isinstance(excinfo.value, ValueError)  # create_rag_system always raised ValueError


def test_duplicate_registration_is_rejected_but_idempotent_for_the_same_object():
    reg: Registry[type] = Registry("widget")
    reg.add("a", Widget)
    reg.add("a", Widget)  # re-importing a module must not blow up
    with pytest.raises(ValueError, match="already registered"):
        reg.add("a", type("Other", (), {}))
    with pytest.raises(ValueError, match="already registered"):
        reg.add("x", Widget, aliases=("a",))


def test_mapping_is_a_live_view_for_backward_compatible_dict_access():
    reg: Registry[type] = Registry("widget")
    reg.add("a", Widget)
    assert reg.mapping == {"a": Widget}
    reg.mapping["b"] = Widget  # what monkeypatch.setitem(SYSTEM_REGISTRY, ...) does
    assert reg.get("b") is Widget


def test_entry_points_are_loaded_once_and_bad_plugins_do_not_break_startup(monkeypatch, caplog):
    class Good(Widget):
        pass

    def broken():
        raise ImportError("plugin exploded")

    fake_eps = [SimpleNamespace(name="good", load=lambda: Good), SimpleNamespace(name="bad", load=broken)]
    calls: list[str] = []

    def fake_entry_points(*, group):
        calls.append(group)
        return fake_eps

    monkeypatch.setattr(registry_module.metadata, "entry_points", fake_entry_points)
    reg: Registry[type] = Registry("widget")

    with caplog.at_level("WARNING"):
        reg.load_entry_points("ragbench.widgets")
        reg.load_entry_points("ragbench.widgets")

    assert reg.get("good") is Good and "bad" not in reg
    assert calls == ["ragbench.widgets"]  # second call is a no-op
    assert any("plugin exploded" in record.getMessage() for record in caplog.records)


def test_builtin_registries_are_populated():
    import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
    from ragbench.registry import CHUNKERS, RERANKERS, SYSTEMS

    assert {"bm25", "vector", "hybrid", "hybrid_rerank", "rerank", "parent_doc", "hyde", "llm_heavy"} <= set(SYSTEMS.names())
    assert {"fixed_char", "word", "token", "recursive", "sentence", "semantic", "markdown"} <= set(CHUNKERS.names())
    assert CHUNKERS.get("word") is not CHUNKERS.get("token") and CHUNKERS.get("md") is CHUNKERS.get("markdown")
    assert {"simple_keyword_overlap", "local_relevance", "llm"} <= set(RERANKERS.names())
