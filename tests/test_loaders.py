"""Document loaders: formats, encodings, ignore rules, error handling. Fixtures are generated in-test (no binaries are committed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.schema import DatasetConfig
from ragbench.documents.chunkers import create_chunker
from ragbench.documents.loaders import DocumentLoadError, load_documents, supported_extensions
from ragbench.models.errors import MissingExtraError
from ragbench.registry import LOADERS

ROOT = Path(__file__).resolve().parents[1]


def _write(directory: Path, name: str, content: str | bytes) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


# --- compatibility ------------------------------------------------------------------------------


def test_the_demo_dataset_loads_exactly_as_before():
    documents = load_documents(ROOT / "data" / "demo" / "docs")

    assert [d.doc_id for d in documents] == [f"doc_{i:03d}" for i in range(1, 61)]
    for document in documents:
        source = Path(document.path)
        assert document.text == source.read_text(encoding="utf-8")
        assert document.metadata == {"source_path": str(source), "extension": ".md"}
        assert document.title == next(line[2:].strip() for line in document.text.splitlines() if line.startswith("# "))


def test_supported_extensions_come_from_the_registry_and_cover_the_documented_formats():
    assert {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf", ".docx", ".csv", ".tsv", ".json", ".jsonl"} <= supported_extensions()
    assert LOADERS.get(".markdown") is LOADERS.get(".md") and LOADERS.get(".htm") is LOADERS.get(".html")


def test_a_custom_loader_registered_for_a_new_extension_is_used(tmp_path):
    from ragbench.documents.schema import Document

    def load_notes(path, context):
        return [Document(doc_id="doc_notes", path=str(path), title="Notes", text=path.read_text().upper(), metadata={})]

    LOADERS.add(".notes", load_notes)
    try:
        _write(tmp_path, "a.notes", "hello")
        assert [d.text for d in load_documents(tmp_path)] == ["HELLO"]
    finally:
        LOADERS.mapping.pop(".notes")


def test_nothing_loadable_is_an_error_that_lists_the_supported_formats(tmp_path):
    _write(tmp_path, "image.png", b"\x89PNG")

    with pytest.raises(ValueError, match=r"No supported documents.*\.docx"):
        load_documents(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")


# --- encodings ----------------------------------------------------------------------------------


def test_bom_utf16_cp1252_and_crlf_are_decoded_and_non_utf8_files_are_reported(tmp_path):
    _write(tmp_path, "bom.txt", b"\xef\xbb\xbfHello BOM")
    _write(tmp_path, "utf16.txt", "Hello sixteen, café".encode("utf-16"))
    _write(tmp_path, "legacy.txt", "Crème brûlée costs 5€".encode("cp1252"))
    _write(tmp_path, "crlf.txt", b"line one\r\nline two\rline three")
    warnings: list[str] = []

    by_name = {Path(d.path).name: d.text for d in load_documents(tmp_path, warnings=warnings)}

    assert by_name["bom.txt"] == "Hello BOM"
    assert by_name["utf16.txt"] == "Hello sixteen, café"
    assert by_name["legacy.txt"] == "Crème brûlée costs 5€"
    assert by_name["crlf.txt"] == "line one\nline two\nline three"
    assert any("legacy.txt" in w and "cp1252" in w for w in warnings) and not any("bom.txt" in w for w in warnings)


def test_undecodable_bytes_are_replaced_with_a_warning_instead_of_crashing(tmp_path):
    _write(tmp_path, "broken.txt", b"good \x81\x8d\x8f\x90\x9d text")
    warnings: list[str] = []

    (document,) = load_documents(tmp_path, warnings=warnings)

    assert document.text.startswith("good ") and document.text.endswith(" text") and "�" in document.text
    assert any("broken.txt" in w and "replac" in w for w in warnings)


# --- HTML ---------------------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html><head><title>  Refund &amp; Returns
   Policy </title>
<style>body { color: red; } .x::after { content: "</p>"; }</style>
<script>var x = "<p>not text</p>"; if (a < b && c > d) { document.write("<\\/script>"); }</script>
</head>
<body>
<!-- a comment with <b>tags</b> -->
<h1>Returns</h1>
<p>You have 30&nbsp;days &mdash; or 14 for &lt;sale&gt; items.<br>Second line &#169; 2026.</p>
<noscript>Enable JavaScript</noscript>
<ul><li>Keep the receipt</li><li>Keep the box</li></ul>
<table><tr><td>Item</td><td>Days</td></tr><tr><td>Shoes</td><td>30</td></tr></table>
<pre>  preformatted   text </pre>
</body></html>"""


def test_html_title_body_entities_and_nested_scripts_and_styles(tmp_path):
    _write(tmp_path, "policy.html", HTML)

    (document,) = load_documents(tmp_path)

    assert document.title == "Refund & Returns Policy"
    text = document.text
    assert "Returns" in text and "You have 30 days — or 14 for <sale> items." in text and "Second line © 2026." in text
    for leaked in ("color: red", "not text", "document.write", "comment", "Enable JavaScript", "&amp;", "&mdash;"):
        assert leaked not in text, leaked
    assert "<p>" not in text and "</" not in text
    assert "Keep the receipt" in text and "Shoes" in text and "preformatted" in text
    # Block elements break lines so chunkers can split on paragraphs; inline whitespace is collapsed.
    assert text.splitlines()[0] == "Returns" and "Keep the receipt\nKeep the box" in text.replace("\n\n", "\n")
    assert "\n\n\n" not in text and not text.startswith("\n") and not text.endswith("\n")


def test_html_title_falls_back_to_the_first_h1_then_the_file_name(tmp_path):
    _write(tmp_path, "a.html", "<html><body><h1>Heading Title</h1><p>x</p></body></html>")
    _write(tmp_path, "my_page-name.htm", "<p>no title at all</p>")

    titles = {Path(d.path).name: d.title for d in load_documents(tmp_path)}

    assert titles == {"a.html": "Heading Title", "my_page-name.htm": "My Page Name"}


# --- CSV / TSV / JSON ---------------------------------------------------------------------------

CSV = 'id,question,answer,team\nq1,"What is the refund window?","Thirty days, with receipt",support\nq2,How long is shipping?,Five days,logistics\n,Blank id row,Kept,ops\n'


def test_csv_yields_one_document_per_row_with_header_context(tmp_path):
    _write(tmp_path, "doc_faq.csv", CSV)

    documents = load_documents(tmp_path)

    assert [d.doc_id for d in documents] == ["doc_faq#1", "doc_faq#2", "doc_faq#3"]
    assert documents[0].text == "id: q1\nquestion: What is the refund window?\nanswer: Thirty days, with receipt\nteam: support"
    assert [d.metadata["row"] for d in documents] == [1, 2, 3] and documents[0].metadata["extension"] == ".csv"
    assert documents[0].title == "Doc Faq row 1" and Path(documents[0].path).name == "doc_faq.csv"


def test_tabular_options_choose_text_columns_and_id_column(tmp_path):
    _write(tmp_path, "doc_faq.csv", CSV)
    config = DatasetConfig(documents_path=tmp_path, questions_path=tmp_path / "q", tabular={"text_columns": ["question", "answer"], "id_column": "id"})

    documents = load_documents(tmp_path, tabular=config.tabular)

    assert [d.doc_id for d in documents] == ["doc_faq#q1", "doc_faq#q2", "doc_faq#3"], "rows without an id fall back to their row number"
    assert documents[1].text == "question: How long is shipping?\nanswer: Five days" and documents[1].metadata["row"] == 2
    assert documents[0].metadata["id"] == "q1"
    with pytest.raises(DocumentLoadError, match="no column 'nope'"):
        load_documents(tmp_path, tabular=DatasetConfig(documents_path=tmp_path, questions_path=tmp_path, tabular={"text_columns": ["nope"]}).tabular)


def test_tsv_and_json_and_jsonl_records(tmp_path):
    _write(tmp_path / "t", "doc_t.tsv", "a\tb\n1\t2\n3\t4\n")
    _write(tmp_path / "j", "doc_j.json", json.dumps([{"q": "one", "a": "x"}, {"q": "two", "a": "y", "extra": [1, 2]}]))
    _write(tmp_path / "l", "doc_l.jsonl", '{"q": "one"}\n\n{"q": "two"}\n')
    _write(tmp_path / "o", "doc_o.json", json.dumps({"title": "config", "nested": {"k": "v"}}))

    tsv, array, lines, single = (load_documents(tmp_path / d) for d in "tjlo")

    assert [d.text for d in tsv] == ["a: 1\nb: 2", "a: 3\nb: 4"]
    assert [d.doc_id for d in array] == ["doc_j#1", "doc_j#2"] and array[1].text == 'q: two\na: y\nextra: [1, 2]'
    assert [d.text for d in lines] == ["q: one", "q: two"]
    assert len(single) == 1 and single[0].doc_id == "doc_o" and '"nested"' in single[0].text and "row" not in single[0].metadata


def test_an_empty_or_header_only_csv_is_skipped_with_a_warning(tmp_path):
    _write(tmp_path, "empty.csv", "")
    _write(tmp_path, "header.csv", "a,b\n")
    _write(tmp_path, "real.txt", "content")
    warnings: list[str] = []

    documents = load_documents(tmp_path, warnings=warnings)

    assert [Path(d.path).name for d in documents] == ["real.txt"] and len(warnings) == 2


# --- DOCX ---------------------------------------------------------------------------------------


def _docx(path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.core_properties.title = "Employee Handbook"
    document.add_heading("Leave policy", level=1)
    document.add_paragraph("Employees receive 25 days of leave.")
    document.add_heading("Carry-over", level=2)
    document.add_paragraph("Up to five days carry over.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Type", "Days"
    table.cell(1, 0).text, table.cell(1, 1).text = "Annual", "25"
    document.add_paragraph("")
    document.add_paragraph("Closing remarks.")
    document.save(str(path))


def test_docx_headings_paragraphs_and_tables_keep_their_order_and_structure(tmp_path):
    _docx(tmp_path / "handbook.docx")

    (document,) = load_documents(tmp_path)

    assert document.title == "Employee Handbook"
    assert document.text == "# Leave policy\n\nEmployees receive 25 days of leave.\n\n## Carry-over\n\nUp to five days carry over.\n\nType | Days\nAnnual | 25\n\nClosing remarks."
    assert document.metadata["extension"] == ".docx"
    sections = create_chunker({"type": "markdown", "chunk_size": 200, "chunk_overlap": 0, "min_chunk_size": 1}).chunk([document])
    assert [c.metadata["heading_path"] for c in sections] == [["Leave policy"], ["Leave policy", "Carry-over"]]


def test_docx_title_falls_back_to_the_first_heading_then_the_file_name(tmp_path):
    docx = pytest.importorskip("docx")
    titled = docx.Document()
    titled.add_heading("First Heading", level=1)
    titled.save(str(tmp_path / "a.docx"))
    plain = docx.Document()
    plain.add_paragraph("just text")
    plain.save(str(tmp_path / "b_file.docx"))

    titles = {Path(d.path).name: d.title for d in load_documents(tmp_path)}

    assert titles == {"a.docx": "First Heading", "b_file.docx": "B File"}


def test_docx_without_the_extra_raises_an_actionable_error(tmp_path, monkeypatch):
    _write(tmp_path, "x.docx", b"PK\x03\x04")
    monkeypatch.setitem(sys.modules, "docx", None)

    with pytest.raises(MissingExtraError, match=r"ragbench\[docx\]"):
        load_documents(tmp_path)
    with pytest.raises(MissingExtraError, match=r"ragbench\[docx\]"):
        load_documents(tmp_path, on_error="skip")  # a missing extra is an environment problem, never a "bad file"


# --- PDF ----------------------------------------------------------------------------------------


def _pdf_bytes(pages: list[str]) -> bytes:
    """A minimal valid PDF: one text line per page, Helvetica."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    return out + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def test_pdf_pages_are_recorded_as_spans_and_chunks_carry_their_page(tmp_path):
    pytest.importorskip("pypdf")
    _write(tmp_path, "report.pdf", _pdf_bytes(["Alpha page one text", "", "Gamma page three text"]))

    (document,) = load_documents(tmp_path)

    spans = document.metadata["page_spans"]
    assert [page for _, _, page in spans] == [1, 3], "empty pages are skipped but keep their page numbers"
    assert [document.text[start:end] for start, end, _ in spans] == ["Alpha page one text", "Gamma page three text"]
    chunks = create_chunker({"type": "sentence", "chunk_size": 6, "chunk_overlap": 0}).chunk([document])
    by_page = {c.metadata["page"]: c.text for c in chunks}
    assert by_page == {1: "Alpha page one text", 3: "Gamma page three text"}
    assert all("page_end" not in c.metadata for c in chunks)
    spanning = create_chunker({"type": "word", "chunk_size": 100, "chunk_overlap": 0}).chunk([document])
    assert spanning[0].metadata["page"] == 1 and spanning[0].metadata["page_end"] == 3


def test_documents_without_pages_get_no_page_metadata(tmp_path):
    (document,) = load_documents(_write(tmp_path, "a.txt", "plain text here").parent)

    assert all("page" not in c.metadata for c in create_chunker({"type": "word"}).chunk([document]))


def test_pdf_without_pypdf_raises_helpful_error(tmp_path, monkeypatch):
    _write(tmp_path, "doc.pdf", b"%PDF-1.4 stub")
    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(MissingExtraError, match=r"ragbench\[pdf\]"):
        load_documents(tmp_path)


# --- ignore rules, include / exclude, size limits -----------------------------------------------


def _tree(root: Path) -> None:
    for name in ["keep.md", "notes.md", "drafts/wip.md", "drafts/final/ok.md", "sub/a.md", "sub/b.tmp.md", "sub/deep/c.md", "top.tmp.md", "archive/old.md"]:
        _write(root, name, f"# {name}\n\ncontent of {name}")


def _names(root: Path, **kwargs) -> list[str]:
    return sorted(Path(d.path).relative_to(root).as_posix() for d in load_documents(root, **kwargs))


def test_ragbenchignore_uses_gitignore_style_patterns(tmp_path):
    _tree(tmp_path)
    _write(tmp_path, ".ragbenchignore", "# drafts are not documentation\ndrafts/\n*.tmp.md\n/notes.md\n\narchive/**\n!drafts/final/ok.md\n")

    assert _names(tmp_path) == ["keep.md", "sub/a.md", "sub/deep/c.md"]


def test_ragbenchignore_directory_rule_wins_over_a_negation_inside_the_ignored_directory(tmp_path):
    """Like git: a file cannot be re-included when a parent directory is excluded."""
    _tree(tmp_path)
    _write(tmp_path, ".ragbenchignore", "drafts/\n!drafts/final/ok.md\n")

    assert "drafts/final/ok.md" not in _names(tmp_path)


def test_ragbenchignore_negation_reincludes_a_file_excluded_by_a_glob(tmp_path):
    _tree(tmp_path)
    _write(tmp_path, ".ragbenchignore", "*.tmp.md\n!sub/b.tmp.md\n")

    names = _names(tmp_path)
    assert "sub/b.tmp.md" in names and "top.tmp.md" not in names


def test_anchored_and_unanchored_patterns(tmp_path):
    _tree(tmp_path)
    _write(tmp_path, ".ragbenchignore", "/notes.md\nc.md\n")

    names = _names(tmp_path)
    assert "notes.md" not in names and "sub/deep/c.md" not in names and "sub/a.md" in names
    _write(tmp_path / "x", "notes.md", "n")
    _write(tmp_path / "x", "sub/notes.md", "n")
    _write(tmp_path / "x", ".ragbenchignore", "/notes.md\n")
    assert _names(tmp_path / "x") == ["sub/notes.md"]


def test_include_and_exclude_globs(tmp_path):
    _tree(tmp_path)

    assert _names(tmp_path, include=["sub/*.md"]) == ["sub/a.md", "sub/b.tmp.md"]
    assert _names(tmp_path, include=["**/a.md", "keep.md"]) == ["keep.md", "sub/a.md"]
    assert _names(tmp_path, exclude=["drafts/**", "*.tmp.md", "archive/*"]) == ["keep.md", "notes.md", "sub/a.md", "sub/deep/c.md"]
    assert _names(tmp_path, include=["sub/**"], exclude=["*.tmp.md"]) == ["sub/a.md", "sub/deep/c.md"]


def test_files_over_the_size_limit_raise_or_are_skipped(tmp_path):
    _write(tmp_path, "big.txt", "x" * 2_000_000)
    _write(tmp_path, "small.txt", "fine")

    with pytest.raises(DocumentLoadError, match=r"big\.txt.*2\.0 MB.*max_file_mb"):
        load_documents(tmp_path, max_file_mb=1)
    warnings: list[str] = []
    documents = load_documents(tmp_path, max_file_mb=1, on_error="skip", warnings=warnings)
    assert [Path(d.path).name for d in documents] == ["small.txt"] and len(warnings) == 1 and "big.txt" in warnings[0]


# --- error handling -----------------------------------------------------------------------------


def test_a_corrupt_file_raises_by_default_and_is_skipped_with_a_warning_on_request(tmp_path):
    pytest.importorskip("pypdf")
    _write(tmp_path, "corrupt.pdf", b"this is not a pdf at all")
    _write(tmp_path, "bad.json", "{not json")
    _write(tmp_path, "good.md", "# Good\n\nfine")

    with pytest.raises(DocumentLoadError, match=r"corrupt\.pdf|bad\.json"):
        load_documents(tmp_path)
    warnings: list[str] = []
    documents = load_documents(tmp_path, on_error="skip", warnings=warnings)

    assert [Path(d.path).name for d in documents] == ["good.md"]
    assert len(warnings) == 2 and any("corrupt.pdf" in w for w in warnings) and any("bad.json" in w for w in warnings)


def test_skip_mode_with_nothing_loadable_left_still_fails_clearly(tmp_path):
    _write(tmp_path, "bad.json", "{not json")

    with pytest.raises(ValueError, match="No supported documents"):
        load_documents(tmp_path, on_error="skip")


# --- config, run summary, CLI -------------------------------------------------------------------


def test_dataset_config_accepts_the_loader_options_and_rejects_nonsense(tmp_path):
    config = DatasetConfig(documents_path=tmp_path, questions_path=tmp_path, include=["*.md"], exclude=["drafts/**"], on_error="skip", max_file_mb=5)
    assert (config.include, config.exclude, config.on_error, config.max_file_mb) == (["*.md"], ["drafts/**"], "skip", 5)
    assert DatasetConfig(documents_path=tmp_path, questions_path=tmp_path).on_error == "raise"
    for bad in ({"on_error": "ignore"}, {"max_file_mb": 0}, {"tabular": {"text_colums": ["a"]}}):
        with pytest.raises(ValueError):
            DatasetConfig(documents_path=tmp_path, questions_path=tmp_path, **bad)


def _experiment(tmp_path: Path, **dataset: object) -> Path:
    docs = tmp_path / "docs"
    _write(docs, "doc_001.md", "# Refunds\n\nRefunds are accepted within thirty days of delivery.")
    _write(docs, "doc_002.md", "# Shipping\n\nShipping takes five business days.")
    _write(docs, "broken.json", "{oops")
    _write(docs, "skipme.md", "# Skip\n\nignored by config")
    (tmp_path / "q.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "What is the refund window?",
                "reference_answer": "Thirty days",
                "expected_keywords": ["thirty"],
                "relevant_doc_ids": ["doc_001"],
                "category": "direct_fact",
                "difficulty": "easy",
                "answer_type": "single_fact",
            }
        )
        + "\n"
    )
    config = {
        "run": {"name": "loaders", "output_dir": str(tmp_path / "out")},
        "dataset": {"documents_path": str(docs), "questions_path": str(tmp_path / "q.jsonl"), **dataset},
        "systems": [{"type": "bm25"}],
        "evaluation": {"judge_enabled": False},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_run_honours_the_dataset_options_and_records_document_warnings_in_the_summary(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    out = run_benchmark(_experiment(tmp_path, on_error="skip", exclude=["skip*"]), force_mock=True)

    summary = json.loads((out / "run_summary.json").read_text())
    assert any("broken.json" in w for w in summary["document_warnings"]) and len(summary["document_warnings"]) == 1
    with pytest.raises(DocumentLoadError, match="broken.json"):
        run_benchmark(_experiment(tmp_path / "strict"), force_mock=True)


def test_inspect_dataset_lists_loader_warnings_and_takes_the_loader_options(tmp_path):
    config = yaml.safe_load(_experiment(tmp_path).read_text())["dataset"]
    base = ["inspect-dataset", "--docs", config["documents_path"], "--questions", config["questions_path"]]

    failing = CliRunner().invoke(app, base)
    skipping = CliRunner().invoke(app, [*base, "--on-error", "skip", "--exclude", "skip*"])

    assert failing.exit_code == 2 and "broken.json" in failing.output
    assert skipping.exit_code == 0 and "Documents" in skipping.output and "broken.json" in skipping.output.replace("\n", "")
    assert "skipme" not in skipping.output
