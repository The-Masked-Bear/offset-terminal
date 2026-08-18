"""Document writing.

The only assertion that matters for a file format is that a real reader would
accept it, so these tests unzip the packages and parse every XML part, and walk
the PDF's xref table to check the offsets point at real objects.  A document
that "looks right" in a string comparison but fails to open is worthless.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from offset.tools.base import Danger, ToolContext, Toolbox
from offset.tools.documents import (
    Documents,
    blocks_from,
    document_tools,
    write_docx,
    write_odt,
    write_pdf,
    write_xlsx,
)
from offset.tools.runtime import Approval, Runtime
from offset.providers.base import ToolCall


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path, root=None, timeout=20.0)  # unrestricted, as granted at startup


@pytest.fixture()
def runtime(ctx):
    return Runtime(Toolbox(document_tools()), ctx, Approval(mode="full"))


def call(**args) -> ToolCall:
    return ToolCall(id="c1", name="document", args=args)


BLOCKS = [
    {"kind": "heading", "text": "Findings", "level": 2},
    {"kind": "paragraph", "text": "The parser is the bottleneck."},
    {"kind": "bullets", "items": ["one & only", "two < three", 'quote " here']},
    {"kind": "table", "rows": [["stage", "ms"], ["lex", "12"], ["parse", "480"]]},
    {"kind": "code", "text": "def parse(src):\n    return grammar.match(src)"},
    {"kind": "quote", "text": "Measure, then cut."},
]


def parts(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None, "the archive is corrupt"
        return {n: zf.read(n) for n in zf.namelist()}


# -- the model --------------------------------------------------------------


def test_a_plain_string_becomes_paragraphs():
    got = blocks_from("first para\n\nsecond para")
    assert [b.kind for b in got] == ["paragraph", "paragraph"]
    assert got[1].text == "second para"


def test_an_unknown_block_kind_is_refused():
    with pytest.raises(ValueError) as caught:
        blocks_from([{"kind": "interpretive-dance", "text": "no"}])
    assert "interpretive-dance" in str(caught.value)


def test_heading_levels_are_clamped():
    assert blocks_from([{"kind": "heading", "level": 99, "text": "x"}])[0].level == 6
    assert blocks_from([{"kind": "heading", "level": -3, "text": "x"}])[0].level == 1


# -- docx -------------------------------------------------------------------


def test_docx_is_a_valid_package_that_parses(tmp_path):
    out = tmp_path / "report.docx"
    write_docx(out, blocks_from(BLOCKS), title="Parser Report")
    got = parts(out)

    for required in ("[Content_Types].xml", "_rels/.rels", "word/document.xml",
                    "word/styles.xml", "word/_rels/document.xml.rels"):
        assert required in got, f"{required} missing: a reader will reject this"
    for name, body in got.items():
        ET.fromstring(body)  # every part must be well-formed XML

    document = got["word/document.xml"].decode()
    assert "Parser Report" in document
    assert "The parser is the bottleneck." in document
    assert "<w:tbl>" in document and "</w:tbl>" in document


def test_docx_escapes_xml_metacharacters(tmp_path):
    out = tmp_path / "escapes.docx"
    write_docx(out, blocks_from([{"kind": "paragraph", "text": 'a < b & c > d "q"'}]))
    body = parts(out)["word/document.xml"].decode()
    assert "&lt;" in body and "&amp;" in body
    assert "a < b" not in body, "raw metacharacters would break the part"
    ET.fromstring(parts(out)["word/document.xml"])


def test_docx_relationships_point_at_parts_that_exist(tmp_path):
    out = tmp_path / "rels.docx"
    write_docx(out, blocks_from(["hello"]))
    got = parts(out)
    rels = ET.fromstring(got["_rels/.rels"])
    targets = [r.attrib["Target"].lstrip("/") for r in rels]
    assert targets and all(t in got for t in targets)
    doc_rels = ET.fromstring(got["word/_rels/document.xml.rels"])
    for r in doc_rels:
        assert f"word/{r.attrib['Target']}" in got


def test_docx_declares_a_content_type_for_every_override(tmp_path):
    out = tmp_path / "types.docx"
    write_docx(out, blocks_from(["hi"]))
    got = parts(out)
    types = ET.fromstring(got["[Content_Types].xml"])
    ns = "{http://schemas.openxmlformats.org/package/2006/content-types}"
    for override in types.findall(f"{ns}Override"):
        assert override.attrib["PartName"].lstrip("/") in got


# -- xlsx -------------------------------------------------------------------


def test_xlsx_is_valid_and_keeps_numbers_numeric(tmp_path):
    out = tmp_path / "data.xlsx"
    write_xlsx(out, [["stage", "ms"], ["lex", "12"], ["parse", "480.5"]], sheet="Timing")
    got = parts(out)
    for required in ("[Content_Types].xml", "_rels/.rels", "xl/workbook.xml",
                     "xl/worksheets/sheet1.xml", "xl/_rels/workbook.xml.rels"):
        assert required in got
    for body in got.values():
        ET.fromstring(body)

    sheet = got["xl/worksheets/sheet1.xml"].decode()
    assert 't="inlineStr"' in sheet, "text cells need a type"
    assert "<v>12</v>" in sheet and "<v>480.5</v>" in sheet, "numbers must not be strings"
    assert "Timing" in got["xl/workbook.xml"].decode()


def test_xlsx_column_letters_go_past_z(tmp_path):
    out = tmp_path / "wide.xlsx"
    write_xlsx(out, [[f"c{i}" for i in range(28)]])
    sheet = parts(out)["xl/worksheets/sheet1.xml"].decode()
    assert 'r="Z1"' in sheet and 'r="AA1"' in sheet and 'r="AB1"' in sheet


def test_xlsx_dimension_matches_the_data(tmp_path):
    out = tmp_path / "dim.xlsx"
    write_xlsx(out, [["a", "b", "c"], ["d", "e", "f"]])
    sheet = parts(out)["xl/worksheets/sheet1.xml"].decode()
    assert 'ref="A1:C2"' in sheet


# -- odt --------------------------------------------------------------------


def test_odt_stores_its_mimetype_first_and_uncompressed(tmp_path):
    """ODF requires it; LibreOffice sniffs the first entry."""
    out = tmp_path / "doc.odt"
    write_odt(out, blocks_from(BLOCKS), title="Notes")
    with zipfile.ZipFile(out) as zf:
        first = zf.infolist()[0]
        assert first.filename == "mimetype"
        assert first.compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == b"application/vnd.oasis.opendocument.text"
    for body in parts(out).values():
        if body.lstrip().startswith(b"<"):
            ET.fromstring(body)


def test_odt_contains_the_content(tmp_path):
    out = tmp_path / "doc.odt"
    write_odt(out, blocks_from(BLOCKS), title="Notes")
    content = parts(out)["content.xml"].decode()
    assert "Notes" in content and "bottleneck" in content
    assert "<table:table" in content


# -- pdf --------------------------------------------------------------------


def test_pdf_has_a_valid_header_trailer_and_xref(tmp_path):
    out = tmp_path / "report.pdf"
    write_pdf(out, blocks_from(BLOCKS), title="Parser Report")
    raw = out.read_bytes()

    assert raw.startswith(b"%PDF-1.")
    assert raw.rstrip().endswith(b"%%EOF")

    startxref = int(re.search(rb"startxref\s+(\d+)", raw).group(1))
    assert raw[startxref : startxref + 4] == b"xref", "startxref must point at the table"

    table = raw[startxref:]
    count = int(re.search(rb"xref\s+0\s+(\d+)", table).group(1))
    offsets = [int(m.group(1)) for m in re.finditer(rb"^(\d{10}) \d{5} n", table, re.M)]
    assert len(offsets) == count - 1, "every object needs an xref row"
    for n, off in enumerate(offsets, 1):
        assert raw[off:].startswith(f"{n} 0 obj".encode()), f"xref row {n} points at the wrong byte"

    size = int(re.search(rb"/Size (\d+)", raw).group(1))
    assert size == count


def test_pdf_paginates_long_input(tmp_path):
    out = tmp_path / "long.pdf"
    write_pdf(out, blocks_from([{"kind": "paragraph", "text": f"line {i}"} for i in range(400)]))
    raw = out.read_bytes()
    assert raw.count(b"/Type /Page\n") == 0  # exact form differs; count via /Contents
    pages = raw.count(b"/Type /Page ")
    assert pages > 1, "400 paragraphs must not fit on one page"
    assert re.search(rb"/Count (\d+)", raw).group(1).decode() == str(pages)


def test_pdf_escapes_parentheses(tmp_path):
    out = tmp_path / "parens.pdf"
    write_pdf(out, blocks_from(["a (b) c \\ d"]))
    raw = out.read_bytes()
    assert rb"\(b\)" in raw, "unescaped parentheses corrupt the content stream"


def test_pdf_content_stream_length_is_accurate(tmp_path):
    out = tmp_path / "len.pdf"
    write_pdf(out, blocks_from(["measured"]))
    raw = out.read_bytes()
    for match in re.finditer(rb"<< /Length (\d+) >>\nstream\n(.*?)\nendstream", raw, re.S):
        assert int(match.group(1)) == len(match.group(2)), "a wrong /Length breaks rendering"


# -- the tool ---------------------------------------------------------------


def test_the_tool_writes_every_format(tmp_path, runtime):
    for suffix in (".docx", ".odt", ".pdf", ".md", ".txt", ".html"):
        got = runtime.execute(call(path=f"out{suffix}", title="T", content=BLOCKS))
        assert got.result.ok, f"{suffix}: {got.result.error}"
        assert (tmp_path / f"out{suffix}").stat().st_size > 0
        assert got.result.data["format"] == suffix[1:]


def test_the_tool_writes_a_spreadsheet(tmp_path, runtime):
    got = runtime.execute(call(path="book.xlsx", rows=[["a", "1"], ["b", "2"]]))
    assert got.result.ok, got.result.error
    assert "xl/worksheets/sheet1.xml" in parts(tmp_path / "book.xlsx")


def test_a_spreadsheet_without_rows_is_refused(runtime):
    assert "needs `rows`" in runtime.execute(call(path="empty.xlsx")).result.error


def test_an_unknown_extension_lists_what_is_supported(runtime):
    got = runtime.execute(call(path="thing.wat", text="x"))
    assert not got.result.ok and ".docx" in got.result.error


def test_an_empty_document_is_refused(runtime):
    assert "content" in runtime.execute(call(path="void.md")).result.error


def test_overwrite_can_be_declined(tmp_path, runtime):
    (tmp_path / "keep.md").write_text("original", encoding="utf-8")
    got = runtime.execute(call(path="keep.md", text="new", overwrite=False))
    assert not got.result.ok and "exists" in got.result.error
    assert (tmp_path / "keep.md").read_text() == "original"


def test_it_can_write_outside_the_workspace(tmp_path, runtime):
    """The whole point of Danger.FULL: Documents/, a USB stick, anywhere."""
    elsewhere = tmp_path.parent / f"outside-{tmp_path.name}.docx"
    try:
        got = runtime.execute(call(path=str(elsewhere), title="Outside", text="hello"))
        assert got.result.ok, got.result.error
        assert elsewhere.exists()
        assert "word/document.xml" in parts(elsewhere)
    finally:
        elsewhere.unlink(missing_ok=True)


def test_a_bounded_context_still_refuses_to_escape(tmp_path):
    """Without the startup grant, the boundary holds even for a FULL tool."""
    bounded = Runtime(Toolbox(document_tools()), ToolContext(cwd=tmp_path), Approval(mode="full"))
    got = bounded.execute(call(path="../escaped.md", text="x"))
    assert not got.result.ok and "escape" in got.result.error


def test_the_tool_declares_full_danger():
    assert Documents().danger is Danger.FULL
    assert not Documents().parallel_safe


def test_pdf_text_survives_a_round_trip_through_winansi(tmp_path):
    """Regression: raw UTF-8 in a WinAnsi string turned a bullet into `?`."""
    out = tmp_path / "enc.pdf"
    write_pdf(out, blocks_from([
        {"kind": "bullets", "items": ["first"]},
        {"kind": "paragraph", "text": "caf\u00e9 na\u00efve \u2014 dash, \u201ccurly\u201d, 50% \u00b1 2"},
    ]), title="Encoding")
    raw = out.read_bytes()

    assert rb"\225" in raw, "the bullet (U+2022 -> cp1252 0x95) must be an octal escape"
    assert rb"\351" in raw, "the e-acute in cafe must be escaped too"
    assert "caf\u00e9".encode() not in raw, "raw UTF-8 would be read as WinAnsi and mangled"
    for match in re.finditer(rb"<< /Length (\d+) >>\nstream\n(.*?)\nendstream", raw, re.S):
        assert int(match.group(1)) == len(match.group(2)), "escaping must keep /Length exact"


def test_pdf_replaces_glyphs_the_base_font_cannot_show(tmp_path):
    out = tmp_path / "cjk.pdf"
    write_pdf(out, blocks_from(["japanese: \u65e5\u672c\u8a9e"]))
    raw = out.read_bytes()
    assert b"japanese: ???" in raw, "an unmappable glyph must degrade visibly"
    assert "\u65e5\u672c\u8a9e".encode() not in raw
