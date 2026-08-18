"""Writing real office documents, with no third-party dependencies.

A .docx is a zip of XML parts, an .xlsx is a zip of XML parts, an .odt is a zip
of XML parts with one uncompressed `mimetype` entry first, and a PDF is a small
object graph with a byte-offset table.  All four are entirely reachable from the
standard library, which matters on a Raspberry Pi where `pip install` of a
wheel-less package is a five-minute affair.

The parts written here are the minimum a conforming reader requires — not a
subset that "usually opens".  Word, LibreOffice and Excel all validate the
relationship graph and the content-type declarations, so those are complete
even where they look like boilerplate.

Everything in this module is `Danger.FULL`: it writes wherever it is pointed.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Sequence
from xml.sax.saxutils import escape, quoteattr

from offset.tools.base import Danger, Tool, ToolContext, ToolResult

#: Text is the only thing a caller supplies, so escaping is centralised here.
def x(text: Any) -> str:
    return escape("" if text is None else str(text), {'"': "&quot;", "'": "&apos;"})


# --------------------------------------------------------------------------
# the document model
# --------------------------------------------------------------------------

BLOCK_KINDS: Final = ("heading", "paragraph", "bullets", "numbers", "table", "code", "quote", "break")


@dataclass(slots=True)
class Block:
    kind: str
    text: str = ""
    level: int = 1
    items: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    bold: bool = False
    italic: bool = False

    @classmethod
    def parse(cls, raw: Any) -> "Block":
        if isinstance(raw, str):
            return cls("paragraph", raw)
        if not isinstance(raw, dict):
            raise ValueError("each block must be a string or an object")
        kind = str(raw.get("kind") or raw.get("type") or "paragraph").lower()
        if kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {kind!r}; use one of {', '.join(BLOCK_KINDS)}")
        rows = raw.get("rows") or ()
        return cls(
            kind=kind,
            text=str(raw.get("text") or ""),
            level=max(1, min(int(raw.get("level") or 1), 6)),
            items=tuple(str(i) for i in (raw.get("items") or ())),
            rows=tuple(tuple(str(c) for c in row) for row in rows if isinstance(row, (list, tuple))),
            bold=bool(raw.get("bold")),
            italic=bool(raw.get("italic")),
        )

    def lines(self) -> list[str]:
        """Flat text, for the formats that have no structure of their own."""
        if self.kind == "heading":
            return [self.text.upper(), "=" * len(self.text)] if self.level == 1 else [f"{'#' * self.level} {self.text}"]
        if self.kind in ("bullets", "numbers"):
            mark = (lambda i: "- ") if self.kind == "bullets" else (lambda i: f"{i + 1}. ")
            return [f"{mark(i)}{item}" for i, item in enumerate(self.items)]
        if self.kind == "table":
            return ["  |  ".join(row) for row in self.rows]
        if self.kind == "break":
            return [""]
        if self.kind == "quote":
            return [f"> {self.text}"]
        return [self.text]


def blocks_from(raw: Any) -> list[Block]:
    if isinstance(raw, str):
        # The schema declares `content` as a string, because a bare anyOf is
        # rejected by strict providers. A model that sends the richer block form
        # therefore sends it as JSON text, so accept that here.
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                return [Block.parse(item) for item in decoded]
        # A plain string is a document: split on blank lines into paragraphs.
        chunks = [c.strip() for c in raw.split("\n\n")]
        return [Block("paragraph", c) for c in chunks if c]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("`content` must be a string or a list of blocks")
    return [Block.parse(item) for item in raw]


# --------------------------------------------------------------------------
# zip helper
# --------------------------------------------------------------------------


def _write_zip(path: Path, parts: Sequence[tuple[str, str]], *, first_stored: tuple[str, str] | None = None) -> None:
    """Write a package.  `first_stored` lands uncompressed at offset zero,
    which ODF requires of its `mimetype` entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        if first_stored is not None:
            name, body = first_stored
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, body)
        for name, body in parts:
            zf.writestr(name, body)


# --------------------------------------------------------------------------
# .docx
# --------------------------------------------------------------------------

_DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOCX_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _docx_styles() -> str:
    heads = "".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{n}">'
        f'<w:name w:val="heading {n}"/><w:basedOn w:val="Normal"/>'
        f"<w:pPr><w:keepNext/><w:outlineLvl w:val=\"{n - 1}\"/>"
        f'<w:spacing w:before="{max(120, 360 - n * 40)}" w:after="120"/></w:pPr>'
        f'<w:rPr><w:b/><w:sz w:val="{max(22, 36 - (n - 1) * 4)}"/></w:rPr></w:style>'
        for n in range(1, 7)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f"<w:styles {_W}>"
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
        '<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/>'
        '<w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/><w:basedOn w:val="Normal"/>'
        '<w:pPr><w:ind w:left="720"/></w:pPr><w:rPr><w:i/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="HTML Preformatted"/>'
        '<w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/></w:rPr></w:style>'
        f"{heads}</w:styles>"
    )


def _docx_run(text: str, *, bold: bool = False, italic: bool = False, mono: bool = False) -> str:
    props = ""
    if bold or italic or mono:
        props = "<w:rPr>" + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "")
        props += '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>' if mono else ""
        props += "</w:rPr>"
    return f'{"<w:r>"}{props}<w:t xml:space="preserve">{x(text)}</w:t></w:r>'


def _docx_para(runs: str, *, style: str | None = None) -> str:
    pr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{pr}{runs}</w:p>"


def _docx_body(blocks: Iterable[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        if b.kind == "heading":
            out.append(_docx_para(_docx_run(b.text, bold=True), style=f"Heading{b.level}"))
        elif b.kind == "bullets":
            for item in b.items:
                out.append(_docx_para(_docx_run(f"\u2022  {item}"), style="ListParagraph"))
        elif b.kind == "numbers":
            for i, item in enumerate(b.items, 1):
                out.append(_docx_para(_docx_run(f"{i}.  {item}"), style="ListParagraph"))
        elif b.kind == "quote":
            out.append(_docx_para(_docx_run(b.text, italic=True), style="Quote"))
        elif b.kind == "code":
            for line in b.text.split("\n"):
                out.append(_docx_para(_docx_run(line, mono=True), style="Code"))
        elif b.kind == "break":
            out.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        elif b.kind == "table" and b.rows:
            width = max(len(r) for r in b.rows)
            grid = "".join(f'<w:gridCol w:w="{9360 // max(1, width)}"/>' for _ in range(width))
            rows = []
            for n, row in enumerate(b.rows):
                cells = []
                for c in range(width):
                    value = row[c] if c < len(row) else ""
                    shade = '<w:shd w:val="clear" w:fill="EEEEEE"/>' if n == 0 else ""
                    cells.append(
                        f"<w:tc><w:tcPr>{shade}</w:tcPr>"
                        f"{_docx_para(_docx_run(value, bold=n == 0))}</w:tc>"
                    )
                rows.append(f"<w:tr>{''.join(cells)}</w:tr>")
            out.append(
                "<w:tbl><w:tblPr>"
                '<w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
                '<w:tblBorders>'
                + "".join(
                    f'<w:{side} w:val="single" w:sz="8" w:space="0" w:color="111111"/>'
                    for side in ("top", "left", "bottom", "right", "insideH", "insideV")
                )
                + "</w:tblBorders></w:tblPr>"
                f"<w:tblGrid>{grid}</w:tblGrid>{''.join(rows)}</w:tbl>"
            )
        else:
            out.append(_docx_para(_docx_run(b.text, bold=b.bold, italic=b.italic)))
    return "".join(out)


def write_docx(path: Path, blocks: Sequence[Block], *, title: str = "") -> None:
    heading = [Block("heading", title, 1)] if title else []
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f"<w:document {_W}><w:body>"
        f"{_docx_body([*heading, *blocks])}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        "</w:body></w:document>"
    )
    _write_zip(path, [
        ("[Content_Types].xml", _DOCX_CONTENT_TYPES),
        ("_rels/.rels", _DOCX_RELS),
        ("word/_rels/document.xml.rels", _DOCX_DOC_RELS),
        ("word/styles.xml", _docx_styles()),
        ("word/document.xml", document),
    ])


# --------------------------------------------------------------------------
# .xlsx
# --------------------------------------------------------------------------

_XLSX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""

_XLSX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_XLSX_WB_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""

_S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _column(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _numeric(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def write_xlsx(path: Path, rows: Sequence[Sequence[str]], *, sheet: str = "Sheet1") -> None:
    """Inline strings, so there is no shared-string table to keep consistent."""
    body = []
    for r, row in enumerate(rows, 1):
        cells = []
        for c, value in enumerate(row):
            ref = f"{_column(c)}{r}"
            text = "" if value is None else str(value)
            if text == "":
                continue
            if _numeric(text):
                cells.append(f'<c r="{ref}"><v>{x(text)}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{x(text)}</t></is></c>')
        body.append(f'<row r="{r}">{"".join(cells)}</row>')
    widest = max((len(r) for r in rows), default=1)
    dimension = f"A1:{_column(max(0, widest - 1))}{max(1, len(rows))}"
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f"<worksheet {_S}><dimension ref={quoteattr(dimension)}/>"
        f"<sheetData>{''.join(body)}</sheetData></worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f"<workbook {_S} {_R}><sheets>"
        f"<sheet name={quoteattr(sheet[:31] or 'Sheet1')} sheetId=\"1\" r:id=\"rId1\"/>"
        "</sheets></workbook>"
    )
    _write_zip(path, [
        ("[Content_Types].xml", _XLSX_CONTENT_TYPES),
        ("_rels/.rels", _XLSX_RELS),
        ("xl/_rels/workbook.xml.rels", _XLSX_WB_RELS),
        ("xl/workbook.xml", workbook),
        ("xl/worksheets/sheet1.xml", worksheet),
    ])


# --------------------------------------------------------------------------
# .odt
# --------------------------------------------------------------------------

_ODT_MANIFEST = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>
<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

_ODF_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"'
)


def write_odt(path: Path, blocks: Sequence[Block], *, title: str = "") -> None:
    body: list[str] = []
    if title:
        body.append(f'<text:h text:outline-level="1">{x(title)}</text:h>')
    for b in blocks:
        if b.kind == "heading":
            body.append(f'<text:h text:outline-level="{b.level}">{x(b.text)}</text:h>')
        elif b.kind in ("bullets", "numbers"):
            tag = "text:list"
            items = "".join(f"<text:list-item><text:p>{x(i)}</text:p></text:list-item>" for i in b.items)
            body.append(f"<{tag}>{items}</{tag}>")
        elif b.kind == "table" and b.rows:
            width = max(len(r) for r in b.rows)
            cols = f'<table:table-column table:number-columns-repeated="{width}"/>'
            rows = "".join(
                "<table:table-row>"
                + "".join(
                    f"<table:table-cell office:value-type=\"string\"><text:p>"
                    f"{x(row[c] if c < len(row) else '')}</text:p></table:table-cell>"
                    for c in range(width)
                )
                + "</table:table-row>"
                for row in b.rows
            )
            body.append(f'<table:table table:name="Table1">{cols}{rows}</table:table>')
        elif b.kind == "code":
            for line in b.text.split("\n"):
                body.append(f'<text:p text:style-name="Preformatted">{x(line)}</text:p>')
        elif b.kind == "break":
            body.append("<text:p/>")
        else:
            body.append(f"<text:p>{x(b.text)}</text:p>")
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<office:document-content {_ODF_NS} office:version="1.2">'
        f"<office:body><office:text>{''.join(body)}</office:text></office:body>"
        "</office:document-content>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<office:document-styles {_ODF_NS} office:version="1.2">'
        "<office:styles>"
        '<style:style style:name="Preformatted" style:family="paragraph">'
        '<style:text-properties style:font-name="Liberation Mono"/></style:style>'
        "</office:styles></office:document-styles>"
    )
    _write_zip(
        path,
        [("META-INF/manifest.xml", _ODT_MANIFEST), ("content.xml", content), ("styles.xml", styles)],
        first_stored=("mimetype", "application/vnd.oasis.opendocument.text"),
    )


# --------------------------------------------------------------------------
# .pdf
# --------------------------------------------------------------------------

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN = 56
LEADING = 15


def _pdf_escape(text: str) -> str:
    """Encode for a WinAnsi Type1 string literal.

    Everything above ASCII becomes an octal escape, because the raw UTF-8 bytes
    would be read as WinAnsi and come out as mojibake - a bullet arrived as `?`
    before this existed. Escaping also keeps the byte count equal to the string
    length, which is what makes the stream's /Length exact.
    """
    out: list[str] = []
    for ch in text:
        if ch in "\\()":
            out.append("\\" + ch)
            continue
        if ch in "\r\n\t":
            out.append(" ")
            continue
        try:
            code = ch.encode("cp1252")[0]
        except (UnicodeEncodeError, IndexError):
            code = 0x3F  # '?', the honest stand-in for an unmappable glyph
        out.append(chr(code) if 32 <= code < 127 else f"\\{code:03o}")
    return "".join(out)


def _pdf_wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        line = ""
        for word in para.split(" "):
            while len(word) > width:
                if line:
                    out.append(line)
                    line = ""
                out.append(word[:width])
                word = word[width:]
            if len(line) + len(word) + (1 if line else 0) <= width:
                line += (" " if line else "") + word
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def _pdf_lines(blocks: Sequence[Block], title: str) -> list[tuple[str, str]]:
    """(style, text) pairs, where style picks the font and size."""
    rows: list[tuple[str, str]] = []
    if title:
        rows.append(("h1", title))
        rows.append(("gap", ""))
    for b in blocks:
        if b.kind == "heading":
            rows.append(("gap", ""))
            rows.append((f"h{min(b.level, 3)}", b.text))
        elif b.kind in ("bullets", "numbers"):
            for i, item in enumerate(b.items, 1):
                mark = "\u2022  " if b.kind == "bullets" else f"{i}.  "
                for n, line in enumerate(_pdf_wrap(mark + item, 84)):
                    rows.append(("body", line if n == 0 else "   " + line))
        elif b.kind == "table":
            for n, row in enumerate(b.rows):
                rows.append(("mono" if n else "th", "  |  ".join(row)))
        elif b.kind == "code":
            for line in b.text.split("\n"):
                rows.append(("mono", line))
        elif b.kind == "break":
            rows.append(("page", ""))
        elif b.kind == "quote":
            for line in _pdf_wrap(b.text, 80):
                rows.append(("quote", "   " + line))
        else:
            for line in _pdf_wrap(b.text, 88):
                rows.append(("body", line))
        rows.append(("gap", ""))
    return rows


_FONTS: Final = {
    "h1": ("F2", 20), "h2": ("F2", 15), "h3": ("F2", 12.5), "th": ("F2", 10),
    "body": ("F1", 11), "quote": ("F3", 11), "mono": ("F4", 9.5), "gap": ("F1", 6),
}


def write_pdf(path: Path, blocks: Sequence[Block], *, title: str = "") -> None:
    """A real PDF: object graph, exact xref offsets, valid trailer."""
    rows = _pdf_lines(blocks, title)
    pages: list[list[tuple[str, str]]] = [[]]
    y = PAGE_H - MARGIN
    for style, text in rows:
        if style == "page" or y < MARGIN + LEADING:
            pages.append([])
            y = PAGE_H - MARGIN
            if style == "page":
                continue
        size = _FONTS.get(style, _FONTS["body"])[1]
        step = LEADING if style not in ("h1", "h2") else LEADING + size * 0.5
        pages[-1].append((style, text))
        y -= step
    if len(pages) > 1 and not pages[-1]:
        pages.pop()

    streams: list[str] = []
    for page in pages:
        parts = ["BT"]
        cursor = PAGE_H - MARGIN
        for style, text in page:
            font, size = _FONTS.get(style, _FONTS["body"])
            step = LEADING if style not in ("h1", "h2") else LEADING + size * 0.5
            parts.append(f"/{font} {size:g} Tf")
            parts.append(f"1 0 0 1 {MARGIN} {cursor:.1f} Tm")
            if text:
                parts.append(f"({_pdf_escape(text)}) Tj")
            cursor -= step
        parts.append("ET")
        streams.append("\n".join(parts))

    # object numbering: 1 catalog, 2 pages, 3..6 fonts, then page+content pairs
    first_page_obj = 7
    kids = " ".join(f"{first_page_obj + 2 * i} 0 R" for i in range(len(pages)))
    objects: list[str] = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
    ]
    for i, stream in enumerate(streams):
        content_obj = first_page_obj + 2 * i + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R /F4 6 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream.encode('latin-1', 'replace'))} >>\nstream\n{stream}\nendstream")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1", "replace")
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


# --------------------------------------------------------------------------
# plain formats
# --------------------------------------------------------------------------


def write_text(path: Path, blocks: Sequence[Block], *, title: str = "") -> None:
    lines: list[str] = []
    if title:
        lines += [title.upper(), "=" * len(title), ""]
    for b in blocks:
        lines += b.lines()
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_markdown(path: Path, blocks: Sequence[Block], *, title: str = "") -> None:
    lines: list[str] = []
    if title:
        lines += [f"# {title}", ""]
    for b in blocks:
        if b.kind == "heading":
            lines.append(f"{'#' * min(b.level + 1, 6)} {b.text}")
        elif b.kind == "bullets":
            lines += [f"- {i}" for i in b.items]
        elif b.kind == "numbers":
            lines += [f"{n}. {i}" for n, i in enumerate(b.items, 1)]
        elif b.kind == "code":
            lines += ["```", *b.text.split("\n"), "```"]
        elif b.kind == "quote":
            lines.append(f"> {b.text}")
        elif b.kind == "table" and b.rows:
            width = max(len(r) for r in b.rows)
            head = list(b.rows[0]) + [""] * (width - len(b.rows[0]))
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "|".join([" --- "] * width) + "|")
            for row in b.rows[1:]:
                padded = list(row) + [""] * (width - len(row))
                lines.append("| " + " | ".join(padded) + " |")
        else:
            lines.append(b.text)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, blocks: Sequence[Block], *, title: str = "") -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for b in blocks:
            if b.kind == "table":
                writer.writerows(b.rows)
            elif b.items:
                writer.writerows([[i] for i in b.items])
            elif b.text:
                writer.writerow([b.text])


def write_html(path: Path, blocks: Sequence[Block], *, title: str = "") -> None:
    body: list[str] = []
    if title:
        body.append(f"<h1>{x(title)}</h1>")
    for b in blocks:
        if b.kind == "heading":
            body.append(f"<h{min(b.level + 1, 6)}>{x(b.text)}</h{min(b.level + 1, 6)}>")
        elif b.kind in ("bullets", "numbers"):
            tag = "ul" if b.kind == "bullets" else "ol"
            body.append(f"<{tag}>" + "".join(f"<li>{x(i)}</li>" for i in b.items) + f"</{tag}>")
        elif b.kind == "code":
            body.append(f"<pre><code>{x(b.text)}</code></pre>")
        elif b.kind == "quote":
            body.append(f"<blockquote>{x(b.text)}</blockquote>")
        elif b.kind == "table" and b.rows:
            head = "".join(f"<th>{x(c)}</th>" for c in b.rows[0])
            rest = "".join("<tr>" + "".join(f"<td>{x(c)}</td>" for c in r) + "</tr>" for r in b.rows[1:])
            body.append(f"<table border=1 cellspacing=0 cellpadding=4><tr>{head}</tr>{rest}</table>")
        else:
            body.append(f"<p>{x(b.text)}</p>")
    page = (
        "<!doctype html><html><head><meta charset=utf-8>"
        f"<title>{x(title or path.stem)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:44rem;margin:3rem auto;padding:0 1rem}"
        "pre{background:#f4f4f0;padding:.75rem;border:2px solid #111}"
        "blockquote{border-left:4px solid #111;margin:0;padding-left:1rem;font-style:italic}</style>"
        f"</head><body>{''.join(body)}</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


WRITERS: Final[dict[str, Any]] = {
    ".docx": write_docx,
    ".xlsx": None,  # handled specially: it takes rows, not blocks
    ".odt": write_odt,
    ".pdf": write_pdf,
    ".md": write_markdown,
    ".markdown": write_markdown,
    ".txt": write_text,
    ".text": write_text,
    ".csv": None,
    ".html": write_html,
    ".htm": write_html,
}

FORMATS: Final = tuple(sorted(WRITERS))


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


class Documents(Tool):
    """Create a real office document anywhere on the machine.

    `Danger.FULL` because the whole point is that it can write to Documents/,
    a USB stick, or anywhere else the user asked for - not just the workspace.
    """

    name = "document"
    description = (
        "Create a document: .docx .xlsx .odt .pdf .md .txt .csv .html. "
        "Pass `text` for plain prose, or `content` as a list of blocks "
        "({kind: heading|paragraph|bullets|numbers|table|code|quote|break, ...})."
    )
    danger = Danger.FULL
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "destination; the extension picks the format"},
            "title": {"type": "string"},
            # Two well-typed fields rather than one union. `content` used to be
            # a bare `anyOf`, which strict providers reject - and because every
            # tool ships on every request, that one schema failed every message.
            "text": {
                "type": "string",
                "description": "the body as plain prose; blank lines separate paragraphs",
            },
            "content": {
                "type": "array",
                "description": "the body as structured blocks; use instead of `text`",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(BLOCK_KINDS)},
                        "text": {"type": "string"},
                        "level": {"type": "integer", "minimum": 1, "maximum": 6},
                        "items": {"type": "array", "items": {"type": "string"}},
                        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                    },
                },
            },
            "rows": {
                "type": "array",
                "description": "for .xlsx/.csv: rows of cells",
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "sheet": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        "required": ["path"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args["path"])
        suffix = target.suffix.lower()
        if suffix not in WRITERS:
            return ToolResult.fail(f"unsupported format {suffix or '(none)'}; use one of {', '.join(FORMATS)}")
        if target.exists() and not args.get("overwrite", True):
            return ToolResult.fail(f"{target} exists and overwrite is false")

        body = args.get("content") or args.get("text") or ""
        rows = args.get("rows")
        try:
            if suffix == ".xlsx":
                if not rows:
                    return ToolResult.fail("an .xlsx needs `rows`")
                write_xlsx(target, [[str(c) for c in row] for row in rows], sheet=str(args.get("sheet") or "Sheet1"))
            elif suffix == ".csv":
                blocks = [Block("table", rows=tuple(tuple(str(c) for c in r) for r in rows))] if rows \
                    else blocks_from(body)
                write_csv(target, blocks)
            else:
                blocks = blocks_from(body)
                if not blocks and not args.get("title"):
                    return ToolResult.fail("give the document some `text`, `content`, or a `title`")
                WRITERS[suffix](target, blocks, title=str(args.get("title") or ""))
        except (ValueError, TypeError) as exc:
            return ToolResult.fail(f"could not build the document: {exc}")
        except OSError as exc:
            return ToolResult.fail(f"could not write {target}: {exc}")

        size = target.stat().st_size
        return ToolResult(
            content=f"wrote {target} ({size} bytes, {suffix[1:]})",
            display=f"document {target.name} ({size} bytes)",
            data={"path": str(target), "bytes": size, "format": suffix[1:]},
        )


def document_tools() -> list[Tool]:
    return [Documents()]
