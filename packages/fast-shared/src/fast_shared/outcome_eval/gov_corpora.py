"""Paired authoritative U.S. Government PDF <-> XML / HTML / XLSX / transcription benchmark builder & loader.

Builds deterministic, SHA-256 verified page benchmarks across all three U.S. federal branches
using keyless Public-Domain sources:
- Legislative: GovInfo ``PLAW`` (USLM XML + PDF), ``BILLS`` (XML + PDF), and ``USCODE`` (HTML + PDF)
- Executive: Federal Register API + GovInfo ``FR`` (3-column PDF + GPO XML / ``GPOTABLE``),
  GovInfo ``BUDGET`` OMB Analytical Perspectives (single-page PDF <-> official ``.xlsx`` tables),
  and LoC ``By the People`` (``btp-data`` curated archival transcriptions + scans)
- Judicial: GovInfo ``USCOURTS`` born-digital CM/ECF opinions + U.S. Supreme Court slip opinions
  (``incrediblecrab/us-scotus-opinions``, byte-for-byte ``supremecourt.gov`` PDFs)
"""

from __future__ import annotations

import hashlib
import html as _html
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from .hub_loader import compute_file_sha256, resolve_cache_dir, stratified_sha256_sample

_USER_AGENT = "fast-silicon-outcome-eval/1.0"
_FR_API_URL = (
    "https://www.federalregister.gov/api/v1/documents.json"
    "?per_page=25&fields[]=document_number&fields[]=title&fields[]=type"
    "&fields[]=pdf_url&fields[]=full_text_xml_url&fields[]=page_length"
)
_LOC_BTP_CSV_BLOB_URL = "https://api.github.com/repositories/747276533/git/blobs/d989e78ac45299052edf77068898299a0bcba750"
_LOC_BTP_CSV_SHA256 = "31a0cd667e13150ad8bfdb06db8391f314abd295dc97e162003a73c3669ecc36"
_SCOTUS_REPO_REV = "be78dadb26ee80559b98af1258cb42d94afd7e0a"
_SCOTUS_PARQUETS: tuple[tuple[str, str], ...] = (
    ("opinions-of-the-court/data/OT2026.parquet", "e624d6af6e7b82f3197085376cb3d9e93771fe9b50ebcb1a452fa89fc6593a11"),
    ("opinions-of-the-court/data/OT2025.parquet", "b209a7223fdfce86722680d55d8b068fdf37fbd747b40592b451d845e329fad3"),
)
_BUDGET_CANDIDATE_GRANULES: tuple[tuple[str, str], ...] = (
    ("BUDGET-2027-PER", "BUDGET-2027-PER-4-6"),
    ("BUDGET-2027-PER", "BUDGET-2027-PER-4-9"),
    ("BUDGET-2027-PER", "BUDGET-2027-PER-3-3"),
    ("BUDGET-2027-PER", "BUDGET-2027-PER-4-7"),
    ("BUDGET-2027-PER", "BUDGET-2027-PER-4-8"),
)

_LEAF_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "AGENCY",
        "SUBAGY",
        "CFR",
        "DEPDOC",
        "RIN",
        "SUBJECT",
        "HD",
        "P",
        "FP",
        "FTNT",
        "TITLE",
        "SUBTITLE",
        "PART",
        "SUBPART",
        "EXTRACT",
        "NOTE",
        "p",
        "content",
        "heading",
        "header",
        "num",
        "chapeau",
        "continuation",
        "text",
        "proviso",
        "official-title",
        "officialTitle",
        "enactingFormula",
        "legis-num",
        "docTitle",
        "longTitle",
        "purpose",
        "quoted-block",
    }
)
_HEADING_TAGS: frozenset[str] = frozenset(
    {"SUBJECT", "HD", "TITLE", "SUBTITLE", "heading", "header", "official-title", "officialTitle", "docTitle"}
)
_SKIP_XML_SUBTREES: frozenset[str] = frozenset({"meta", "metadata", "dublinCore", "PREMIS", "mods"})


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def gpotable_to_html(table_elem: ET.Element) -> str:
    """Convert a GovInfo Federal Register ``<GPOTABLE>`` XML subtree into normalized HTML ``<table>``."""
    parts: list[str] = ["<table>"]

    def _emit_row(cells: Sequence[tuple[str, int]], tag_name: str) -> None:
        if not cells:
            return
        row_s = ["<tr>"]
        for txt, span in cells:
            esc = _html.escape(_clean_ws(txt))
            if span > 1:
                row_s.append(f'<{tag_name} colspan="{span}">{esc}</{tag_name}>')
            else:
                row_s.append(f"<{tag_name}>{esc}</{tag_name}>")
        row_s.append("</tr>")
        parts.append("".join(row_s))

    for child in table_elem:
        ctag = _local_tag(child.tag)
        if ctag == "BOXHD":
            hdr_cells: list[tuple[str, int]] = []
            for hcell in child:
                if _local_tag(hcell.tag) == "CHED":
                    hdr_cells.append(("".join(hcell.itertext()), 1))
            _emit_row(hdr_cells, "th")
        elif ctag == "ROW":
            exp_raw = child.attrib.get("EXPSTB")
            span_all = (int(exp_raw) + 1) if (exp_raw and exp_raw.isdigit()) else 1
            row_cells: list[tuple[str, int]] = []
            ents = [e for e in child if _local_tag(e.tag) == "ENT"]
            for idx, ent in enumerate(ents):
                sp = span_all if (len(ents) == 1 and span_all > 1 and idx == 0) else 1
                row_cells.append(("".join(ent.itertext()), sp))
            _emit_row(row_cells, "td")

    parts.append("</table>")
    return "".join(parts)


def _html_table_elem_to_str(table_elem: ET.Element) -> str:
    """Serialize a standard XHTML/USLM ``<table>`` subtree preserving cell text and spans."""
    parts: list[str] = ["<table>"]
    for tr in table_elem.iter():
        if _local_tag(tr.tag) != "tr":
            continue
        cells_out: list[str] = []
        for cell in tr:
            ctag = _local_tag(cell.tag).lower()
            if ctag not in ("td", "th"):
                continue
            attrs: list[str] = []
            for ak in ("colspan", "rowspan"):
                av = cell.attrib.get(ak)
                if av and str(av).isdigit() and int(av) > 1:
                    attrs.append(f'{ak}="{int(av)}"')
            attr_s = (" " + " ".join(attrs)) if attrs else ""
            cells_out.append(f"<{ctag}{attr_s}>{_html.escape(_clean_ws(''.join(cell.itertext())))}</{ctag}>")
        if cells_out:
            parts.append("<tr>" + "".join(cells_out) + "</tr>")
    parts.append("</table>")
    return "".join(parts)


def parse_govinfo_xml_gold(xml_text: str) -> dict[str, Any]:
    """Parse Federal Register, USLM Public Law, or Bill XML into ordered ``text_blocks`` and ``tables``."""
    cleaned = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text or "", flags=re.IGNORECASE)
    root = ET.fromstring(cleaned)

    text_blocks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    order_idx = 0

    def _walk(node: ET.Element) -> None:
        nonlocal order_idx
        tag = _local_tag(node.tag)
        if tag in _SKIP_XML_SUBTREES:
            return
        if tag == "GPOTABLE":
            html_tbl = gpotable_to_html(node)
            if "<tr>" in html_tbl:
                tables.append({"order": order_idx, "html": html_tbl})
                order_idx += 1
            return
        if tag.lower() == "table":
            html_tbl = _html_table_elem_to_str(node)
            if "<tr>" in html_tbl:
                tables.append({"order": order_idx, "html": html_tbl})
                order_idx += 1
            return

        if tag in _LEAF_BLOCK_TAGS:
            has_nested_leaf = any(_local_tag(ch.tag) in (_LEAF_BLOCK_TAGS | {"GPOTABLE", "table"}) for ch in node)
            if not has_nested_leaf:
                txt = _clean_ws("".join(node.itertext()))
                if txt:
                    ctype = "title" if tag in _HEADING_TAGS else "text_block"
                    text_blocks.append({"category_type": ctype, "order": order_idx, "text": txt})
                    order_idx += 1
                return
            lead_txt = _clean_ws(node.text or "")
            if lead_txt:
                ctype = "title" if tag in _HEADING_TAGS else "text_block"
                text_blocks.append({"category_type": ctype, "order": order_idx, "text": lead_txt})
                order_idx += 1

        for child in node:
            _walk(child)

    _walk(root)
    return {"text_blocks": text_blocks, "tables": tables, "equations": []}


def extract_born_digital_pdf_page_blocks(pdf_path: str | Path, *, page_index: int = 0) -> tuple[int, str, list[dict[str, Any]]]:
    """Extract embedded text and paragraph blocks from a born-digital PDF page via ``pypdfium2``."""
    import pypdfium2 as pdfium  # type: ignore[import-not-found]

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        n_pages = len(doc)
        if page_index < 0 or page_index >= n_pages:
            raise IndexError(f"page_index={page_index} out of range for {pdf_path} (n_pages={n_pages})")
        page = doc[page_index]
        tp = page.get_textpage()
        raw_text = tp.get_text_bounded() or ""
        tp.close()
        page.close()
    finally:
        doc.close()

    blocks: list[dict[str, Any]] = []
    paras = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", raw_text) if p.strip()]
    if not paras and raw_text.strip():
        paras = [raw_text.strip()]
    for idx, p in enumerate(paras):
        cleaned = _clean_ws(p)
        if cleaned:
            blocks.append({"category_type": "text_block", "order": idx, "text": cleaned})
    return n_pages, raw_text, blocks


def slice_gold_to_page_prefix(
    gold: Mapping[str, Any],
    page_native_text: str,
    *,
    n_pdf_pages: int,
) -> dict[str, Any]:
    """Keep only gold ``text_blocks`` & ``tables`` matching the rendered page when a PDF has >1 pages."""
    if n_pdf_pages <= 1 or not page_native_text.strip():
        return {
            "text_blocks": list(gold.get("text_blocks") or []),
            "tables": list(gold.get("tables") or []),
            "equations": list(gold.get("equations") or []),
        }

    norm_page = re.sub(r"\s+", " ", page_native_text).strip().lower()
    kept_blocks: list[dict[str, Any]] = []
    for blk in gold.get("text_blocks") or []:
        txt = _clean_ws(str(blk.get("text") or ""))
        if not txt:
            continue
        probe = txt[:48].lower()
        if len(probe) >= 12 and probe in norm_page:
            kept_blocks.append(dict(blk))
        elif len(txt) < 12 and txt.lower() in norm_page:
            kept_blocks.append(dict(blk))

    kept_tables: list[dict[str, Any]] = []
    for tbl in gold.get("tables") or []:
        plain = re.sub(r"<[^>]+>", " ", str(tbl.get("html") or ""))
        toks = [w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", plain)][:6]
        if toks and sum(1 for tok in toks if tok in norm_page) >= max(1, (len(toks) + 1) // 2):
            kept_tables.append(dict(tbl))

    if not kept_blocks and (gold.get("text_blocks") or []):
        kept_blocks = [dict(b) for b in (gold.get("text_blocks") or [])[:4]]
    return {"text_blocks": kept_blocks, "tables": kept_tables, "equations": list(gold.get("equations") or [])}


class _USCodeHTMLParser(HTMLParser):
    """Parse GPO U.S. Code section/chapter HTML into ordered headings, paragraphs, and tables."""

    def __init__(self, *, wait_for_doc_comment: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.started = not wait_for_doc_comment
        self._skip_depth = 0
        self._block_tag: str | None = None
        self._block_buf: list[str] = []
        self._in_table = False
        self._curr_tr: list[tuple[str, str, int, int]] = []
        self._rows: list[list[tuple[str, str, int, int]]] = []
        self._cell_tag: str | None = None
        self._cell_buf: list[str] = []
        self._cell_spans: tuple[int, int] = (1, 1)
        self.text_blocks: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self._order = 0

    def handle_comment(self, data: str) -> None:
        if "documentid:" in data:
            self.started = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("head", "style", "script"):
            self._skip_depth += 1
            return
        if self._skip_depth or not self.started:
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if t == "table":
            self._in_table = True
            self._rows = []
            return
        if self._in_table:
            if t == "tr":
                self._curr_tr = []
            elif t in ("th", "td"):
                self._cell_tag = t
                self._cell_buf = []
                cs = int(attr_map["colspan"]) if attr_map.get("colspan", "").isdigit() else 1
                rs = int(attr_map["rowspan"]) if attr_map.get("rowspan", "").isdigit() else 1
                self._cell_spans = (max(1, cs), max(1, rs))
            return
        if t in ("h3", "h4", "p"):
            self._block_tag = t
            self._block_buf = []
        elif t == "br" and self._block_tag is not None:
            self._block_buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("head", "style", "script") and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth or not self.started:
            return
        if self._in_table:
            if t in ("th", "td") and self._cell_tag == t:
                self._curr_tr.append((t, _clean_ws("".join(self._cell_buf)), self._cell_spans[0], self._cell_spans[1]))
                self._cell_tag = None
            elif t == "tr" and self._curr_tr:
                self._rows.append(self._curr_tr)
                self._curr_tr = []
            elif t == "table":
                if self._rows:
                    parts = ["<table>"]
                    for row in self._rows:
                        parts.append("<tr>")
                        for ctag, txt, cs, rs in row:
                            attrs_s = ""
                            if cs > 1:
                                attrs_s += f' colspan="{cs}"'
                            if rs > 1:
                                attrs_s += f' rowspan="{rs}"'
                            parts.append(f"<{ctag}{attrs_s}>{_html.escape(txt)}</{ctag}>")
                        parts.append("</tr>")
                    parts.append("</table>")
                    self.tables.append({"order": self._order, "html": "".join(parts)})
                    self._order += 1
                self._in_table = False
                self._rows = []
            return
        if t == self._block_tag:
            txt = _clean_ws("".join(self._block_buf))
            if txt:
                ctype = "title" if t in ("h3", "h4") else "text_block"
                self.text_blocks.append({"category_type": ctype, "order": self._order, "text": txt})
                self._order += 1
            self._block_tag = None
            self._block_buf = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self.started:
            return
        if self._cell_tag is not None:
            self._cell_buf.append(data)
        elif self._block_tag is not None:
            self._block_buf.append(data)


def parse_uscode_html_gold(html_text: str) -> dict[str, Any]:
    """Parse GovInfo U.S. Code ``.htm`` into ordered ``text_blocks`` and ``tables``."""
    raw = html_text or ""
    parser = _USCodeHTMLParser(wait_for_doc_comment=("documentid:" in raw))
    parser.feed(raw)
    parser.close()
    return {"text_blocks": parser.text_blocks, "tables": parser.tables, "equations": []}


def _xlsx_col_row(ref: str) -> tuple[int, int] | None:
    m = re.match(r"^([A-Z]+)(\d+)$", (ref or "").strip().upper())
    if not m:
        return None
    col = 0
    for ch in m.group(1):
        col = col * 26 + (ord(ch) - 64)
    return int(m.group(2)) - 1, col - 1


def parse_budget_xlsx_tables(xlsx_bytes: bytes) -> list[dict[str, Any]]:
    """Extract compact single-table spreadsheets from an OMB ``.xlsx`` workbook into HTML ``<table>`` gold."""
    out: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as xzf:
        wb = ET.fromstring(xzf.read("xl/workbook.xml"))
        rel_targets: dict[str, str] = {}
        if "xl/_rels/workbook.xml.rels" in xzf.namelist():
            rels = ET.fromstring(xzf.read("xl/_rels/workbook.xml.rels"))
            for rel in rels:
                rid = rel.attrib.get("Id")
                tgt = rel.attrib.get("Target", "").lstrip("/")
                if rid and tgt:
                    rel_targets[rid] = tgt if tgt.startswith("xl/") else f"xl/{tgt}"

        ss: list[str] = []
        if "xl/sharedStrings.xml" in xzf.namelist():
            sst = ET.fromstring(xzf.read("xl/sharedStrings.xml"))
            for si in sst:
                if _local_tag(si.tag) == "si":
                    ss.append("".join(t.text or "" for t in si.iter() if _local_tag(t.tag) == "t"))

        for idx, sh in enumerate([el for el in wb.iter() if _local_tag(el.tag) == "sheet"], 1):
            sname = sh.attrib.get("name", f"sheet{idx}")
            low_name = sname.lower()
            if any(bad in low_name for bad in ("web only", "data for", "info", "contents")):
                continue
            rid = next((v for k, v in sh.attrib.items() if k == "id" or k.endswith("}id")), "")
            sp = rel_targets.get(rid, f"xl/worksheets/sheet{idx}.xml")
            if sp not in xzf.namelist():
                continue

            root = ET.fromstring(xzf.read(sp))
            spans: dict[tuple[int, int], tuple[int, int]] = {}
            covered: set[tuple[int, int]] = set()
            for mc in root.iter():
                if _local_tag(mc.tag) != "mergeCell" or ":" not in mc.attrib.get("ref", ""):
                    continue
                p_a, p_b = mc.attrib["ref"].split(":", 1)
                c_a = _xlsx_col_row(p_a)
                c_b = _xlsx_col_row(p_b)
                if c_a and c_b:
                    r0, col0 = c_a
                    r1, col1 = c_b
                    spans[(r0, col0)] = (r1 - r0 + 1, col1 - col0 + 1)
                    for rr in range(r0, r1 + 1):
                        for cc in range(col0, col1 + 1):
                            if (rr, cc) != (r0, col0):
                                covered.add((rr, cc))

            grid_rows: list[tuple[int, dict[int, str]]] = []
            for r_idx_fallback, r_el in enumerate(el for el in root.iter() if _local_tag(el.tag) == "row"):
                r_idx = int(r_el.attrib["r"]) - 1 if str(r_el.attrib.get("r", "")).isdigit() else r_idx_fallback
                rmap: dict[int, str] = {}
                for c_el in r_el:
                    if _local_tag(c_el.tag) != "c":
                        continue
                    rc = _xlsx_col_row(c_el.attrib.get("r", ""))
                    if rc is None:
                        continue
                    r_idx = rc[0]
                    col_idx = rc[1]
                    t_attr = c_el.attrib.get("t")
                    v_el = next((ch for ch in c_el if _local_tag(ch.tag) == "v"), None)
                    if t_attr == "inlineStr":
                        val = "".join(t.text or "" for t in c_el.iter() if _local_tag(t.tag) == "t")
                    elif v_el is None or v_el.text is None:
                        val = ""
                    elif t_attr == "s":
                        val = ss[int(v_el.text)]
                    else:
                        val = v_el.text
                    val = _clean_ws(re.sub(r"\.{3,}", "", val))
                    if val:
                        rmap[col_idx] = val
                if rmap:
                    grid_rows.append((r_idx, rmap))

            if len(grid_rows) < 3 or len(grid_rows) > 65:
                continue
            head_title = " ".join(v for _, v in sorted(grid_rows[0][1].items()))
            m_tbl = re.search(r"Table\s+(\d+)[–-](\d+)", head_title, re.I)
            if not m_tbl:
                continue
            maj, min_ = m_tbl.group(1), m_tbl.group(2)

            # Separate pre-table caption/unit rows and trailing footnotes from rectangular table body
            body_start = 1
            text_blocks: list[dict[str, Any]] = [{"category_type": "title", "order": 0, "text": head_title}]
            while body_start < len(grid_rows):
                row_vals = [v for _, v in sorted(grid_rows[body_start][1].items())]
                if len(row_vals) == 1 and row_vals[0].startswith("(") and row_vals[0].endswith(")"):
                    text_blocks.append({"category_type": "text_block", "order": len(text_blocks), "text": row_vals[0]})
                    body_start += 1
                elif len(grid_rows[body_start][1]) <= 1 and body_start < 2:
                    text_blocks.append({"category_type": "title", "order": len(text_blocks), "text": row_vals[0]})
                    body_start += 1
                else:
                    break

            body_end = len(grid_rows)
            while body_end > body_start:
                row_vals = [v for _, v in sorted(grid_rows[body_end - 1][1].items())]
                joined_tail = " ".join(row_vals)
                if re.match(r"^(?:\d+/|Note:|Source:|\*)", joined_tail):
                    body_end -= 1
                else:
                    break
            table_rows = grid_rows[body_start:body_end]
            if len(table_rows) < 2:
                continue

            parts = ["<table>"]
            labels: list[str] = []
            for rel_i, (r_idx, rmap) in enumerate(table_rows):
                cells_html: list[str] = []
                row_vals = [v for _, v in sorted(rmap.items())]
                is_hdr = rel_i == 0 or (
                    rel_i == 1 and any(re.fullmatch(r"(?:19|20)\d{2}(?:[-–]\d{2,4})?", x) or "total" in x.lower() for x in row_vals)
                )
                ctag = "th" if is_hdr else "td"
                for c_idx, val in sorted(rmap.items()):
                    if (r_idx, c_idx) in covered:
                        continue
                    rs, cs = spans.get((r_idx, c_idx), (1, 1))
                    attrs_s = ""
                    if cs > 1:
                        attrs_s += f' colspan="{cs}"'
                    if rs > 1:
                        attrs_s += f' rowspan="{rs}"'
                    cells_html.append(f"<{ctag}{attrs_s}>{_html.escape(val)}</{ctag}>")
                    if not is_hdr and len(val) >= 15 and not re.fullmatch(r"[\d,.\-]+", val) and len(labels) < 12:
                        labels.append(val[:24].lower())
                if cells_html:
                    parts.append("<tr>" + "".join(cells_html) + "</tr>")
            parts.append("</table>")
            table_html = "".join(parts)
            if "<tr>" not in table_html:
                continue
            out.append(
                {
                    "sheet_name": sname,
                    "table_id": (maj, min_),
                    "title": head_title,
                    "text_blocks": text_blocks,
                    "tables": [{"order": len(text_blocks), "html": table_html}],
                    "labels": labels,
                    "n_rows": len(table_rows),
                }
            )
    return out


def _http_get_bytes(url: str, *, timeout: int = 30, headers: Mapping[str, str] | None = None) -> bytes:
    req_hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        req_hdrs.update(dict(headers))
    req = urllib.request.Request(url, headers=req_hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _write_cached_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def _cache_fr_record(
    root: Path,
    rec: Mapping[str, Any],
    *,
    pick_table_page: bool = False,
) -> dict[str, Any] | None:
    """Download one FR PDF+XML pair, align gold to a representative page, and cache with SHA-256."""
    doc_no = str(rec["document_number"])
    xml_bytes = _http_get_bytes(str(rec["full_text_xml_url"]))
    gold_full = parse_govinfo_xml_gold(xml_bytes.decode("utf-8", errors="replace"))
    if pick_table_page and not (gold_full.get("tables") or []):
        return None

    pdf_bytes = _http_get_bytes(str(rec["pdf_url"]))
    pdf_rel = f"files/fr/{doc_no}.pdf"
    xml_rel = f"files/fr/{doc_no}.xml"
    pdf_sha = _write_cached_bytes(root / pdf_rel, pdf_bytes)
    xml_sha = _write_cached_bytes(root / xml_rel, xml_bytes)

    page_idx = 0
    n_pages, p0_text, _ = extract_born_digital_pdf_page_blocks(root / pdf_rel, page_index=0)
    gold = slice_gold_to_page_prefix(gold_full, p0_text, n_pdf_pages=n_pages)
    if pick_table_page and not (gold.get("tables") or []) and n_pages > 1:
        for cand_p in range(1, min(n_pages, 6)):
            _, cand_txt, _ = extract_born_digital_pdf_page_blocks(root / pdf_rel, page_index=cand_p)
            cand_gold = slice_gold_to_page_prefix(gold_full, cand_txt, n_pdf_pages=n_pages)
            if cand_gold.get("tables"):
                page_idx = cand_p
                gold = cand_gold
                break

    return {
        "id": f"fr:{doc_no}",
        "branch": "executive",
        "source": "govinfo_fr",
        "title": str(rec.get("title") or doc_no),
        "pdf_rel": pdf_rel,
        "pdf_sha256": pdf_sha,
        "gold_rel": xml_rel,
        "gold_sha256": xml_sha,
        "page_index": page_idx,
        **gold,
    }


def _select_fr_candidates(results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    short = [
        r
        for r in results
        if r.get("document_number")
        and r.get("pdf_url")
        and r.get("full_text_xml_url")
        and 1 <= int(r.get("page_length") or 99) <= 6
    ]
    if short:
        return short
    return [r for r in results if r.get("document_number") and r.get("pdf_url") and r.get("full_text_xml_url")]


def _stage_fr_items(root: Path, *, per_source: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_doc_nos: set[str] = set()

    if per_source >= 1:
        try:
            tbl_payload = json.loads(
                _http_get_bytes(_FR_API_URL + "&conditions[type][]=RULE&conditions[term]=table").decode("utf-8")
            )
            tbl_cands = _select_fr_candidates(list(tbl_payload.get("results") or []))
            probe_recs = stratified_sha256_sample(
                tbl_cands,
                dataset_key="us_gov_ocr:govinfo_fr:table",
                id_fn=lambda r: str(r["document_number"]),
                total_n=min(len(tbl_cands), max(5, per_source)),
            )
            for rec in probe_recs:
                cached = _cache_fr_record(root, rec, pick_table_page=True)
                if cached is not None:
                    items.append(cached)
                    seen_doc_nos.add(str(rec["document_number"]))
                    break
        except Exception:
            pass

    need = max(0, int(per_source) - len(items))
    if need > 0:
        payload = json.loads(_http_get_bytes(_FR_API_URL + "&conditions[type][]=RULE").decode("utf-8"))
        results = list(payload.get("results") or [])
        if len(results) < per_source * 2:
            extra = json.loads(_http_get_bytes(_FR_API_URL).decode("utf-8"))
            results.extend(extra.get("results") or [])
        cands = [r for r in _select_fr_candidates(results) if str(r["document_number"]) not in seen_doc_nos]
        picked = stratified_sha256_sample(
            cands,
            dataset_key="us_gov_ocr:govinfo_fr",
            id_fn=lambda r: str(r["document_number"]),
            total_n=need,
        )
        for rec in picked:
            cached = _cache_fr_record(root, rec, pick_table_page=False)
            if cached is not None:
                items.append(cached)
                seen_doc_nos.add(str(rec["document_number"]))
    return items


def _stage_govinfo_sitemap_items(
    root: Path,
    *,
    collection: str,
    branch: str,
    year: int,
    per_source: int,
) -> list[dict[str, Any]]:
    sm_url = f"https://www.govinfo.gov/sitemap/{collection}_{year}_sitemap.xml"
    sm_text = _http_get_bytes(sm_url).decode("utf-8", errors="replace")
    locs = re.findall(r"<loc>([^<]+)</loc>", sm_text)
    pkg_ids = [u.rstrip("/").rsplit("/", 1)[-1] for u in locs if "/app/details/" in u]
    picked = stratified_sha256_sample(
        pkg_ids,
        dataset_key=f"us_gov_ocr:{collection}:{year}",
        id_fn=lambda pid: pid,
        total_n=max(per_source * 3, per_source),
    )

    xml_subdir = "uslm" if collection == "PLAW" else "xml"
    items: list[dict[str, Any]] = []
    for pid in picked:
        if len(items) >= per_source:
            break
        pdf_url = f"https://www.govinfo.gov/content/pkg/{pid}/pdf/{pid}.pdf"
        xml_url = f"https://www.govinfo.gov/content/pkg/{pid}/{xml_subdir}/{pid}.xml"
        try:
            pdf_bytes = _http_get_bytes(pdf_url)
            xml_bytes = _http_get_bytes(xml_url)
        except Exception:
            continue
        if not pdf_bytes.startswith(b"%PDF") or xml_bytes.lstrip().startswith(b"<!DOCTYPE html"):
            continue
        pdf_rel = f"files/{collection.lower()}/{pid}.pdf"
        xml_rel = f"files/{collection.lower()}/{pid}.xml"
        pdf_sha = _write_cached_bytes(root / pdf_rel, pdf_bytes)
        xml_sha = _write_cached_bytes(root / xml_rel, xml_bytes)
        try:
            gold_full = parse_govinfo_xml_gold(xml_bytes.decode("utf-8", errors="replace"))
            n_pages, p0_text, _ = extract_born_digital_pdf_page_blocks(root / pdf_rel, page_index=0)
        except Exception:
            continue
        gold = slice_gold_to_page_prefix(gold_full, p0_text, n_pdf_pages=n_pages)
        if not gold["text_blocks"]:
            continue
        items.append(
            {
                "id": f"{collection.lower()}:{pid}",
                "branch": branch,
                "source": f"govinfo_{collection.lower()}",
                "title": pid,
                "pdf_rel": pdf_rel,
                "pdf_sha256": pdf_sha,
                "gold_rel": xml_rel,
                "gold_sha256": xml_sha,
                "page_index": 0,
                **gold,
            }
        )
    return items


def _stage_uscourts_items(root: Path, *, year: int, per_source: int) -> list[dict[str, Any]]:
    idx_text = _http_get_bytes("https://www.govinfo.gov/sitemap/USCOURTS_sitemap_index.xml").decode("utf-8", errors="replace")
    all_smaps = [u for u in re.findall(r"<loc>([^<]+)</loc>", idx_text) if f"_{year}_sitemap.xml" in u]
    if not all_smaps:
        all_smaps = [u for u in re.findall(r"<loc>([^<]+)</loc>", idx_text) if "_2024_sitemap.xml" in u]
    chosen_smaps = stratified_sha256_sample(
        all_smaps,
        dataset_key=f"us_gov_ocr:uscourts_smaps:{year}",
        id_fn=lambda u: u,
        total_n=min(4, len(all_smaps)),
    )

    pkg_ids: list[str] = []
    for sm_url in chosen_smaps:
        try:
            sm_txt = _http_get_bytes(sm_url).decode("utf-8", errors="replace")
            for u in re.findall(r"<loc>([^<]+)</loc>", sm_txt):
                if "/app/details/" in u:
                    pkg_ids.append(u.rstrip("/").rsplit("/", 1)[-1])
        except Exception:
            continue

    picked = stratified_sha256_sample(
        pkg_ids,
        dataset_key=f"us_gov_ocr:uscourts:{year}",
        id_fn=lambda pid: pid,
        total_n=max(per_source * 6, 8),
    )

    items: list[dict[str, Any]] = []
    for pid in picked:
        if len(items) >= per_source:
            break
        pdf_bytes = b""
        direct_url = f"https://www.govinfo.gov/content/pkg/{pid}/pdf/{pid}-0.pdf"
        try:
            pdf_bytes = _http_get_bytes(direct_url)
        except Exception:
            try:
                zbytes = _http_get_bytes(f"https://www.govinfo.gov/content/pkg/{pid}.zip")
                zf = zipfile.ZipFile(io.BytesIO(zbytes))
                pdf_members = sorted(n for n in zf.namelist() if n.endswith(".pdf"))
                if pdf_members:
                    pdf_bytes = zf.read(pdf_members[0])
            except Exception:
                continue
        if not pdf_bytes.startswith(b"%PDF"):
            continue
        pdf_rel = f"files/uscourts/{pid}.pdf"
        pdf_sha = _write_cached_bytes(root / pdf_rel, pdf_bytes)
        try:
            _, raw_p0, blocks = extract_born_digital_pdf_page_blocks(root / pdf_rel, page_index=0)
        except Exception:
            continue
        if len(raw_p0.strip()) < 150 or not blocks:
            continue
        items.append(
            {
                "id": f"uscourts:{pid}",
                "branch": "judicial",
                "source": "govinfo_uscourts",
                "title": pid,
                "pdf_rel": pdf_rel,
                "pdf_sha256": pdf_sha,
                "gold_rel": None,
                "gold_sha256": None,
                "page_index": 0,
                "text_blocks": blocks,
                "tables": [],
                "equations": [],
            }
        )
    return items


def _stage_uscode_items(root: Path, *, year: int, per_source: int) -> list[dict[str, Any]]:
    sm_year = min(int(year), 2024)
    sm_url = f"https://www.govinfo.gov/sitemap/USCODE_{sm_year}_sitemap.xml"
    sm_text = _http_get_bytes(sm_url).decode("utf-8", errors="replace")
    locs = re.findall(r"<loc>([^<]+)</loc>", sm_text)
    pkg_ids = [u.rstrip("/").rsplit("/", 1)[-1] for u in locs if "/app/details/" in u]
    picked_titles = stratified_sha256_sample(
        pkg_ids,
        dataset_key=f"us_gov_ocr:USCODE:{sm_year}",
        id_fn=lambda pid: pid,
        total_n=max(per_source * 5, 10),
    )

    items: list[dict[str, Any]] = []
    for pid in picked_titles:
        if len(items) >= per_source:
            break
        toc_url = f"https://www.govinfo.gov/content/pkg/{pid}/html/{pid}-toc.htm"
        try:
            toc_text = _http_get_bytes(toc_url).decode("utf-8", errors="replace")
        except Exception:
            continue
        chapters = re.findall(r'three-column-analysis-style-content-left">(\d+[a-z]?)\.?</div>', toc_text, re.I)
        if not chapters:
            chapters = ["1", "2"]
        chosen_ch = stratified_sha256_sample(
            list(dict.fromkeys(chapters)),
            dataset_key=f"us_gov_ocr:USCODE:{pid}",
            id_fn=lambda ch: ch,
            total_n=1,
        )
        for ch in chosen_ch:
            htm_url = f"https://www.govinfo.gov/content/pkg/{pid}/html/{pid}-chap{ch}.htm"
            pdf_url = f"https://www.govinfo.gov/content/pkg/{pid}/pdf/{pid}-chap{ch}.pdf"
            try:
                htm_bytes = _http_get_bytes(htm_url)
                pdf_bytes = _http_get_bytes(pdf_url)
            except Exception:
                continue
            if not pdf_bytes.startswith(b"%PDF") or b"<html" not in htm_bytes[:512].lower():
                continue
            pdf_rel = f"files/uscode/{pid}-chap{ch}.pdf"
            htm_rel = f"files/uscode/{pid}-chap{ch}.htm"
            pdf_sha = _write_cached_bytes(root / pdf_rel, pdf_bytes)
            htm_sha = _write_cached_bytes(root / htm_rel, htm_bytes)
            try:
                gold_full = parse_uscode_html_gold(htm_bytes.decode("utf-8", errors="replace"))
                n_pages, p0_text, _ = extract_born_digital_pdf_page_blocks(root / pdf_rel, page_index=0)
            except Exception:
                continue
            best_idx = 0
            best_gold = slice_gold_to_page_prefix(gold_full, p0_text, n_pdf_pages=n_pages)
            best_score = (100 if best_gold.get("tables") else 0) + len(best_gold.get("text_blocks") or [])
            for cand_p in range(1, min(n_pages, 5)):
                try:
                    _, p_txt, _ = extract_born_digital_pdf_page_blocks(root / pdf_rel, page_index=cand_p)
                except Exception:
                    continue
                cand_gold = slice_gold_to_page_prefix(gold_full, p_txt, n_pdf_pages=n_pages)
                score = (100 if cand_gold.get("tables") else 0) + len(cand_gold.get("text_blocks") or [])
                if score > best_score:
                    best_idx, best_gold, best_score = cand_p, cand_gold, score
            if not (best_gold.get("text_blocks") or best_gold.get("tables")):
                continue
            items.append(
                {
                    "id": f"uscode:{pid}-chap{ch}",
                    "branch": "legislative",
                    "source": "govinfo_uscode",
                    "title": f"{pid}-chap{ch}",
                    "pdf_rel": pdf_rel,
                    "pdf_sha256": pdf_sha,
                    "gold_rel": htm_rel,
                    "gold_sha256": htm_sha,
                    "page_index": best_idx,
                    **best_gold,
                }
            )
            if len(items) >= per_source:
                break
    return items


def _stage_budget_items(root: Path, *, per_source: int) -> list[dict[str, Any]]:
    import pypdfium2 as pdfium  # type: ignore[import-not-found]

    chosen_granules = stratified_sha256_sample(
        list(_BUDGET_CANDIDATE_GRANULES),
        dataset_key="us_gov_ocr:govinfo_budget",
        id_fn=lambda pair: pair[1],
        total_n=len(_BUDGET_CANDIDATE_GRANULES),
    )

    items: list[dict[str, Any]] = []
    for pkg, gran in chosen_granules:
        if len(items) >= per_source:
            break
        pdf_url = f"https://www.govinfo.gov/content/pkg/{pkg}/pdf/{gran}.pdf"
        xls_url = f"https://www.govinfo.gov/content/pkg/{pkg}/xls/{gran}.xlsx"
        try:
            pdf_bytes = _http_get_bytes(pdf_url)
            xls_bytes = _http_get_bytes(xls_url)
        except Exception:
            continue
        if not pdf_bytes.startswith(b"%PDF"):
            continue
        try:
            tbl_specs = parse_budget_xlsx_tables(xls_bytes)
        except Exception:
            continue
        if not tbl_specs:
            continue

        pdf_rel = f"files/budget/{gran}.pdf"
        xls_rel = f"files/budget/{gran}.xlsx"
        pdf_sha = _write_cached_bytes(root / pdf_rel, pdf_bytes)
        xls_sha = _write_cached_bytes(root / xls_rel, xls_bytes)

        doc = pdfium.PdfDocument(str(root / pdf_rel))
        try:
            page_texts: list[str] = []
            for p_i in range(len(doc)):
                tp = doc[p_i].get_textpage()
                page_texts.append(tp.get_text_bounded() or "")
                tp.close()
        finally:
            doc.close()

        for spec in tbl_specs:
            if len(items) >= per_source:
                break
            maj, min_ = spec["table_id"]
            head_pat = re.compile(rf"^\s*Table\s+(?:0?{maj})[–-]{min_}\.", re.M | re.I)
            labels = list(spec.get("labels") or [])
            matched_pi: int | None = None
            for p_i, pt in enumerate(page_texts):
                if not head_pat.search(pt) or "available online" in pt.lower():
                    continue
                head_lines = " ".join(ln.strip().lower() for ln in pt.splitlines()[:3])
                if "continued" in head_lines:
                    continue
                all_caps = re.findall(r"^\s*Table\s+\d+[–-]\d+\.", pt, re.M | re.I)
                if len(all_caps) != 1:
                    continue
                if p_i + 1 < len(page_texts) and re.search(rf"Table\s+(?:0?{maj})[–-]{min_}.*Continued", page_texts[p_i + 1], re.I):
                    continue
                norm_pt = re.sub(r"\s+", " ", pt).lower()
                if labels:
                    hits = sum(1 for lb in labels[:10] if lb in norm_pt)
                    if hits / min(len(labels), 10) < 0.7:
                        continue
                matched_pi = p_i
                break

            if matched_pi is None:
                continue
            items.append(
                {
                    "id": f"budget:{gran}_p{matched_pi}",
                    "branch": "executive",
                    "source": "govinfo_budget",
                    "title": spec["title"],
                    "pdf_rel": pdf_rel,
                    "pdf_sha256": pdf_sha,
                    "gold_rel": xls_rel,
                    "gold_sha256": xls_sha,
                    "page_index": matched_pi,
                    "text_blocks": list(spec["text_blocks"]),
                    "tables": list(spec["tables"]),
                    "equations": [],
                }
            )
    return items


def _stage_scotus_items(root: Path, *, per_source: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    candidates: list[dict[str, Any]] = []
    cache_dir = root / "files" / "scotus" / "parquets"
    for rel_pq, expected_sha in _SCOTUS_PARQUETS:
        if len(candidates) >= max(per_source * 2, per_source):
            break
        dest_pq = cache_dir / Path(rel_pq).name
        if not dest_pq.is_file() or compute_file_sha256(dest_pq) != expected_sha:
            url = f"https://huggingface.co/datasets/incrediblecrab/us-scotus-opinions/resolve/{_SCOTUS_REPO_REV}/{rel_pq}"
            data = _http_get_bytes(url, timeout=60)
            got_sha = hashlib.sha256(data).hexdigest()
            if got_sha != expected_sha:
                raise ValueError(f"Checksum mismatch for SCOTUS parquet {rel_pq}: expected {expected_sha}, got {got_sha}")
            _write_cached_bytes(dest_pq, data)
        tbl = pq.read_table(dest_pq, columns=["id", "docket", "title", "image_pages", "ocr_pages", "xcheck_equal", "media_type", "file", "file_sha256"])
        cols = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
        for i in range(tbl.num_rows):
            if cols["image_pages"][i] != 0 or cols["ocr_pages"][i] != 0 or not cols["xcheck_equal"][i]:
                continue
            if cols["media_type"][i] != "application/pdf" or not isinstance(cols["file"][i], (bytes, bytearray, memoryview)):
                continue
            pdf_bytes = bytes(cols["file"][i])
            if not pdf_bytes.startswith(b"%PDF"):
                continue
            if cols["file_sha256"][i] and hashlib.sha256(pdf_bytes).hexdigest() != cols["file_sha256"][i]:
                continue
            raw_id = str(cols["id"][i] or f"row_{i}")
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("_")
            candidates.append(
                {
                    "id": f"scotus:{slug}",
                    "slug": slug,
                    "title": str(cols["title"][i] or cols["docket"][i] or slug),
                    "pdf_bytes": pdf_bytes,
                }
            )

    picked = stratified_sha256_sample(
        candidates,
        dataset_key="us_gov_ocr:scotus_opinions",
        id_fn=lambda r: r["id"],
        total_n=per_source,
    )
    items: list[dict[str, Any]] = []
    for rec in picked:
        pdf_rel = f"files/scotus/{rec['slug']}.pdf"
        pdf_sha = _write_cached_bytes(root / pdf_rel, rec["pdf_bytes"])
        try:
            _, raw_p0, blocks = extract_born_digital_pdf_page_blocks(root / pdf_rel, page_index=0)
        except Exception:
            continue
        if len(raw_p0.strip()) < 150 or not blocks:
            continue
        items.append(
            {
                "id": rec["id"],
                "branch": "judicial",
                "source": "scotus_opinions",
                "title": rec["title"],
                "pdf_rel": pdf_rel,
                "pdf_sha256": pdf_sha,
                "gold_rel": None,
                "gold_sha256": None,
                "page_index": 0,
                "text_blocks": blocks,
                "tables": [],
                "equations": [],
            }
        )
    return items


def _stage_default_loc_btp_csv(root: Path) -> Path:
    csv_p = root / "files" / "loc_btp" / "btp_anthony.csv"
    if csv_p.is_file() and compute_file_sha256(csv_p) == _LOC_BTP_CSV_SHA256:
        return csv_p
    data = _http_get_bytes(_LOC_BTP_CSV_BLOB_URL, headers={"Accept": "application/vnd.github.raw"}, timeout=60)
    got = hashlib.sha256(data).hexdigest()
    if got != _LOC_BTP_CSV_SHA256:
        raise ValueError(f"Checksum mismatch for LoC btp-data CSV: expected {_LOC_BTP_CSV_SHA256}, got {got}")
    _write_cached_bytes(csv_p, data)
    return csv_p


def load_loc_btp_items(
    parquet_or_csv_path: str | Path,
    *,
    max_items: int | None = None,
    images_dir: str | Path | None = None,
    download_images: bool = False,
) -> list[dict[str, Any]]:
    """Load LoC ``By the People`` volunteer-transcribed archival pages."""
    import pyarrow.parquet as pq

    src = Path(parquet_or_csv_path)
    if not src.is_file():
        raise FileNotFoundError(f"LoC By the People dataset file not found: {src}")

    rows: list[dict[str, Any]] = []
    if src.suffix.lower() == ".parquet":
        tbl = pq.read_table(src)
        cols = tbl.column_names
        py_cols = [tbl.column(c).to_pylist() for c in cols]
        rows = [dict(zip(cols, vals)) for vals in zip(*py_cols)]
    else:
        import csv

        with src.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))

    candidates: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        status = str(r.get("AssetStatus") or "").strip().lower()
        if status and status != "completed":
            continue
        text = _clean_ws(str(r.get("Transcription") or r.get("FullText") or ""))
        img_url = str(r.get("DownloadUrl") or r.get("DownloadURL") or r.get("TiffUrl") or "").strip()
        if img_url.startswith("http://tile.loc.gov/"):
            img_url = "https://" + img_url[len("http://") :]
        if len(text) < 40 or not img_url:
            continue
        camp = str(r.get("Campaign") or "loc_btp").strip()
        aid = str(r.get("AssetId") or r.get("id") or f"row_{idx}")
        candidates.append(
            {
                "id": f"loc_btp:{aid}",
                "branch": "executive",
                "source": "loc_btp",
                "title": camp,
                "campaign": camp,
                "image_url": img_url,
                "image_path": None,
                "text_blocks": [{"category_type": "text_block", "order": 0, "text": text}],
                "tables": [],
                "equations": [],
            }
        )

    selected = stratified_sha256_sample(
        candidates,
        dataset_key="us_gov_ocr:loc_btp",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["campaign"],
        total_n=max_items,
    )
    if download_images and images_dir is not None:
        img_root = Path(images_dir)
        for item in selected:
            ext = ".tif" if ".tif" in item["image_url"].lower() else ".jpg"
            dest = img_root / f"{hashlib.sha256(item['id'].encode('utf-8')).hexdigest()[:16]}{ext}"
            if not dest.is_file():
                data = _http_get_bytes(item["image_url"])
                _write_cached_bytes(dest, data)
            item["image_path"] = str(dest)
    return selected


_CANON_GOV_SOURCE: dict[str, str] = {
    "plaw": "govinfo_plaw",
    "govinfo_plaw": "govinfo_plaw",
    "bills": "govinfo_bills",
    "govinfo_bills": "govinfo_bills",
    "uscode": "govinfo_uscode",
    "govinfo_uscode": "govinfo_uscode",
    "fr": "govinfo_fr",
    "govinfo_fr": "govinfo_fr",
    "budget": "govinfo_budget",
    "govinfo_budget": "govinfo_budget",
    "loc_btp": "loc_btp",
    "uscourts": "govinfo_uscourts",
    "govinfo_uscourts": "govinfo_uscourts",
    "scotus": "scotus_opinions",
    "scotus_opinions": "scotus_opinions",
}


def stage_us_gov_ocr(
    *,
    branches: Sequence[str] = ("legislative", "executive", "judicial"),
    per_source: int = 3,
    year: int = 2025,
    cache_dir: str | Path | None = None,
    loc_btp_path: str | Path | None = None,
    auto_loc_btp: bool = False,
    sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Download and verify paired Public-Domain U.S. Gov PDF<->XML/HTML/XLSX/scan items into ``cache_dir/us_gov_ocr``."""
    eff_branches = tuple(str(b).strip().lower() for b in branches if str(b).strip())
    for b in eff_branches:
        if b not in ("legislative", "executive", "judicial"):
            raise ValueError(f"Unsupported U.S. Gov branch {b!r}")

    eff_sources: set[str] | None = None
    if sources is not None:
        eff_sources = set()
        for raw_s in sources:
            tok = str(raw_s).strip().lower()
            if not tok or tok == "all":
                eff_sources = None
                break
            if tok not in _CANON_GOV_SOURCE:
                raise ValueError(f"Unsupported U.S. Gov source {raw_s!r}")
            eff_sources.add(_CANON_GOV_SOURCE[tok])

    def _want(canon: str) -> bool:
        return eff_sources is None or canon in eff_sources

    root = resolve_cache_dir(cache_dir) / "us_gov_ocr"
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    if "legislative" in eff_branches:
        if _want("govinfo_plaw"):
            items.extend(_stage_govinfo_sitemap_items(root, collection="PLAW", branch="legislative", year=min(year, 2024), per_source=per_source))
        if _want("govinfo_bills"):
            items.extend(_stage_govinfo_sitemap_items(root, collection="BILLS", branch="legislative", year=year, per_source=per_source))
        if _want("govinfo_uscode"):
            items.extend(_stage_uscode_items(root, year=year, per_source=per_source))
    if "executive" in eff_branches:
        if _want("govinfo_fr"):
            items.extend(_stage_fr_items(root, per_source=per_source))
        if _want("govinfo_budget"):
            items.extend(_stage_budget_items(root, per_source=per_source))
        want_btp = _want("loc_btp") and (loc_btp_path is not None or auto_loc_btp or (eff_sources is not None and "loc_btp" in eff_sources))
        if want_btp:
            btp_src = Path(loc_btp_path) if loc_btp_path is not None else _stage_default_loc_btp_csv(root)
            for it in load_loc_btp_items(
                btp_src,
                max_items=per_source,
                images_dir=root / "files" / "loc_btp",
                download_images=True,
            ):
                raw_img = it.pop("image_path", None)
                if not raw_img:
                    continue
                img_p = Path(str(raw_img))
                if not img_p.is_file():
                    continue
                it["image_rel"] = str(img_p.relative_to(root))
                it["image_sha256"] = compute_file_sha256(img_p)
                items.append(it)
    if "judicial" in eff_branches:
        if _want("govinfo_uscourts"):
            items.extend(_stage_uscourts_items(root, year=year, per_source=per_source))
        if _want("scotus_opinions"):
            items.extend(_stage_scotus_items(root, per_source=per_source))

    manifest_p = root / "manifest.json"
    existing_items: list[dict[str, Any]] = []
    if manifest_p.is_file():
        try:
            old_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Corrupt U.S. Gov OCR manifest at {manifest_p}: {exc}") from exc
        if not isinstance(old_manifest, dict):
            raise ValueError(f"Corrupt U.S. Gov OCR manifest at {manifest_p}: expected JSON object")
        raw_old_items = old_manifest.get("items") or []
        if not isinstance(raw_old_items, list):
            raise ValueError(f"Corrupt U.S. Gov OCR manifest at {manifest_p}: expected 'items' list")
        existing_items = [dict(it) for it in raw_old_items if isinstance(it, dict)]

    targeted_sources: set[str] = {str(it.get("source") or "unknown") for it in items}
    if "legislative" in eff_branches:
        for src in ("govinfo_plaw", "govinfo_bills", "govinfo_uscode"):
            if _want(src):
                targeted_sources.add(src)
    if "executive" in eff_branches:
        for src in ("govinfo_fr", "govinfo_budget"):
            if _want(src):
                targeted_sources.add(src)
        if want_btp:
            targeted_sources.add("loc_btp")
    if "judicial" in eff_branches:
        for src in ("govinfo_uscourts", "scotus_opinions"):
            if _want(src):
                targeted_sources.add(src)

    new_ids = {(str(it.get("source") or "unknown"), str(it.get("id") or "")) for it in items if it.get("id")}
    merged_items: list[dict[str, Any]] = []
    for old_it in existing_items:
        old_src = str(old_it.get("source") or "unknown")
        old_id = str(old_it.get("id") or "")
        if old_src in targeted_sources or (old_id and (old_src, old_id) in new_ids):
            continue
        merged_items.append(old_it)
    merged_items.extend(items)

    manifest = {"schema_version": 1, "year": int(year), "items": merged_items}
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    manifest_sha = _write_cached_bytes(manifest_p, manifest_bytes)
    sources_out = sorted({str(it.get("source") or "unknown") for it in merged_items})
    all_branches = tuple(dict.fromkeys([*eff_branches, *(str(it.get("branch") or "") for it in merged_items if it.get("branch"))]))
    return {
        "cache_dir": str(root),
        "manifest_path": str(manifest_p),
        "manifest_sha256": manifest_sha,
        "n_items": len(merged_items),
        "by_branch": {b: sum(1 for it in merged_items if it.get("branch") == b) for b in all_branches},
        "by_source": {s: sum(1 for it in merged_items if it.get("source") == s) for s in sources_out},
    }


def load_us_gov_ocr(
    *,
    branches: Sequence[str] | None = None,
    max_items: int | None = None,
    cache_dir: str | Path | None = None,
    verify_checksums: bool = True,
) -> list[dict[str, Any]]:
    """Load cached, SHA-256 verified paired U.S. Gov PDF<->XML items from ``cache_dir/us_gov_ocr``."""
    root = resolve_cache_dir(cache_dir) / "us_gov_ocr"
    manifest_p = root / "manifest.json"
    if not manifest_p.is_file():
        raise FileNotFoundError(
            f"Paired U.S. Gov OCR manifest not found at {manifest_p}; run `scripts/benchmark_outcomes.py stage-gov` first."
        )
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    allowed_branches = {str(b).strip().lower() for b in branches} if branches else None

    candidates: list[dict[str, Any]] = []
    for raw in manifest.get("items") or []:
        br = str(raw.get("branch") or "").lower()
        if allowed_branches is not None and br not in allowed_branches:
            continue
        pdf_rel = raw.get("pdf_rel")
        img_rel = raw.get("image_rel")
        pdf_path = (root / str(pdf_rel)) if pdf_rel else None
        img_path = (root / str(img_rel)) if img_rel else None
        if verify_checksums and pdf_path is not None:
            if not pdf_path.is_file():
                raise FileNotFoundError(f"Missing cached U.S. Gov PDF: {pdf_path}")
            exp = raw.get("pdf_sha256")
            if exp and compute_file_sha256(pdf_path) != exp:
                raise ValueError(f"Checksum mismatch for {pdf_path}")
        if verify_checksums and img_path is not None:
            if not img_path.is_file():
                raise FileNotFoundError(f"Missing cached U.S. Gov image: {img_path}")
            exp_img = raw.get("image_sha256")
            if exp_img and compute_file_sha256(img_path) != exp_img:
                raise ValueError(f"Checksum mismatch for {img_path}")
        if verify_checksums and raw.get("gold_rel") and raw.get("gold_sha256"):
            gold_path = root / str(raw["gold_rel"])
            if not gold_path.is_file():
                raise FileNotFoundError(f"Missing cached U.S. Gov XML: {gold_path}")
            if compute_file_sha256(gold_path) != raw["gold_sha256"]:
                raise ValueError(f"Checksum mismatch for {gold_path}")
        candidates.append(
            {
                **raw,
                "pdf_path": str(pdf_path) if pdf_path is not None else None,
                "image_path": str(img_path) if img_path is not None else None,
            }
        )

    return stratified_sha256_sample(
        candidates,
        dataset_key="us_gov_ocr:manifest",
        id_fn=lambda r: str(r["id"]),
        stratum_fn=lambda r: f"{r.get('branch') or 'unknown'}:{r.get('source') or 'unknown'}",
        total_n=max_items,
    )


__all__ = [
    "extract_born_digital_pdf_page_blocks",
    "gpotable_to_html",
    "load_loc_btp_items",
    "load_us_gov_ocr",
    "parse_budget_xlsx_tables",
    "parse_govinfo_xml_gold",
    "parse_uscode_html_gold",
    "slice_gold_to_page_prefix",
    "stage_us_gov_ocr",
]
