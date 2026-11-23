"""Category 5 deterministic pure-Python OCR scorers: olmOCR-Bench, TEDS/TEDS-S, GriTS, OmniDocBench, CORD-v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import difflib
from html.parser import HTMLParser
import json
import re
from typing import Any, Mapping, Sequence
import unicodedata

import numpy as np

from .chat_reasoning import repeat_4gram_rate
from .structured import (
    _strip_json_fences,
    evaluate_structured_extraction,
    linear_sum_assignment_max,
)

# ---------------------------------------------------------------------------
# Exact Bit-Parallel String Edit Distance, Fuzzy Substring & ANLS Primitives
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^[ ]{0,3}(```+|~~~+)(.*)$")
_HTML_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_TABLE_TAG_RE = re.compile(r"<(/?)(table|tr|td|th)\b[^>]*>", re.IGNORECASE)
_GFM_SEP_CELL_RE = re.compile(r"^:?-{1,}:?$")
_IMG_TAG_RE = re.compile(r"<img\b[^>]*?/?>", re.IGNORECASE)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)|\\\((.+?)\\\)", re.DOTALL)
_NUMERIC_TOKEN_RE = re.compile(
    r"(?:\(\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?\)|[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)"
)
_MD_ATX_PREFIX_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_MD_EMPH_RE = re.compile(r"(?<!\\)(?:\*\*|__|~~|\*|_)")
_UNICODE_PUNCT_TRANS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
    }
)


def levenshtein_distance(a: str, b: str) -> int:
    """Exact character-level Levenshtein distance via Myers' bit-parallel algorithm (pure Python)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Keep bitmask width = min(len(a), len(b)) for tight bigint operations.
    if len(a) > len(b):
        a, b = b, a
    m = len(a)
    peq: dict[str, int] = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    pv = mask
    mv = 0
    score = m
    high_bit = 1 << (m - 1)
    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high_bit:
            score += 1
        elif mh & high_bit:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return int(score)


def normalized_edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein distance in ``[0.0, 1.0]`` (``0.0`` when both strings are empty)."""
    if not a and not b:
        return 0.0
    denom = max(len(a), len(b))
    return float(levenshtein_distance(a, b)) / float(denom)


def normalized_indel_similarity(a: str, b: str) -> float:
    """Normalized Indel Similarity ``1 - NID`` in ``[0.0, 1.0]``."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())


def normalize_ocr_text(text: str, *, case_sensitive: bool = True, strip_markdown: bool = False) -> str:
    """Normalize Unicode punctuation, whitespace, and optional Markdown emphasis for OCR matching."""
    s = unicodedata.normalize("NFKC", text or "").translate(_UNICODE_PUNCT_TRANS)
    if strip_markdown:
        s = _MD_EMPH_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    if not case_sensitive:
        s = s.lower()
    return s


def anls_score(pred: str, gold: str, *, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (ICDAR 2019 ANLS) for a single string pair."""
    pn = normalize_ocr_text(pred, case_sensitive=False)
    gn = normalize_ocr_text(gold, case_sensitive=False)
    if not gn and not pn:
        return 1.0
    if not gn or not pn:
        return 0.0
    ned = normalized_edit_distance(pn, gn)
    return (1.0 - ned) if ned < float(threshold) else 0.0


def fuzzy_find_substring(pattern: str, text: str, *, max_diffs: int = 0) -> tuple[bool, int, int]:
    """Exact semi-global Levenshtein substring search via Myers' bit-parallel algorithm.

    Returns ``(matched, best_dist, end_index_exclusive)``. ``end_index_exclusive`` is the earliest
    text end index where edit distance ``<= max_diffs`` is achieved (or overall minimum-distance end).
    """
    if not pattern:
        return False, 0, -1
    if not text:
        return False, len(pattern), -1
    max_d = max(0, int(max_diffs))
    if max_d == 0:
        idx = text.find(pattern)
        if idx >= 0:
            return True, 0, idx + len(pattern)
        return False, len(pattern), -1

    # Fast path: exact substring hit immediately wins with dist=0.
    exact_idx = text.find(pattern)
    if exact_idx >= 0:
        return True, 0, exact_idx + len(pattern)

    m = len(pattern)
    peq: dict[str, int] = {}
    for i, ch in enumerate(pattern):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    pv = mask
    mv = 0
    score = m
    high_bit = 1 << (m - 1)
    best_dist = m
    best_end = -1
    first_match_end = -1
    first_match_dist = m

    for j, ch in enumerate(text):
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high_bit:
            score += 1
        elif mh & high_bit:
            score -= 1
        # Semi-global shift (zero cost on row 0: omit `| 1` on ph).
        ph = (ph << 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
        if score < best_dist:
            best_dist = score
            best_end = j + 1
        if score <= max_d and first_match_end < 0:
            first_match_end = j + 1
            first_match_dist = score
            if score == 0:
                break

    if first_match_end >= 0:
        return True, int(first_match_dist), int(first_match_end)
    return False, int(best_dist), int(best_end)


# ---------------------------------------------------------------------------
# Pure-Python Markdown / HTML Structural Validation
# ---------------------------------------------------------------------------


def _split_gfm_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        n_bs = 0
        idx = len(s) - 2
        while idx >= 0 and s[idx] == "\\":
            n_bs += 1
            idx -= 1
        if (n_bs % 2) == 0:
            s = s[:-1]
    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in s:
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == "\\":
            buf.append(ch)
            escaped = True
        elif ch == "|":
            cells.append("".join(buf).strip())
            buf.clear()
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


def _is_gfm_sep_row(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_gfm_row(line)
    return bool(cells) and all(bool(_GFM_SEP_CELL_RE.match(c.replace(" ", ""))) for c in cells)


def validate_ocr_markdown(text: str, *, allow_html_tables: bool = True) -> dict[str, Any]:
    """Deterministic structural validator for OCR Markdown/HTML output (zero external deps)."""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    errors: list[str] = []

    fence_char: str | None = None
    fence_len = 0
    in_display_math = False
    gfm_tables = 0
    gfm_rect = True

    idx = 0
    n = len(lines)
    while idx < n:
        ln = lines[idx]
        m_f = _FENCE_RE.match(ln)
        if m_f:
            tok = m_f.group(1)
            if fence_char is None:
                fence_char, fence_len = tok[0], len(tok)
            elif tok[0] == fence_char and len(tok) >= fence_len and not m_f.group(2).strip():
                fence_char, fence_len = None, 0
            idx += 1
            continue
        if fence_char is not None:
            idx += 1
            continue

        stripped = ln.strip()
        if stripped == "$$":
            in_display_math = not in_display_math
            idx += 1
            continue
        if "$$" in stripped:
            count_dd = stripped.count("$$")
            if count_dd % 2 == 1:
                in_display_math = not in_display_math

        if idx + 1 < n and "|" in ln and _is_gfm_sep_row(lines[idx + 1]):
            hdr_cells = _split_gfm_row(ln)
            sep_cells = _split_gfm_row(lines[idx + 1])
            exp_cols = len(sep_cells)
            gfm_tables += 1
            if len(hdr_cells) != exp_cols:
                gfm_rect = False
                errors.append(f"GFM header col mismatch ({len(hdr_cells)} vs {exp_cols})")
            j = idx + 2
            while j < n and "|" in lines[j] and lines[j].strip() and not _FENCE_RE.match(lines[j]):
                row_cells = _split_gfm_row(lines[j])
                if len(row_cells) != exp_cols:
                    gfm_rect = False
                    errors.append(f"GFM row col mismatch ({len(row_cells)} vs {exp_cols})")
                j += 1
            idx = j
            continue
        idx += 1

    fence_balanced = fence_char is None
    math_balanced = not in_display_math
    if not fence_balanced:
        errors.append("Unclosed fenced code block")
    if not math_balanced:
        errors.append("Unbalanced $$ display math delimiter")

    html_table_blocks = _HTML_TABLE_BLOCK_RE.findall(s)
    html_tables = len(html_table_blocks)
    html_balanced = True
    open_t = len(re.findall(r"<table\b", s, re.IGNORECASE))
    close_t = len(re.findall(r"</table>", s, re.IGNORECASE))
    if open_t != close_t:
        html_balanced = False
        errors.append(f"Unbalanced <table> tags ({open_t} open vs {close_t} close)")
    if html_tables > 0 and not allow_html_tables:
        errors.append(f"Unexpected raw HTML tables ({html_tables}) in GFM-only mode")

    valid = bool(s.strip()) and fence_balanced and math_balanced and html_balanced and gfm_rect and (allow_html_tables or html_tables == 0)
    return {
        "valid": valid,
        "table_count": gfm_tables + html_tables,
        "gfm_table_count": gfm_tables,
        "html_table_count": html_tables,
        "fence_balanced": fence_balanced,
        "math_balanced": math_balanced,
        "html_tables_balanced": html_balanced,
        "gfm_tables_rectangular": gfm_rect,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Table Representation, HTML/GFM Parsing, Exact TEDS / TEDS-S, GriTS & Numeric F1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableCell:
    """Single logical cell inside a parsed HTML or GFM table."""

    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


@dataclass(frozen=True)
class ParsedTable:
    """Canonical 2D table grid + row-grouped logical cells."""

    n_rows: int
    n_cols: int
    rows: tuple[tuple[TableCell, ...], ...]
    grid: tuple[tuple[str, ...], ...]
    span_grid: tuple[tuple[tuple[int, int, int, int], ...], ...]


class _CanonicalHtmlTableParser(HTMLParser):
    """Parse HTML ``<table>`` into logical row cells and exact 2D occupancy grids."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.row_cells: list[list[TableCell]] = []
        self._in_tr = False
        self._in_cell = False
        self._cell_is_th = False
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._cell_buf: list[str] = []
        self._occupied: dict[tuple[int, int], tuple[str, tuple[int, int, int, int]]] = {}
        self._cur_row = -1
        self._cur_col = 0
        self._table_depth = 0

    @staticmethod
    def _parse_span(raw: str | None) -> int:
        if not raw:
            return 1
        try:
            return max(1, min(64, int(str(raw).strip())))
        except ValueError:
            return 1

    def _flush_cell(self) -> None:
        if not self._in_cell:
            return
        txt = " ".join("".join(self._cell_buf).split())
        if self._cur_row < 0:
            self._cur_row = 0
            self._cur_col = 0
            self.row_cells.append([])
        r = self._cur_row
        while (r, self._cur_col) in self._occupied:
            self._cur_col += 1
        c = self._cur_col
        rs = self._cell_rowspan
        cs = self._cell_colspan
        cell = TableCell(text=txt, row=r, col=c, rowspan=rs, colspan=cs, is_header=self._cell_is_th)
        self.row_cells[r].append(cell)
        for dr in range(rs):
            for dc in range(cs):
                # Relative box coordinates `(dr, dc, rs - dr, cs - dc)` encode exact 2D span topology for GriTS.
                self._occupied[(r + dr, c + dc)] = (txt, (-dr, -dc, rs - dr, cs - dc))
        self._cur_col = c + cs
        self._in_cell = False
        self._cell_buf = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low == "table":
            self._table_depth += 1
        elif self._table_depth <= 1:
            if low == "tr":
                self._flush_cell()
                self._cur_row += 1
                self._cur_col = 0
                self.row_cells.append([])
                self._in_tr = True
            elif low in ("td", "th"):
                self._flush_cell()
                amap = {k.lower(): v for k, v in attrs}
                self._in_cell = True
                self._cell_is_th = low == "th"
                self._cell_colspan = self._parse_span(amap.get("colspan"))
                self._cell_rowspan = self._parse_span(amap.get("rowspan"))
                self._cell_buf = []
            elif low == "br" and self._in_cell:
                self._cell_buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low == "table":
            self._flush_cell()
            self._table_depth = max(0, self._table_depth - 1)
        elif self._table_depth <= 1:
            if low in ("td", "th"):
                self._flush_cell()
            elif low == "tr":
                self._flush_cell()
                self._in_tr = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)

    def build(self) -> ParsedTable:
        self._flush_cell()
        non_empty_rows = [r for r in self.row_cells if r]
        if not self._occupied:
            return ParsedTable(0, 0, (), (), ())
        n_rows = max(r for r, _ in self._occupied) + 1
        n_cols = max(c for _, c in self._occupied) + 1
        grid: list[tuple[str, ...]] = []
        span_grid: list[tuple[tuple[int, int, int, int], ...]] = []
        for r in range(n_rows):
            g_row: list[str] = []
            s_row: list[tuple[int, int, int, int]] = []
            for c in range(n_cols):
                entry = self._occupied.get((r, c))
                if entry is None:
                    g_row.append("")
                    s_row.append((0, 0, 1, 1))
                else:
                    g_row.append(entry[0])
                    s_row.append(entry[1])
            grid.append(tuple(g_row))
            span_grid.append(tuple(s_row))
        return ParsedTable(
            n_rows=n_rows,
            n_cols=n_cols,
            rows=tuple(tuple(r) for r in non_empty_rows),
            grid=tuple(grid),
            span_grid=tuple(span_grid),
        )


def parse_html_table(html_str: str) -> ParsedTable:
    """Parse an HTML ``<table>`` string into canonical logical rows and 2D occupancy grids."""
    s = (html_str or "").strip()
    if not s:
        return ParsedTable(0, 0, (), (), ())
    if "<table" not in s.lower() and ("<tr" in s.lower() or "<td" in s.lower() or "<th" in s.lower()):
        s = f"<table>{s}</table>"
    parser = _CanonicalHtmlTableParser()
    parser.feed(s)
    parser.close()
    return parser.build()


def parse_gfm_table_lines(lines: Sequence[str]) -> ParsedTable:
    """Parse GFM pipe-table lines (header + separator + body) into a ``ParsedTable``."""
    if len(lines) < 2 or not _is_gfm_sep_row(lines[1]):
        return ParsedTable(0, 0, (), (), ())
    raw_rows = [_split_gfm_row(lines[0])]
    for ln in lines[2:]:
        if "|" in ln and ln.strip():
            raw_rows.append(_split_gfm_row(ln))
    n_cols = max((len(r) for r in raw_rows), default=0)
    if n_cols == 0:
        return ParsedTable(0, 0, (), (), ())

    logical_rows: list[tuple[TableCell, ...]] = []
    grid: list[tuple[str, ...]] = []
    span_grid: list[tuple[tuple[int, int, int, int], ...]] = []
    for r_idx, row in enumerate(raw_rows):
        padded = [re.sub(r"\\\|", "|", c).strip() for c in row] + [""] * (n_cols - len(row))
        logical_rows.append(
            tuple(
                TableCell(text=" ".join(val.split()), row=r_idx, col=c_idx, rowspan=1, colspan=1, is_header=(r_idx == 0))
                for c_idx, val in enumerate(padded)
            )
        )
        grid.append(tuple(" ".join(val.split()) for val in padded))
        span_grid.append(tuple((0, 0, 1, 1) for _ in range(n_cols)))
    return ParsedTable(
        n_rows=len(raw_rows),
        n_cols=n_cols,
        rows=tuple(logical_rows),
        grid=tuple(grid),
        span_grid=tuple(span_grid),
    )


def extract_tables_from_markdown(text: str) -> list[ParsedTable]:
    """Extract all HTML ``<table>`` and GFM pipe tables from OCR Markdown output in document order."""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    found: list[tuple[int, ParsedTable]] = []

    for m in _HTML_TABLE_BLOCK_RE.finditer(s):
        pt = parse_html_table(m.group(0))
        if pt.n_rows > 0 and pt.n_cols > 0:
            found.append((m.start(), pt))

    # Mask HTML table spans before scanning GFM pipe tables so embedded pipes don't double-count.
    masked = _HTML_TABLE_BLOCK_RE.sub(lambda m: "\n" * m.group(0).count("\n"), s)
    lines = masked.split("\n")
    line_offsets: list[int] = []
    pos = 0
    for ln in lines:
        line_offsets.append(pos)
        pos += len(ln) + 1

    i = 0
    n = len(lines)
    fence_char: str | None = None
    fence_len = 0
    while i < n:
        ln = lines[i]
        mf = _FENCE_RE.match(ln)
        if mf:
            tok = mf.group(1)
            if fence_char is None:
                fence_char, fence_len = tok[0], len(tok)
            elif tok[0] == fence_char and len(tok) >= fence_len and not mf.group(2).strip():
                fence_char, fence_len = None, 0
            i += 1
            continue
        if fence_char is not None:
            i += 1
            continue
        if i + 1 < n and "|" in ln and _is_gfm_sep_row(lines[i + 1]):
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip() and not _FENCE_RE.match(lines[j]):
                j += 1
            pt = parse_gfm_table_lines(lines[i:j])
            if pt.n_rows > 0 and pt.n_cols > 0:
                found.append((line_offsets[i], pt))
            i = j
            continue
        i += 1

    found.sort(key=lambda pair: pair[0])
    return [pt for _, pt in found]


@dataclass(frozen=True)
class _DomNode:
    tag: str
    colspan: int
    rowspan: int
    text: str
    children: tuple[_DomNode, ...]


def _table_to_dom(table: ParsedTable) -> _DomNode:
    tr_nodes: list[_DomNode] = []
    for row in table.rows:
        td_nodes = tuple(
            _DomNode(
                tag="td",
                colspan=cell.colspan,
                rowspan=cell.rowspan,
                text=normalize_ocr_text(cell.text, case_sensitive=True),
                children=(),
            )
            for cell in row
        )
        tr_nodes.append(_DomNode(tag="tr", colspan=1, rowspan=1, text="", children=td_nodes))
    return _DomNode(tag="table", colspan=1, rowspan=1, text="", children=tuple(tr_nodes))


def _dom_rename_cost(a: _DomNode, b: _DomNode, *, structure_only: bool) -> float:
    if a.tag != b.tag:
        return 1.0
    if a.tag == "td":
        if a.colspan != b.colspan or a.rowspan != b.rowspan:
            return 1.0
        if structure_only:
            return 0.0
        return normalized_edit_distance(a.text, b.text)
    return 0.0


def _postorder_index(root: _DomNode) -> tuple[list[_DomNode], list[int], list[int]]:
    """Build 1-indexed postorder node array, leftmost-leaf indices, and Zhang-Shasha keyroots."""
    nodes: list[_DomNode] = []
    lmost: list[int] = []

    def _dfs(node: _DomNode) -> int:
        first_leaf: int | None = None
        for ch in node.children:
            ch_lm = _dfs(ch)
            if first_leaf is None:
                first_leaf = ch_lm
        idx_1 = len(nodes) + 1
        nodes.append(node)
        lm = idx_1 if first_leaf is None else first_leaf
        lmost.append(lm)
        return lm

    _dfs(root)
    last_by_lm: dict[int, int] = {}
    for idx_1, lm in enumerate(lmost, 1):
        last_by_lm[lm] = idx_1
    keyroots = sorted(last_by_lm.values())
    return nodes, lmost, keyroots


def _zhang_shasha_ted(root_a: _DomNode, root_b: _DomNode, *, structure_only: bool) -> tuple[float, int]:
    """Exact Zhang-Shasha (1989) ordered tree edit distance; returns ``(dist, max_nodes)``."""
    na, la, kra = _postorder_index(root_a)
    nb, lb, krb = _postorder_index(root_b)
    n1, n2 = len(na), len(nb)
    max_n = max(n1, n2)
    if n1 == 0 or n2 == 0:
        return float(max_n), max_n

    # Fast exact row-level DP fallback for massive tables (>300 DOM nodes) to bound worst-case latency.
    if n1 > 300 or n2 > 300:
        rows_a = root_a.children
        rows_b = root_b.children
        ra_n, rb_n = len(rows_a), len(rows_b)
        dp_r = [[0.0] * (rb_n + 1) for _ in range(ra_n + 1)]
        for i in range(1, ra_n + 1):
            dp_r[i][0] = dp_r[i - 1][0] + 1.0 + float(len(rows_a[i - 1].children))
        for j in range(1, rb_n + 1):
            dp_r[0][j] = dp_r[0][j - 1] + 1.0 + float(len(rows_b[j - 1].children))
        for i in range(1, ra_n + 1):
            ca = rows_a[i - 1].children
            for j in range(1, rb_n + 1):
                cb = rows_b[j - 1].children
                m_c, k_c = len(ca), len(cb)
                prev = [float(x) for x in range(k_c + 1)]
                for ci in range(1, m_c + 1):
                    cur = [float(ci)] + [0.0] * k_c
                    a_cell = ca[ci - 1]
                    for cj in range(1, k_c + 1):
                        rc = _dom_rename_cost(a_cell, cb[cj - 1], structure_only=structure_only)
                        cur[cj] = min(prev[cj] + 1.0, cur[cj - 1] + 1.0, prev[cj - 1] + rc)
                    prev = cur
                sub_cost = prev[k_c]
                dp_r[i][j] = min(
                    dp_r[i - 1][j] + 1.0 + float(m_c),
                    dp_r[i][j - 1] + 1.0 + float(k_c),
                    dp_r[i - 1][j - 1] + sub_cost,
                )
        return float(dp_r[ra_n][rb_n]), max_n

    td = np.zeros((n1 + 1, n2 + 1), dtype=np.float64)
    for x in kra:
        lx = la[x - 1]
        for y in krb:
            ly = lb[y - 1]
            m = x - lx + 1
            k = y - ly + 1
            fd = np.zeros((m + 1, k + 1), dtype=np.float64)
            for i in range(1, m + 1):
                fd[i, 0] = fd[i - 1, 0] + 1.0
            for j in range(1, k + 1):
                fd[0, j] = fd[0, j - 1] + 1.0
            for i in range(1, m + 1):
                ni = lx + i - 1
                l_ni = la[ni - 1]
                node_a = na[ni - 1]
                for j in range(1, k + 1):
                    nj = ly + j - 1
                    l_nj = lb[nj - 1]
                    if l_ni == lx and l_nj == ly:
                        rc = _dom_rename_cost(node_a, nb[nj - 1], structure_only=structure_only)
                        val = min(fd[i - 1, j] + 1.0, fd[i, j - 1] + 1.0, fd[i - 1, j - 1] + rc)
                        fd[i, j] = val
                        td[ni, nj] = val
                    else:
                        pi = l_ni - lx
                        pj = l_nj - ly
                        fd[i, j] = min(
                            fd[i - 1, j] + 1.0,
                            fd[i, j - 1] + 1.0,
                            fd[pi, pj] + td[ni, nj],
                        )
    return float(td[n1, n2]), max_n


def compute_teds(pred: ParsedTable | str, gold: ParsedTable | str, *, structure_only: bool = False) -> float:
    """Compute PubTabNet TEDS (or TEDS-S when ``structure_only=True``) in ``[0.0, 1.0]``."""
    pt_p = parse_html_table(pred) if isinstance(pred, str) else pred
    pt_g = parse_html_table(gold) if isinstance(gold, str) else gold
    if pt_p.n_rows == 0 and pt_g.n_rows == 0:
        return 1.0
    if pt_p.n_rows == 0 or pt_g.n_rows == 0:
        return 0.0
    dom_p = _table_to_dom(pt_p)
    dom_g = _table_to_dom(pt_g)
    dist, max_n = _zhang_shasha_ted(dom_p, dom_g, structure_only=structure_only)
    return float(max(0.0, min(1.0, 1.0 - (dist / float(max(1, max_n))))))


def compute_teds_structure(pred: ParsedTable | str, gold: ParsedTable | str) -> float:
    """Compute structure-only TEDS-S in ``[0.0, 1.0]``."""
    return compute_teds(pred, gold, structure_only=True)


def _span_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return float(inter) / float(max(1, union))


def _factored_2d_grits(
    rows_a: int,
    cols_a: int,
    rows_b: int,
    cols_b: int,
    sim_fn: Any,
) -> float:
    """Factored 2D maximum-similarity subgrid F1 (GriTS, Smock et al. AAAI 2023)."""
    if rows_a == 0 and rows_b == 0:
        return 1.0
    if rows_a == 0 or rows_b == 0 or cols_a == 0 or cols_b == 0:
        return 0.0

    # Align rows via exact 1D LCS-style DP where row-pair similarity uses column-aligned cell DP.
    row_dp = np.zeros((rows_a + 1, rows_b + 1), dtype=np.float64)
    for ra in range(rows_a):
        for rb in range(rows_b):
            col_dp = np.zeros((cols_a + 1, cols_b + 1), dtype=np.float64)
            for ca in range(cols_a):
                for cb in range(cols_b):
                    s_val = float(sim_fn(ra, ca, rb, cb))
                    col_dp[ca + 1, cb + 1] = max(
                        col_dp[ca, cb + 1],
                        col_dp[ca + 1, cb],
                        col_dp[ca, cb] + s_val,
                    )
            row_sim = float(col_dp[cols_a, cols_b])
            row_dp[ra + 1, rb + 1] = max(
                row_dp[ra, rb + 1],
                row_dp[ra + 1, rb],
                row_dp[ra, rb] + row_sim,
            )
    matched = float(row_dp[rows_a, rows_b])
    denom = float(rows_a * cols_a + rows_b * cols_b)
    return float(max(0.0, min(1.0, (2.0 * matched) / max(1.0, denom))))


def compute_grits(pred: ParsedTable | str, gold: ParsedTable | str) -> dict[str, float]:
    """Compute GriTS-Top (2D span topology) and GriTS-Con (2D cell text content) in ``[0.0, 1.0]``."""
    pt_p = parse_html_table(pred) if isinstance(pred, str) else pred
    pt_g = parse_html_table(gold) if isinstance(gold, str) else gold
    if pt_p.n_rows == 0 and pt_g.n_rows == 0:
        return {"grits_top": 1.0, "grits_con": 1.0, "grits_mean": 1.0}
    if pt_p.n_rows == 0 or pt_g.n_rows == 0:
        return {"grits_top": 0.0, "grits_con": 0.0, "grits_mean": 0.0}

    p_top = pt_p.span_grid
    g_top = pt_g.span_grid
    p_txt = [[normalize_ocr_text(c, case_sensitive=False) for c in row] for row in pt_p.grid]
    g_txt = [[normalize_ocr_text(c, case_sensitive=False) for c in row] for row in pt_g.grid]

    top = _factored_2d_grits(
        pt_p.n_rows,
        pt_p.n_cols,
        pt_g.n_rows,
        pt_g.n_cols,
        lambda ra, ca, rb, cb: _span_iou(p_top[ra][ca], g_top[rb][cb]),
    )
    con = _factored_2d_grits(
        pt_p.n_rows,
        pt_p.n_cols,
        pt_g.n_rows,
        pt_g.n_cols,
        lambda ra, ca, rb, cb: 1.0 - normalized_edit_distance(p_txt[ra][ca], g_txt[rb][cb]),
    )
    return {
        "grits_top": round(top, 6),
        "grits_con": round(con, 6),
        "grits_mean": round(0.5 * (top + con), 6),
    }


def _normalize_numeric_token(raw: str) -> str:
    s = raw.strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if s.startswith("-"):
        neg = True
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]
    try:
        val = float(s)
        if val == 0.0:
            neg = False
        canon = f"{val:.6f}".rstrip("0").rstrip(".") or "0"
    except ValueError:
        canon = s
    return f"{'-' if neg else ''}{canon}{'%' if is_pct else ''}"


def compute_numeric_cell_f1(pred: ParsedTable | str, gold: ParsedTable | str) -> float:
    """Compute exact multiset F1 over normalized financial/numeric tokens in table cells."""
    pt_p = parse_html_table(pred) if isinstance(pred, str) else pred
    pt_g = parse_html_table(gold) if isinstance(gold, str) else gold

    p_nums = [
        _normalize_numeric_token(m.group(0))
        for row in pt_p.grid
        for cell in row
        for m in _NUMERIC_TOKEN_RE.finditer(cell)
    ]
    g_nums = [
        _normalize_numeric_token(m.group(0))
        for row in pt_g.grid
        for cell in row
        for m in _NUMERIC_TOKEN_RE.finditer(cell)
    ]
    if not p_nums and not g_nums:
        return 1.0
    if not p_nums or not g_nums:
        return 0.0
    cp = Counter(p_nums)
    cg = Counter(g_nums)
    inter = sum((cp & cg).values())
    return float((2.0 * float(inter)) / float(len(p_nums) + len(g_nums)))


# ---------------------------------------------------------------------------
# Formula Canonicalization & Math Matching
# ---------------------------------------------------------------------------

_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|left|right|big|Big|bigg|Bigg|quad|qquad|limits|nolimits)\b"
)
_LATEX_FONT_RE = re.compile(r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|text|textbf|textit|operatorname)\s*\{")
_LATEX_SPACE_CMD_RE = re.compile(r"\\[,;:!\s]")


def canonicalize_latex(expr: str) -> str:
    """Normalize cosmetic LaTeX formatting differences while preserving math symbols and structure."""
    s = unicodedata.normalize("NFKC", expr or "").strip()
    for delim in ("$$", r"\[", r"\]", r"\(", r"\)", "$"):
        if s.startswith(delim):
            s = s[len(delim) :]
        if s.endswith(delim):
            s = s[: -len(delim)]
    s = re.sub(r"\\to(?![A-Za-z])", r"\\rightarrow", s)
    s = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", s)
    s = re.sub(r"\\le(?![A-Za-z])", r"\\leq", s)
    s = re.sub(r"\\neq(?![A-Za-z])", r"\\ne", s)
    s = re.sub(r"\\cdot(?![A-Za-z])", "*", s)
    s = s.replace("−", "-")
    s = _LATEX_WRAPPER_RE.sub("", s)
    s = _LATEX_SPACE_CMD_RE.sub("", s)
    # Strip outer font macro names while keeping their inner arguments.
    prev = None
    while prev != s:
        prev = s
        m = _LATEX_FONT_RE.search(s)
        if not m:
            break
        start = m.start()
        brace_open = m.end() - 1
        depth = 0
        end = -1
        for k in range(brace_open, len(s)):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end < 0:
            break
        s = s[:start] + s[brace_open + 1 : end] + s[end + 1 :]
    s = _WS_RE.sub("", s)
    # Peel redundant single-token braces `{x}` -> `x` repeatedly.
    s = re.sub(r"\{([A-Za-z0-9])\}", r"\1", s)
    return s


def extract_latex_equations(text: str) -> list[str]:
    """Extract display and inline LaTeX equations from Markdown OCR output."""
    eqs: list[str] = []
    for m in _DISPLAY_MATH_RE.finditer(text or ""):
        body = (m.group(1) if m.group(1) is not None else m.group(2)) or ""
        if body.strip():
            eqs.append(body.strip())
    masked = _DISPLAY_MATH_RE.sub(" ", text or "")
    for m in _INLINE_MATH_RE.finditer(masked):
        body = (m.group(1) if m.group(1) is not None else m.group(2)) or ""
        if body.strip():
            eqs.append(body.strip())
    return eqs


# ---------------------------------------------------------------------------
# Suite 1: olmOCR-Bench Evaluator
# ---------------------------------------------------------------------------


def _slice_edge_window(text: str, *, first_n: int | None, last_n: int | None) -> str:
    if first_n is not None and int(first_n) > 0:
        return text[: int(first_n)]
    if last_n is not None and int(last_n) > 0:
        return text[-int(last_n) :]
    return text


def _cell_matches(cell_text: str, target: str, *, max_diffs: int) -> bool:
    cn = normalize_ocr_text(cell_text, case_sensitive=False, strip_markdown=True)
    tn = normalize_ocr_text(target, case_sensitive=False, strip_markdown=True)
    if not tn:
        return cn == ""
    if not cn:
        return False
    if tn in cn:
        return True
    if levenshtein_distance(cn, tn) <= max_diffs:
        return True
    ok, _, _ = fuzzy_find_substring(tn, cn, max_diffs=max_diffs)
    return ok


def run_olmocr_unit_test(
    test_spec: Mapping[str, Any],
    prediction: str,
    *,
    parsed_tables: Sequence[ParsedTable] | None = None,
    extracted_equations: Sequence[str] | None = None,
) -> bool:
    """Execute one deterministic ``allenai/olmOCR-bench`` binary unit test."""
    raw = prediction or ""
    ttype = str(test_spec.get("type") or "").strip().lower()
    max_diffs = max(0, int(test_spec.get("max_diffs") or 0))

    if ttype == "baseline":
        if not raw.strip():
            return False
        if repeat_4gram_rate(raw) >= 0.15:
            return False
        if bool(test_spec.get("check_disallowed_characters", False)):
            if "\ufffd" in raw or "\x00" in raw:
                return False
            ctrl = sum(1 for ch in raw if ord(ch) < 32 and ch not in ("\n", "\r", "\t"))
            if ctrl > max(2, int(0.01 * len(raw))):
                return False
        return True

    if not raw.strip():
        # Empty outputs must fail every presence, absence, order, table, and math assertion.
        return False

    if ttype in ("present", "absent"):
        case_sens = bool(test_spec.get("case_sensitive", True))
        target = normalize_ocr_text(str(test_spec.get("text") or ""), case_sensitive=case_sens, strip_markdown=True)
        if not target:
            return ttype == "absent"
        window = _slice_edge_window(raw, first_n=test_spec.get("first_n"), last_n=test_spec.get("last_n"))
        hay = normalize_ocr_text(window, case_sensitive=case_sens, strip_markdown=True)
        found, _, _ = fuzzy_find_substring(target, hay, max_diffs=max_diffs)
        return found if ttype == "present" else (not found)

    if ttype == "order":
        before = normalize_ocr_text(str(test_spec.get("before") or ""), case_sensitive=True, strip_markdown=True)
        after = normalize_ocr_text(str(test_spec.get("after") or ""), case_sensitive=True, strip_markdown=True)
        if not before or not after:
            return False
        hay = normalize_ocr_text(raw, case_sensitive=True, strip_markdown=True)
        ok_b, _, end_b = fuzzy_find_substring(before, hay, max_diffs=max_diffs)
        if not ok_b:
            return False
        # Search for `after` strictly after `before` starts (allowing small overlap at boundary).
        start_after_search = max(0, end_b - min(len(before), max_diffs + 2))
        ok_a, _, _ = fuzzy_find_substring(after, hay[start_after_search:], max_diffs=max_diffs)
        return ok_a

    if ttype == "table":
        tables = list(parsed_tables) if parsed_tables is not None else extract_tables_from_markdown(raw)
        if not tables:
            return False
        cell_q = str(test_spec.get("cell") or "")
        up_q = test_spec.get("up")
        down_q = test_spec.get("down")
        left_q = test_spec.get("left")
        right_q = test_spec.get("right")
        top_h_q = test_spec.get("top_heading")
        left_h_q = test_spec.get("left_heading")

        for tbl in tables:
            for r in range(tbl.n_rows):
                for c in range(tbl.n_cols):
                    if not _cell_matches(tbl.grid[r][c], cell_q, max_diffs=max_diffs):
                        continue
                    if up_q is not None and not (r > 0 and _cell_matches(tbl.grid[r - 1][c], str(up_q), max_diffs=max_diffs)):
                        continue
                    if down_q is not None and not (
                        r + 1 < tbl.n_rows and _cell_matches(tbl.grid[r + 1][c], str(down_q), max_diffs=max_diffs)
                    ):
                        continue
                    if left_q is not None and not (c > 0 and _cell_matches(tbl.grid[r][c - 1], str(left_q), max_diffs=max_diffs)):
                        continue
                    if right_q is not None and not (
                        c + 1 < tbl.n_cols and _cell_matches(tbl.grid[r][c + 1], str(right_q), max_diffs=max_diffs)
                    ):
                        continue
                    if top_h_q is not None and not any(
                        _cell_matches(tbl.grid[rr][c], str(top_h_q), max_diffs=max_diffs) for rr in range(r)
                    ):
                        continue
                    if left_h_q is not None and not any(
                        _cell_matches(tbl.grid[r][cc], str(left_h_q), max_diffs=max_diffs) for cc in range(c)
                    ):
                        continue
                    return True
        return False

    if ttype == "math":
        target_canon = canonicalize_latex(str(test_spec.get("math") or ""))
        if not target_canon:
            return False
        eqs = list(extracted_equations) if extracted_equations is not None else extract_latex_equations(raw)
        # Allow minor LaTeX formatting tolerance proportional to formula length when KaTeX renderer is absent.
        eff_diffs = max(max_diffs, int(0.08 * len(target_canon)))
        for eq in eqs:
            cand = canonicalize_latex(eq)
            if not cand:
                continue
            if target_canon in cand:
                return True
            ok, _, _ = fuzzy_find_substring(target_canon, cand, max_diffs=eff_diffs)
            if ok:
                return True
        full_canon = canonicalize_latex(raw)
        ok_full, _, _ = fuzzy_find_substring(target_canon, full_canon, max_diffs=eff_diffs)
        return ok_full

    return False


def evaluate_olmocr_bench(
    pages: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    allow_html_tables: bool = True,
) -> dict[str, Any]:
    """Evaluate OCR predictions on ``allenai/olmOCR-bench`` pages and return per-page + macro metrics."""
    n = len(pages)
    if len(predictions) != n:
        raise ValueError(f"Mismatched pages={n} vs predictions={len(predictions)}")

    per_valid: list[bool] = []
    per_scores: list[float] = []
    repeat_rates: list[float] = []
    md_valid_cnt = 0
    cat_stats: dict[str, dict[str, int]] = {}

    for i, page in enumerate(pages):
        pred = predictions[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        rep = repeat_4gram_rate(pred)
        repeat_rates.append(rep)
        md_val = validate_ocr_markdown(pred, allow_html_tables=allow_html_tables)
        if md_val["valid"]:
            md_valid_cnt += 1
        is_valid = bool(pred.strip() and str(fr) != "length" and rep < 0.15)
        per_valid.append(is_valid)

        tests = list(page.get("tests") or [])
        tables = extract_tables_from_markdown(pred)
        eqs = extract_latex_equations(pred)
        passed = 0
        for ts in tests:
            if run_olmocr_unit_test(ts, pred, parsed_tables=tables, extracted_equations=eqs):
                passed += 1
        page_score = (float(passed) / float(len(tests))) if tests else (1.0 if is_valid else 0.0)
        per_scores.append(page_score)

        cat = str(page.get("category") or "unknown")
        st = cat_stats.setdefault(cat, {"n_pages": 0, "n_tests": 0, "n_passed": 0})
        st["n_pages"] += 1
        st["n_tests"] += len(tests)
        st["n_passed"] += passed

    per_category: dict[str, dict[str, Any]] = {}
    cat_rates: list[float] = []
    no_hf_rates: list[float] = []
    for cat in sorted(cat_stats):
        st = cat_stats[cat]
        rate = (float(st["n_passed"]) / float(st["n_tests"])) if st["n_tests"] > 0 else 0.0
        per_category[cat] = {
            "n_pages": st["n_pages"],
            "n_tests": st["n_tests"],
            "n_passed": st["n_passed"],
            "pass_rate": round(rate, 6),
        }
        cat_rates.append(rate)
        if cat != "headers_footers":
            no_hf_rates.append(rate)

    overall_macro = float(np.mean(cat_rates)) if cat_rates else 0.0
    overall_no_hf = float(np.mean(no_hf_rates)) if no_hf_rates else overall_macro
    return {
        "n_pages": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "valid_markdown_rate": round(float(md_valid_cnt) / float(max(1, n)), 6),
        "overall_macro": round(overall_macro, 6),
        "overall_no_hf": round(overall_no_hf, 6),
        "per_category": per_category,
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


# ---------------------------------------------------------------------------
# Suite 2: FinTabNet-OTSL Table Fidelity Evaluator
# ---------------------------------------------------------------------------


def evaluate_fintabnet_otsl(
    items: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    allow_html_tables: bool = True,
) -> dict[str, Any]:
    """Evaluate table extraction on ``docling-project/FinTabNet_OTSL`` (TEDS, TEDS-S, GriTS, Numeric F1)."""
    n = len(items)
    if len(predictions) != n:
        raise ValueError(f"Mismatched items={n} vs predictions={len(predictions)}")

    per_valid: list[bool] = []
    teds_scores: list[float] = []
    teds_s_scores: list[float] = []
    grits_top_scores: list[float] = []
    grits_con_scores: list[float] = []
    num_f1_scores: list[float] = []
    repeat_rates: list[float] = []
    md_valid_cnt = 0

    for i, item in enumerate(items):
        pred_raw = predictions[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        rep = repeat_4gram_rate(pred_raw)
        repeat_rates.append(rep)
        md_val = validate_ocr_markdown(pred_raw, allow_html_tables=allow_html_tables)
        if md_val["valid"]:
            md_valid_cnt += 1

        gold_tbl = parse_html_table(str(item.get("gold_html") or ""))
        pred_tables = extract_tables_from_markdown(pred_raw)
        if not pred_tables and ("<tr" in pred_raw.lower() or "<td" in pred_raw.lower()):
            pt_fb = parse_html_table(pred_raw)
            if pt_fb.n_rows > 0:
                pred_tables = [pt_fb]

        best_tbl = ParsedTable(0, 0, (), (), ())
        best_teds = 0.0
        for cand in pred_tables:
            sc = compute_teds(cand, gold_tbl, structure_only=False)
            if sc >= best_teds:
                best_teds = sc
                best_tbl = cand

        has_table = best_tbl.n_rows > 0 and best_tbl.n_cols > 0
        is_valid = bool(has_table and md_val["valid"] and str(fr) != "length" and rep < 0.15)
        teds_s = compute_teds_structure(best_tbl, gold_tbl) if has_table else 0.0
        grits = compute_grits(best_tbl, gold_tbl) if has_table else {"grits_top": 0.0, "grits_con": 0.0, "grits_mean": 0.0}
        num_f1 = compute_numeric_cell_f1(best_tbl, gold_tbl) if has_table else 0.0

        per_valid.append(is_valid)
        teds_scores.append(best_teds)
        teds_s_scores.append(teds_s)
        grits_top_scores.append(float(grits["grits_top"]))
        grits_con_scores.append(float(grits["grits_con"]))
        num_f1_scores.append(num_f1)

    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "valid_markdown_rate": round(float(md_valid_cnt) / float(max(1, n)), 6),
        "teds": round(float(np.mean(teds_scores)) if teds_scores else 0.0, 6),
        "teds_s": round(float(np.mean(teds_s_scores)) if teds_s_scores else 0.0, 6),
        "grits_top": round(float(np.mean(grits_top_scores)) if grits_top_scores else 0.0, 6),
        "grits_con": round(float(np.mean(grits_con_scores)) if grits_con_scores else 0.0, 6),
        "numeric_cell_f1": round(float(np.mean(num_f1_scores)) if num_f1_scores else 0.0, 6),
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": teds_scores,
    }


# ---------------------------------------------------------------------------
# Suite 3: OmniDocBench-EN Multi-Domain Page Evaluator
# ---------------------------------------------------------------------------


def _strip_tables_and_images_for_prose(md: str, *, strip_display_math: bool = False) -> str:
    s = _HTML_TABLE_BLOCK_RE.sub(" ", md or "")
    s = _IMG_TAG_RE.sub(" ", s)
    if strip_display_math:
        s = _DISPLAY_MATH_RE.sub(" ", s)
    s = _MD_ATX_PREFIX_RE.sub("", s)
    lines = s.replace("\r\n", "\n").split("\n")
    kept: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if i + 1 < n and "|" in lines[i] and _is_gfm_sep_row(lines[i + 1]):
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                j += 1
            i = j
            continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept)


def _reading_order_concordance(pred_prose: str, ordered_blocks: Sequence[Mapping[str, Any]]) -> float:
    """Pairwise reading-order concordance across located gold blocks in ``[0.0, 1.0]``."""
    if len(ordered_blocks) < 2:
        return 1.0
    hay = normalize_ocr_text(pred_prose, case_sensitive=False, strip_markdown=True)
    if not hay:
        return 0.0
    positions: list[int] = []
    for blk in ordered_blocks:
        probe = normalize_ocr_text(str(blk.get("text") or ""), case_sensitive=False, strip_markdown=True)[:48]
        if len(probe) < 6:
            continue
        max_d = max(1, int(0.15 * len(probe)))
        ok, _, end_idx = fuzzy_find_substring(probe, hay, max_diffs=max_d)
        if ok:
            positions.append(end_idx - len(probe))
    if len(positions) < 2:
        return 1.0 if len(positions) == 1 else 0.0
    concordant = 0
    total = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total += 1
            if positions[i] <= positions[j]:
                concordant += 1
    return float(concordant) / float(max(1, total))


def _bipartite_mean_match(
    preds: Sequence[Any],
    golds: Sequence[Any],
    sim_fn: Any,
) -> float:
    """Hungarian-matched F1-style mean similarity over variable-count tables or formulas."""
    if not golds and not preds:
        return 1.0
    if not golds or not preds:
        return 0.0
    mat = np.zeros((len(preds), len(golds)), dtype=np.float64)
    for r, pe in enumerate(preds):
        for c, ge in enumerate(golds):
            mat[r, c] = float(sim_fn(pe, ge))
    r_ind, c_ind = linear_sum_assignment_max(mat)
    matched = float(np.sum(mat[r_ind, c_ind]))
    # Penalize omitted gold elements while tolerating extra candidate equations/inline formulas gently.
    denom = float(max(len(golds), len(preds)))
    return float(max(0.0, min(1.0, matched / max(1.0, denom))))


def evaluate_omnidocbench(
    pages: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    allow_html_tables: bool = True,
) -> dict[str, Any]:
    """Evaluate end-to-end page parsing on ``opendatalab/OmniDocBench`` English pages."""
    n = len(pages)
    if len(predictions) != n:
        raise ValueError(f"Mismatched pages={n} vs predictions={len(predictions)}")

    per_valid: list[bool] = []
    per_scores: list[float] = []
    text_neds: list[float] = []
    ro_sims: list[float] = []
    table_teds_list: list[float] = []
    formula_sims: list[float] = []
    repeat_rates: list[float] = []
    md_valid_cnt = 0

    for i, page in enumerate(pages):
        pred = predictions[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        rep = repeat_4gram_rate(pred)
        repeat_rates.append(rep)
        md_val = validate_ocr_markdown(pred, allow_html_tables=allow_html_tables)
        if md_val["valid"]:
            md_valid_cnt += 1
        is_valid = bool(pred.strip() and str(fr) != "length" and rep < 0.15)
        per_valid.append(is_valid)

        text_blocks = list(page.get("text_blocks") or [])
        gold_tables = [parse_html_table(str(t.get("html") or "")) for t in (page.get("tables") or [])]
        gold_tables = [gt for gt in gold_tables if gt.n_rows > 0 and gt.n_cols > 0]
        gold_eqs = [canonicalize_latex(str(e.get("latex") or "")) for e in (page.get("equations") or [])]
        gold_eqs = [ge for ge in gold_eqs if ge]

        components: list[float] = []
        if text_blocks:
            gold_prose = normalize_ocr_text(
                " ".join(str(b.get("text") or "") for b in text_blocks),
                case_sensitive=False,
                strip_markdown=True,
            )
            pred_prose_raw = _strip_tables_and_images_for_prose(pred, strip_display_math=bool(gold_eqs))
            pred_prose = normalize_ocr_text(pred_prose_raw, case_sensitive=False, strip_markdown=True)
            ned = normalized_edit_distance(pred_prose, gold_prose)
            text_neds.append(ned)
            components.append(1.0 - ned)
            ro_sims.append(_reading_order_concordance(pred_prose_raw, text_blocks))

        if gold_tables:
            pred_tables = extract_tables_from_markdown(pred)
            t_sc = _bipartite_mean_match(
                pred_tables,
                gold_tables,
                lambda pt, gt: compute_teds(pt, gt, structure_only=False),
            )
            table_teds_list.append(t_sc)
            components.append(t_sc)

        if gold_eqs:
            pred_eqs = [canonicalize_latex(e) for e in extract_latex_equations(pred)]
            pred_eqs = [pe for pe in pred_eqs if pe]
            if not pred_eqs:
                f_sc = 0.0
            else:
                mat = np.zeros((len(pred_eqs), len(gold_eqs)), dtype=np.float64)
                for r, pe in enumerate(pred_eqs):
                    for c, ge in enumerate(gold_eqs):
                        mat[r, c] = 1.0 - normalized_edit_distance(pe, ge)
                r_ind, c_ind = linear_sum_assignment_max(mat)
                # Divide by len(gold_eqs) so inline formulas in prose do not penalize display-equation recall.
                f_sc = float(np.sum(mat[r_ind, c_ind])) / float(len(gold_eqs))
            formula_sims.append(f_sc)
            components.append(f_sc)

        per_scores.append(float(np.mean(components)) if components else (1.0 if is_valid else 0.0))

    mean_text_ned = float(np.mean(text_neds)) if text_neds else 0.0
    mean_table_teds = float(np.mean(table_teds_list)) if table_teds_list else None
    mean_formula_sim = float(np.mean(formula_sims)) if formula_sims else None
    pillars = [1.0 - mean_text_ned]
    if mean_table_teds is not None:
        pillars.append(mean_table_teds)
    if mean_formula_sim is not None:
        pillars.append(mean_formula_sim)
    omnidoc_overall = float(np.mean(pillars))

    return {
        "n_pages": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "valid_markdown_rate": round(float(md_valid_cnt) / float(max(1, n)), 6),
        "omnidoc_overall": round(omnidoc_overall, 6),
        "text_edit_distance": round(mean_text_ned, 6),
        "text_similarity": round(1.0 - mean_text_ned, 6),
        "reading_order_sim": round(float(np.mean(ro_sims)) if ro_sims else 1.0, 6),
        "table_teds": round(mean_table_teds, 6) if mean_table_teds is not None else None,
        "formula_norm_sim": round(mean_formula_sim, 6) if mean_formula_sim is not None else None,
        "n_table_pages": len(table_teds_list),
        "n_formula_pages": len(formula_sims),
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


# ---------------------------------------------------------------------------
# Suite 4: CORD-v2 Visual Structured JSON KIE Evaluator
# ---------------------------------------------------------------------------


def _recursive_leaf_anls(pred: Any, gold: Any) -> float:
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    if isinstance(gold, Mapping) and isinstance(pred, Mapping):
        keys = sorted({str(k) for k, v in gold.items() if v is not None} | {str(k) for k, v in pred.items() if v is not None})
        if not keys:
            return 1.0
        return sum(_recursive_leaf_anls(pred.get(k), gold.get(k)) for k in keys) / float(len(keys))
    if isinstance(gold, list) and isinstance(pred, list):
        if not gold and not pred:
            return 1.0
        if not gold or not pred:
            return 0.0
        mat = np.zeros((len(pred), len(gold)), dtype=np.float64)
        for r, pe in enumerate(pred):
            for c, ge in enumerate(gold):
                mat[r, c] = _recursive_leaf_anls(pe, ge)
        r_ind, c_ind = linear_sum_assignment_max(mat)
        return float((2.0 * float(np.sum(mat[r_ind, c_ind]))) / float(len(pred) + len(gold)))
    return anls_score(str(pred), str(gold))


def _score_structured_kie_suite(
    items: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
    gold_key: str = "gt_parse",
) -> dict[str, Any]:
    base = evaluate_structured_extraction(
        items,
        raw_outputs,
        finish_reasons=finish_reasons,
        token_ids_list=token_ids_list,
        schema_key="schema",
        gold_key=gold_key,
    )
    anls_vals: list[float] = []
    for item, raw, valid in zip(items, raw_outputs, base["per_item_valid"]):
        if not valid:
            anls_vals.append(0.0)
            continue
        try:
            parsed = json.loads(_strip_json_fences(raw or ""))
            anls_vals.append(_recursive_leaf_anls(parsed, item.get(gold_key)))
        except Exception:
            anls_vals.append(0.0)
    return {
        **base,
        "anls": round(float(np.mean(anls_vals)) if anls_vals else 0.0, 6),
    }


def evaluate_cord_v2(
    items: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate visual JSON extraction on ``naver-clova-ix/cord-v2`` via ``grounded_field_f1`` + ANLS."""
    return _score_structured_kie_suite(
        items,
        raw_outputs,
        finish_reasons=finish_reasons,
        token_ids_list=token_ids_list,
    )


# ---------------------------------------------------------------------------
# Suite 5–8: VRDU (FCC Ad-Buy / DOJ FARA), DocLayNet-Gov, PureDocBench, US Gov
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_VRDU_MDY_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$")
_VRDU_YMD_RE = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$")
_VRDU_TEXT_DATE_RE = re.compile(
    r"(?:(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([a-z]+)[,\s\-]+(\d{2,4})|([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,\s\-]+(\d{2,4}))",
    re.IGNORECASE,
)


def _expand_year(y_raw: str) -> int:
    y = int(y_raw)
    if len(y_raw) <= 2:
        return (2000 + y) if y < 40 else (1900 + y)
    return y


def normalize_vrdu_value(val: Any, match_func: str) -> str | None:
    """Canonicalize a VRDU scalar leaf according to its ``meta.json`` match function."""
    if val is None:
        return None
    s = unicodedata.normalize("NFKC", str(val)).translate(_UNICODE_PUNCT_TRANS)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return None

    mf = str(match_func or "GeneralStringMatch")
    if mf == "PriceMatch":
        neg = s.startswith("(") and s.endswith(")")
        cleaned = s[1:-1] if neg else s
        cleaned = re.sub(r"[\$,\s]", "", cleaned)
        try:
            num = float(cleaned)
            if neg:
                num = -abs(num)
            return f"{num:.2f}"
        except ValueError:
            return cleaned.lower()
    if mf == "DateMatch":
        m = _VRDU_YMD_RE.match(s)
        if m:
            return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        m = _VRDU_MDY_RE.match(s)
        if m:
            mo, d, y = int(m.group(1)), int(m.group(2)), _expand_year(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"
        m = _VRDU_TEXT_DATE_RE.search(s)
        if m:
            if m.group(1):
                d, mo_s, y_s = int(m.group(1)), m.group(2).lower(), m.group(3)
            else:
                mo_s, d, y_s = m.group(4).lower(), int(m.group(5)), m.group(6)
            mo = _MONTH_MAP.get(mo_s)
            if mo is not None and 1 <= d <= 31:
                return f"{_expand_year(y_s):04d}-{mo:02d}-{d:02d}"
        return s.lower()
    if mf == "NumericalStringMatch":
        digits_only = re.sub(r"[\s,]", "", s)
        return digits_only
    return s


def normalize_vrdu_parse(obj: Any, meta: Mapping[str, Any]) -> Any:
    """Apply VRDU type-specific normalization across flat fields and nested ``line_items``."""
    if not isinstance(obj, Mapping):
        return obj
    match_map = dict(meta.get("entity_name_to_match_func") or {})
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if k == "line_items" and isinstance(v, list):
            norm_rows: list[dict[str, Any]] = []
            for row in v:
                if not isinstance(row, Mapping):
                    continue
                nr: dict[str, Any] = {}
                for rk, rv in row.items():
                    nv = normalize_vrdu_value(rv, str(match_map.get(rk, "GeneralStringMatch")))
                    if nv is not None:
                        nr[str(rk)] = nv
                if nr:
                    norm_rows.append(nr)
            out["line_items"] = norm_rows
        else:
            nv = normalize_vrdu_value(v, str(match_map.get(k, "GeneralStringMatch")))
            if nv is not None:
                out[str(k)] = nv
    return out


def evaluate_vrdu(
    items: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate structured KIE on ``google-research-datasets/vrdu`` via type-normalized ``grounded_field_f1`` + ANLS."""
    norm_items: list[dict[str, Any]] = []
    norm_raws: list[str] = []
    for item, raw in zip(items, raw_outputs):
        meta = item.get("meta") if isinstance(item.get("meta"), Mapping) else {}
        norm_gold = normalize_vrdu_parse(item.get("gt_parse") or {}, meta)
        norm_items.append({**item, "_norm_gt": norm_gold})
        try:
            parsed = json.loads(_strip_json_fences(raw or ""))
            norm_pred = normalize_vrdu_parse(parsed, meta)
            norm_raws.append(json.dumps(norm_pred, ensure_ascii=False))
        except Exception:
            norm_raws.append(raw or "")
    return _score_structured_kie_suite(
        norm_items,
        norm_raws,
        finish_reasons=finish_reasons,
        token_ids_list=token_ids_list,
        gold_key="_norm_gt",
    )


def evaluate_doclaynet_gov(
    pages: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    allow_html_tables: bool = True,
) -> dict[str, Any]:
    """Evaluate visual OCR on ``DocLayNet-v1.2`` gov/legal/patent pages using layout text cells & reading order."""
    n = len(pages)
    if len(predictions) != n:
        raise ValueError(f"Mismatched pages={n} vs predictions={len(predictions)}")

    per_valid: list[bool] = []
    per_scores: list[float] = []
    text_neds: list[float] = []
    ro_sims: list[float] = []
    hf_leaks: list[float] = []
    repeat_rates: list[float] = []
    md_valid_cnt = 0

    for i, page in enumerate(pages):
        pred = predictions[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        rep = repeat_4gram_rate(pred)
        repeat_rates.append(rep)
        md_val = validate_ocr_markdown(pred, allow_html_tables=allow_html_tables)
        if md_val["valid"]:
            md_valid_cnt += 1
        is_valid = bool(pred.strip() and str(fr) != "length" and rep < 0.15)
        per_valid.append(is_valid)

        text_blocks = list(page.get("text_blocks") or [])
        table_texts = [str(t) for t in (page.get("table_texts") or []) if str(t).strip()]
        gold_parts = [str(b.get("text") or "") for b in text_blocks] + table_texts
        gold_norm = normalize_ocr_text(" ".join(gold_parts), case_sensitive=False, strip_markdown=True)
        pred_norm = normalize_ocr_text(
            _strip_tables_and_images_for_prose(pred, strip_display_math=False),
            case_sensitive=False,
            strip_markdown=True,
        )
        ned = normalized_edit_distance(pred_norm, gold_norm)
        text_neds.append(ned)
        ro = _reading_order_concordance(pred, text_blocks) if text_blocks else 1.0
        ro_sims.append(ro)

        hf_texts = [str(h) for h in (page.get("header_footer_texts") or []) if len(str(h).strip()) >= 4]
        if hf_texts:
            leaked = 0
            for h in hf_texts:
                hn = normalize_ocr_text(h, case_sensitive=False, strip_markdown=True)
                if hn and fuzzy_find_substring(hn, pred_norm, max_diffs=1)[0]:
                    leaked += 1
            hf_leaks.append(float(leaked) / float(len(hf_texts)))

        per_scores.append(0.5 * (1.0 - ned) + 0.5 * ro)

    mean_ned = float(np.mean(text_neds)) if text_neds else 0.0
    mean_ro = float(np.mean(ro_sims)) if ro_sims else 1.0
    overall = 0.5 * (1.0 - mean_ned) + 0.5 * mean_ro
    return {
        "n_pages": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "valid_markdown_rate": round(float(md_valid_cnt) / float(max(1, n)), 6),
        "doclaynet_overall": round(overall, 6),
        "text_edit_distance": round(mean_ned, 6),
        "text_similarity": round(1.0 - mean_ned, 6),
        "reading_order_sim": round(mean_ro, 6),
        "header_footer_leak_rate": round(float(np.mean(hf_leaks)) if hf_leaks else 0.0, 6),
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


def evaluate_puredocbench(
    pages: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    allow_html_tables: bool = True,
) -> dict[str, Any]:
    """Evaluate PureDocBench pages via OmniDocBench metrics with track metadata."""
    res = evaluate_omnidocbench(
        pages,
        predictions,
        finish_reasons=finish_reasons,
        allow_html_tables=allow_html_tables,
    )
    tracks = sorted({str(p.get("track") or "clean") for p in pages})
    domains = sorted({str(p.get("domain") or "03_legal_gov") for p in pages})
    return {
        **res,
        "puredocbench_overall": res["omnidoc_overall"],
        "tracks": tracks,
        "domains": domains,
    }


def evaluate_us_gov_ocr(
    pages: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    allow_html_tables: bool = True,
) -> dict[str, Any]:
    """Evaluate paired U.S. Gov PDF<->XML/transcription pages with branch & source breakdowns."""
    res = evaluate_omnidocbench(
        pages,
        predictions,
        finish_reasons=finish_reasons,
        allow_html_tables=allow_html_tables,
    )
    by_branch: dict[str, dict[str, Any]] = {}
    by_source: dict[str, dict[str, Any]] = {}
    for idx, page in enumerate(pages):
        br = str(page.get("branch") or "unknown")
        src = str(page.get("source") or "unknown")
        sc = float(res["per_item_scores"][idx])
        ok = bool(res["per_item_valid"][idx])
        for bucket_map, key in ((by_branch, br), (by_source, src)):
            entry = bucket_map.setdefault(key, {"_scores": [], "_valid": 0})
            entry["_scores"].append(sc)
            entry["_valid"] += int(ok)
    for bucket_map in (by_branch, by_source):
        for key, entry in bucket_map.items():
            scs = entry.pop("_scores")
            vld = entry.pop("_valid")
            entry["n_pages"] = len(scs)
            entry["n_valid"] = vld
            entry["score"] = round(float(np.mean(scs)) if scs else 0.0, 6)
    return {
        **res,
        "us_gov_overall": res["omnidoc_overall"],
        "per_branch": by_branch,
        "per_source": by_source,
    }


__all__ = [
    "ParsedTable",
    "TableCell",
    "anls_score",
    "canonicalize_latex",
    "compute_grits",
    "compute_numeric_cell_f1",
    "compute_teds",
    "compute_teds_structure",
    "evaluate_cord_v2",
    "evaluate_doclaynet_gov",
    "evaluate_fintabnet_otsl",
    "evaluate_olmocr_bench",
    "evaluate_omnidocbench",
    "evaluate_puredocbench",
    "evaluate_us_gov_ocr",
    "evaluate_vrdu",
    "extract_latex_equations",
    "extract_tables_from_markdown",
    "fuzzy_find_substring",
    "levenshtein_distance",
    "normalize_ocr_text",
    "normalize_vrdu_parse",
    "normalize_vrdu_value",
    "normalized_edit_distance",
    "normalized_indel_similarity",
    "parse_gfm_table_lines",
    "parse_html_table",
    "run_olmocr_unit_test",
    "validate_ocr_markdown",
]
