"""Zero-latency structured Markdown synthesizer, HTML->GFM 2D-grid canonicalizer, validator, and JIT table grammar."""

from __future__ import annotations

from html.parser import HTMLParser
import math
import re
from typing import Any, Sequence

import numpy as np

from .types import MarkdownTableMode, MarkdownValidation, OCRBox, OCRLine

_LIST_BULLET_RE = re.compile(r"^(?:[-*•·]|(?:\d{1,3}|[a-zA-Z])[.)])\s+")
_NUMERIC_CELL_RE = re.compile(r"^(?:[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\(\$?\d[\d,]*(?:\.\d+)?\)|—|-|N/?A)$", re.I)
_FENCE_RE = re.compile(r"^[ ]{0,3}(```+|~~~+)(.*)$")
_HEADING_SPACE_RE = re.compile(r"^(#{1,6})([^\s#])")
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_PAIR_RE = re.compile(r"<(/?)(table|tr|td|th)\b[^>]*>", re.IGNORECASE)
_GFM_SEP_CELL_RE = re.compile(r"^:?-{1,}:?$")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")


def _escape_gfm_cell(text: str) -> str:
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    return _UNESCAPED_PIPE_RE.sub(r"\\|", clean)


def _ends_with_unescaped_pipe(s: str) -> bool:
    if not s.endswith("|"):
        return False
    n_bs = 0
    idx = len(s) - 2
    while idx >= 0 and s[idx] == "\\":
        n_bs += 1
        idx -= 1
    return (n_bs % 2) == 0


def _split_gfm_row_cells(line: str) -> list[str]:
    """Split a GFM pipe-table row on unescaped `|` only, preserving `\\|` inside cells."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if _ends_with_unescaped_pipe(s):
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


def _format_gfm_table(rows: list[list[str]], *, header_rows: int = 1) -> list[str]:
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    if n_cols <= 0:
        return []
    padded = [list(r) + [""] * (n_cols - len(r)) for r in rows]

    # Collapse multi-row table header cleanly when specified
    if header_rows > 1 and len(padded) > header_rows:
        merged_hdr: list[str] = []
        for c in range(n_cols):
            parts: list[str] = []
            for r in range(header_rows):
                val = padded[r][c].strip()
                if val and (not parts or val != parts[-1]):
                    parts.append(val)
            merged_hdr.append(" / ".join(parts))
        header = merged_hdr
        body = padded[header_rows:]
    else:
        header = padded[0]
        body = padded[1:]

    # Right-align columns that contain mostly monetary/numeric values
    seps: list[str] = []
    for c in range(n_cols):
        col_vals = [r[c].strip() for r in body if r[c].strip()]
        num_hits = sum(1 for v in col_vals if _NUMERIC_CELL_RE.match(v))
        seps.append("---:" if col_vals and num_hits * 2 >= len(col_vals) else "---")

    out = [
        "| " + " | ".join(_escape_gfm_cell(v) for v in header) + " |",
        "| " + " | ".join(seps) + " |",
    ]
    for r in body:
        out.append("| " + " | ".join(_escape_gfm_cell(v) for v in r) + " |")
    return out


class _HtmlTableParser(HTMLParser):
    """Exact 2D-occupancy parser expanding `colspan` and `rowspan` without inventing cells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, bool]]] = []
        self._in_tr = False
        self._in_cell = False
        self._cell_is_th = False
        self._cell_buf: list[str] = []
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._occupied: dict[tuple[int, int], tuple[str, bool, int, bool]] = {}
        self._cur_row_idx = -1
        self._cur_col = 0

    def _safe_span(self, val: str | None) -> int:
        if not val:
            return 1
        try:
            return max(1, min(64, int(val.strip())))
        except ValueError:
            return 1

    def _flush_cell(self) -> None:
        if not self._in_cell:
            return
        text = " ".join("".join(self._cell_buf).split())
        if self._cur_row_idx < 0:
            self._cur_row_idx = 0
            self._cur_col = 0
            self.rows.append([])
        r = self._cur_row_idx
        while (r, self._cur_col) in self._occupied:
            self._cur_col += 1
        c = self._cur_col
        for dr in range(self._cell_rowspan):
            for dc in range(self._cell_colspan):
                # Preserve text at origin cell (dr==0, dc==0); keep horizontal colspan text for
                # grouped `<th>` multi-row headers only, leaving body `colspan` & `rowspan` continuations empty.
                cell_txt = text if dr == 0 else ""
                self._occupied[(r + dr, c + dc)] = (
                    cell_txt,
                    self._cell_is_th,
                    self._cell_colspan,
                    dr == 0 and dc == 0,
                )
        self._cur_col += self._cell_colspan
        self._in_cell = False
        self._cell_buf = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "tr":
            self._flush_cell()
            self._cur_row_idx += 1
            self._cur_col = 0
            self._in_tr = True
            while len(self.rows) <= self._cur_row_idx:
                self.rows.append([])
        elif t in ("td", "th"):
            if self._in_cell:
                self._flush_cell()
            if not self._in_tr:
                self._cur_row_idx += 1
                self._cur_col = 0
                self._in_tr = True
                while len(self.rows) <= self._cur_row_idx:
                    self.rows.append([])
            self._in_cell = True
            self._cell_is_th = t == "th"
            self._cell_buf = []
            amap = dict(attrs)
            self._cell_colspan = self._safe_span(amap.get("colspan"))
            self._cell_rowspan = self._safe_span(amap.get("rowspan"))
        elif t == "br" and self._in_cell:
            self._cell_buf.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            self._flush_cell()
        elif t == "tr":
            self._flush_cell()
            self._in_tr = False

    def close(self) -> None:
        super().close()
        self._flush_cell()
        self._in_tr = False

    def materialize(self) -> tuple[list[list[str]], int]:
        if not self._occupied:
            return [], 0
        max_r = max(r for r, _ in self._occupied) + 1
        max_c = max(c for _, c in self._occupied) + 1
        header_rows = 0
        for r in range(max_r):
            row_ths = [self._occupied[(r, c)][1] for c in range(max_c) if (r, c) in self._occupied]
            if row_ths and all(row_ths):
                header_rows = r + 1
            else:
                break
        effective_hdr = max(1, header_rows) if max_r > 1 else 1
        grid: list[list[str]] = [[""] * max_c for _ in range(max_r)]
        for (r, c), (txt, is_th, colspan, is_origin) in self._occupied.items():
            if is_origin or (is_th and header_rows > 1 and r < header_rows and colspan < max_c):
                grid[r][c] = txt
        # Drop completely empty trailing ghost columns
        non_empty_cols = [c for c in range(max_c) if any(grid[r][c] != "" for r in range(max_r))]
        if non_empty_cols:
            last_c = max(non_empty_cols) + 1
            grid = [row[:last_c] for row in grid]
        return grid, min(effective_hdr, max(1, len(grid) - 1))


def _render_html_table_grid(grid: list[list[str]], header_rows: int) -> str:
    if not grid:
        return ""
    import html

    parts = ["<table>"]
    if header_rows > 0:
        parts.append("<thead>")
        for r in range(min(header_rows, len(grid))):
            cells = "".join(f"<th>{html.escape(c)}</th>" for c in grid[r])
            parts.append(f"<tr>{cells}</tr>")
        parts.append("</thead>")
    if header_rows < len(grid):
        parts.append("<tbody>")
        for r in range(header_rows, len(grid)):
            cells = "".join(f"<td>{html.escape(c)}</td>" for c in grid[r])
            parts.append(f"<tr>{cells}</tr>")
        parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def _canonicalize_gfm_tables_in_lines(lines: list[str]) -> list[str]:
    """Ensure every GFM pipe table block has valid separator and uniform column width."""
    out: list[str] = []
    i = 0
    n = len(lines)
    open_fence: tuple[str, int] | None = None
    while i < n:
        ln = lines[i]
        m_f = _FENCE_RE.match(ln)
        if m_f:
            marker, rest = m_f.group(1), m_f.group(2)
            if open_fence is None:
                open_fence = (marker[0], len(marker))
            elif marker[0] == open_fence[0] and len(marker) >= open_fence[1] and not rest.strip():
                open_fence = None
            out.append(ln)
            i += 1
            continue
        if open_fence is None and i + 1 < n and "|" in ln and "|" in lines[i + 1]:
            sep_raw = _split_gfm_row_cells(lines[i + 1])
            if sep_raw and all(_GFM_SEP_CELL_RE.match(c) for c in sep_raw):
                raw_rows: list[list[str]] = []
                hdr_cells = _split_gfm_row_cells(ln)
                raw_rows.append(hdr_cells)
                j = i + 2
                while j < n and lines[j].strip().startswith("|") and "|" in lines[j].strip()[1:]:
                    body_cells = _split_gfm_row_cells(lines[j])
                    raw_rows.append(body_cells)
                    j += 1
                n_cols = max(len(sep_raw), *(len(r) for r in raw_rows))
                norm_rows = [r + [""] * (n_cols - len(r)) for r in raw_rows]
                if out and out[-1].strip() != "":
                    out.append("")
                out.extend(_format_gfm_table(norm_rows, header_rows=1))
                if j < n and lines[j].strip() != "":
                    out.append("")
                i = j
                continue
        out.append(ln)
        i += 1
    return out


def canonicalize_markdown(
    raw_text: str,
    *,
    table_mode: MarkdownTableMode = "gfm",
) -> str:
    """Convert raw OCR Markdown/HTML into cleanly balanced, strictly-rectangular Markdown (`<1 ms`)."""
    txt = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")

    def _convert_html_table_fragment(fragment: str) -> str:
        parser = _HtmlTableParser()
        parser.feed(fragment)
        parser.close()
        grid, header_rows = parser.materialize()
        if not grid:
            return ""
        if table_mode == "html":
            return "\n\n" + _render_html_table_grid(grid, header_rows) + "\n\n"
        return "\n\n" + "\n".join(_format_gfm_table(grid, header_rows=header_rows)) + "\n\n"

    # Recover any truncated trailing `<table...>` missing its closing `</table>` before replacing closed tables
    low = txt.lower()
    last_open = low.rfind("<table")
    last_close = low.rfind("</table>")
    if last_open > last_close:
        txt = txt[:last_open] + _convert_html_table_fragment(txt[last_open:])

    txt = _HTML_TABLE_RE.sub(lambda m: _convert_html_table_fragment(m.group(0)), txt)

    # Fix missing space after ATX heading `#Heading` and close dangling fences/math outside code fences
    lines = txt.split("\n")
    norm_lines: list[str] = []
    open_fence: tuple[str, int] | None = None
    open_math = False
    for ln in lines:
        m_f = _FENCE_RE.match(ln)
        if m_f:
            marker, rest = m_f.group(1), m_f.group(2)
            if open_fence is None:
                open_fence = (marker[0], len(marker))
            elif marker[0] == open_fence[0] and len(marker) >= open_fence[1] and not rest.strip():
                open_fence = None
            norm_lines.append(ln.rstrip())
            continue
        if open_fence is None:
            if ln.strip() == "$$":
                open_math = not open_math
            ln = _HEADING_SPACE_RE.sub(r"\1 \2", ln)
        norm_lines.append(ln.rstrip())

    if open_fence is not None:
        norm_lines.append(open_fence[0] * open_fence[1])
    if open_math:
        norm_lines.append("$$")

    norm_lines = _canonicalize_gfm_tables_in_lines(norm_lines)

    # Collapse 3+ consecutive blank lines down to 1 blank line
    collapsed: list[str] = []
    blank_run = 0
    for ln in norm_lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1 and collapsed:
                collapsed.append("")
        else:
            blank_run = 0
            collapsed.append(ln)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return "\n".join(collapsed)


def _xy_cut_order(lines: Sequence[OCRLine], *, min_gutter: float = 0.03) -> list[OCRLine]:
    """Recursively order lines using X/Y projection gutter splits to handle multi-column pages."""
    items = list(lines)
    if len(items) <= 2:
        med_h = float(np.median([l.box.height for l in items if l.box.height > 0.0])) if items else 0.015
        return [ln for row in _group_visual_rows(items, med_h) for ln in row]

    # 1. Try strong full-height vertical gutter splitting left/right columns
    min_x = min(l.box.x0 for l in items)
    max_x = max(l.box.x1 for l in items)
    if max_x - min_x > 0.35:
        n_bins = 80
        occ = np.zeros(n_bins, dtype=np.int32)
        for l in items:
            b0 = max(0, min(n_bins - 1, int(l.box.x0 * n_bins)))
            b1 = max(b0 + 1, min(n_bins, math.ceil(l.box.x1 * n_bins)))
            occ[b0:b1] += 1
        # Search interior [20%, 80%] for widest zero-occupancy gutter
        lo = int(0.20 * n_bins)
        hi = int(0.80 * n_bins)
        best_start, best_len = -1, 0
        cur_start, cur_len = -1, 0
        for b in range(lo, hi):
            if occ[b] == 0:
                if cur_start < 0:
                    cur_start = b
                    cur_len = 1
                else:
                    cur_len += 1
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
            else:
                cur_start, cur_len = -1, 0
        if best_len / n_bins >= min_gutter:
            split_x = (best_start + 0.5 * best_len) / n_bins
            left = [l for l in items if l.box.x1 <= split_x + 0.005]
            right = [l for l in items if l.box.x0 >= split_x - 0.005]
            if len(left) >= 2 and len(right) >= 2 and len(left) + len(right) == len(items):
                return _xy_cut_order(left, min_gutter=min_gutter) + _xy_cut_order(right, min_gutter=min_gutter)

    # 2. Try horizontal whitespace split (e.g. full-width heading/footer separating multi-column regions)
    heights = [l.box.height for l in items if l.box.height > 0.0]
    med_h = float(np.median(heights)) if heights else 0.015
    by_y = sorted(items, key=lambda l: (l.box.y0, l.box.x0))
    running_y1 = by_y[0].box.y1
    best_gap = 0.0
    best_split = -1
    min_h_gap = max(0.014, med_h * 1.1)
    for k in range(1, len(by_y)):
        gap = by_y[k].box.y0 - running_y1
        if gap >= min_h_gap and gap > best_gap:
            best_gap = gap
            best_split = k
        if by_y[k].box.y1 > running_y1:
            running_y1 = by_y[k].box.y1
    if best_split > 0:
        return _xy_cut_order(by_y[:best_split], min_gutter=min_gutter) + _xy_cut_order(
            by_y[best_split:], min_gutter=min_gutter
        )

    # 3. Leaf region: group into visual rows so horizontal cells on the same row stay left-to-right
    return [ln for row in _group_visual_rows(items, med_h) for ln in row]


def _group_visual_rows(lines: Sequence[OCRLine], med_h: float) -> list[list[OCRLine]]:
    if not lines:
        return []
    tol = max(0.006, 0.55 * med_h)
    ordered = sorted(lines, key=lambda l: (0.5 * (l.box.y0 + l.box.y1), l.box.x0))
    rows: list[list[OCRLine]] = [[ordered[0]]]
    cur_cy = 0.5 * (ordered[0].box.y0 + ordered[0].box.y1)
    for ln in ordered[1:]:
        cy = 0.5 * (ln.box.y0 + ln.box.y1)
        if abs(cy - cur_cy) <= tol:
            rows[-1].append(ln)
            cur_cy = sum(0.5 * (x.box.y0 + x.box.y1) for x in rows[-1]) / len(rows[-1])
        else:
            rows[-1].sort(key=lambda x: x.box.x0)
            rows.append([ln])
            cur_cy = cy
    rows[-1].sort(key=lambda x: x.box.x0)
    return rows


def _extract_table_block(rows: Sequence[list[OCRLine]]) -> list[list[str]]:
    """Cluster horizontal `x`-intervals across multi-cell rows into rectangular table columns."""
    intervals: list[tuple[float, float]] = []
    for row in rows:
        for ln in row:
            intervals.append((ln.box.x0, ln.box.x1))
    if not intervals:
        return []
    intervals.sort()
    cols: list[list[float]] = []
    for x0, x1 in intervals:
        placed = False
        for c in cols:
            if not (x1 < c[0] - 0.008 or x0 > c[1] + 0.008):
                c[0] = min(c[0], x0)
                c[1] = max(c[1], x1)
                placed = True
                break
        if not placed:
            cols.append([x0, x1])
    # Re-sort merged column spans left-to-right
    cols.sort(key=lambda c: c[0])
    # Merge adjacent overlapping intervals after bounds growth
    merged_cols: list[list[float]] = []
    for c in cols:
        if merged_cols and c[0] <= merged_cols[-1][1] + 0.005:
            merged_cols[-1][1] = max(merged_cols[-1][1], c[1])
        else:
            merged_cols.append(c)
    if len(merged_cols) < 2:
        return []

    centers = [0.5 * (c[0] + c[1]) for c in merged_cols]
    grid: list[list[str]] = []
    for row in rows:
        cells = [""] * len(merged_cols)
        for ln in row:
            cx = 0.5 * (ln.box.x0 + ln.box.x1)
            best_col = min(range(len(centers)), key=lambda idx: abs(cx - centers[idx]))
            cells[best_col] = (cells[best_col] + " " + ln.text.strip()).strip()
        grid.append(cells)
    return grid


def synthesize_markdown_from_lines(lines: Sequence[OCRLine]) -> str:
    """Synthesize validly structured GFM Markdown from OCR line boxes in `<0.5 ms` (for ANE Apple Vision)."""
    clean_lines = [ln for ln in lines if ln.text.strip()]
    if not clean_lines:
        return ""

    heights = [ln.box.height for ln in clean_lines if ln.box.height > 0.0]
    med_h = float(np.median(heights)) if heights else 0.015
    body_left = float(np.percentile([ln.box.x0 for ln in clean_lines], 15))

    rows = _group_visual_rows(clean_lines, med_h)

    # Identify contiguous table bands (rows containing >= 2 horizontally separated cells)
    def _is_multi_cell(row: list[OCRLine]) -> bool:
        if len(row) < 2:
            return False
        sep = sum(1 for i in range(1, len(row)) if row[i].box.x0 > row[i - 1].box.x1 + 0.015)
        return sep >= 1

    def _looks_like_two_column_prose(grid: list[list[str]]) -> bool:
        if not grid or len(grid[0]) != 2:
            return False
        non_empty = [c.strip() for r in grid for c in r if c.strip()]
        if not non_empty:
            return True
        num_hits = sum(1 for v in non_empty if _NUMERIC_CELL_RE.match(v))
        if num_hits * 5 >= len(non_empty):  # >= 20% numeric/currency cells -> real key/value table
            return False
        bullet_hits = sum(1 for v in non_empty if _LIST_BULLET_RE.match(v))
        if bullet_hits >= 2:
            return True
        col0_lens = [len(r[0].strip()) for r in grid if r[0].strip()]
        col1_lens = [len(r[1].strip()) for r in grid if r[1].strip()]
        avg0 = (sum(col0_lens) / len(col0_lens)) if col0_lens else 0.0
        avg1 = (sum(col1_lens) / len(col1_lens)) if col1_lens else 0.0
        return avg0 >= 22.0 and avg1 >= 22.0

    def _try_extract_table(start_idx: int) -> tuple[list[list[str]], int] | None:
        if not _is_multi_cell(rows[start_idx]):
            return None
        j = start_idx + 1
        while j < n_rows and (
            _is_multi_cell(rows[j]) or (len(rows[j]) == 1 and j + 1 < n_rows and _is_multi_cell(rows[j + 1]))
        ):
            j += 1
        if j - start_idx < 3:
            return None
        grid = _extract_table_block(rows[start_idx:j])
        if not grid or len(grid[0]) < 2 or _looks_like_two_column_prose(grid):
            return None
        return grid, j

    blocks: list[list[str]] = []
    i = 0
    n_rows = len(rows)
    while i < n_rows:
        tbl = _try_extract_table(i)
        if tbl is not None:
            grid, j = tbl
            blocks.append(_format_gfm_table(grid, header_rows=1))
            i = j
            continue

        # Fallback: process non-table slice through XY-cut column ordering + heading/list/paragraph synthesis
        end_i = i + 1
        while end_i < n_rows and _try_extract_table(end_i) is None:
            end_i += 1
        prose_lines = [ln for r in rows[i:end_i] for ln in r]
        ordered = _xy_cut_order(prose_lines)

        para_buf: list[str] = []
        prev_line: OCRLine | None = None
        col_left = body_left

        def _flush_para() -> None:
            if not para_buf:
                return
            joined = ""
            for frag in para_buf:
                if joined.endswith("-") and frag and frag[0].islower():
                    joined = joined[:-1] + frag
                elif joined:
                    joined = joined + " " + frag
                else:
                    joined = frag
            blocks.append([joined])
            para_buf.clear()

        for ln in ordered:
            txt = " ".join(ln.text.split())
            if not txt:
                continue
            jumped_column = prev_line is not None and ln.box.y0 + 0.05 < prev_line.box.y0
            if prev_line is None or jumped_column:
                col_left = ln.box.x0
            else:
                col_left = min(col_left, ln.box.x0)

            gap_y = (ln.box.y0 - prev_line.box.y1) if (prev_line is not None and not jumped_column) else 0.02
            ratio_h = ln.box.height / max(1e-4, med_h)
            is_short = len(txt) <= 85
            is_centered = abs(0.5 * (ln.box.x0 + ln.box.x1) - 0.5) < 0.08 and ln.box.width < 0.65
            is_caps = txt.isupper() and sum(1 for c in txt if c.isalpha()) >= 4

            if _LIST_BULLET_RE.match(txt):
                _flush_para()
                norm_item = re.sub(r"^[•·*]\s+", "- ", txt)
                blocks.append([norm_item])
            else:
                can_be_heading = (
                    is_short
                    and not txt[0].islower()
                    and not txt.endswith(("-", ",", ";"))
                    and not (
                        prev_line is not None
                        and not jumped_column
                        and gap_y < med_h * 0.95
                        and abs(ln.box.x0 - col_left) < 0.03
                        and not prev_line.text.rstrip().endswith((".", "!", "?", ":"))
                    )
                )
                is_tall_heading = ratio_h >= 1.55
                is_sub_heading = ratio_h >= 1.32 and (gap_y >= med_h * 0.9 or prev_line is None or is_centered or jumped_column)
                is_caps_heading = is_centered and is_caps and len(txt) <= 65 and (gap_y >= med_h * 0.4 or prev_line is None)
                if can_be_heading and (is_tall_heading or is_sub_heading or is_caps_heading):
                    _flush_para()
                    level = "#" if ratio_h >= 1.85 else ("##" if (ratio_h >= 1.45 or is_centered) else "###")
                    blocks.append([f"{level} {txt}"])
                else:
                    # Start new paragraph on large vertical break or column jump or indented line
                    if prev_line is not None and (
                        gap_y > max(0.012, med_h * 1.15)
                        or ln.box.x0 > col_left + 0.045
                        or jumped_column
                    ):
                        _flush_para()
                    para_buf.append(txt)
            prev_line = ln
        _flush_para()
        i = end_i

    out_lines: list[str] = []
    for idx, blk in enumerate(blocks):
        if idx > 0:
            # Keep consecutive list items tight
            if len(blk) == 1 and len(blocks[idx - 1]) == 1 and _LIST_BULLET_RE.match(blk[0]) and _LIST_BULLET_RE.match(blocks[idx - 1][0]):
                pass
            else:
                out_lines.append("")
        out_lines.extend(blk)
    return "\n".join(out_lines)


def validate_markdown(text: str, *, allow_html_tables: bool = True) -> MarkdownValidation:
    """Verify structural Markdown validity (fences, math blocks, rectangular GFM tables, balanced HTML tables)."""
    errors: list[str] = []
    if "\x00" in text:
        errors.append("contains_nul_byte")

    lines = text.split("\n")
    open_fence: tuple[str, int] | None = None
    open_math = False
    for ln in lines:
        mf = _FENCE_RE.match(ln)
        if mf:
            marker, rest = mf.group(1), mf.group(2)
            if open_fence is None:
                open_fence = (marker[0], len(marker))
            elif marker[0] == open_fence[0] and len(marker) >= open_fence[1] and not rest.strip():
                open_fence = None
        elif open_fence is None and ln.strip() == "$$":
            open_math = not open_math

    fence_ok = open_fence is None
    math_ok = not open_math
    if not fence_ok:
        errors.append("unclosed_code_fence")
    if not math_ok:
        errors.append("unclosed_math_block")

    # Validate GFM pipe tables outside fences
    gfm_tables = 0
    gfm_ok = True
    i = 0
    val_fence: tuple[str, int] | None = None
    while i < len(lines):
        ln = lines[i]
        mf = _FENCE_RE.match(ln)
        if mf:
            marker, rest = mf.group(1), mf.group(2)
            if val_fence is None:
                val_fence = (marker[0], len(marker))
            elif marker[0] == val_fence[0] and len(marker) >= val_fence[1] and not rest.strip():
                val_fence = None
            i += 1
            continue
        if val_fence is None and i + 1 < len(lines) and "|" in ln and "|" in lines[i + 1]:
            sep_cells = _split_gfm_row_cells(lines[i + 1])
            if sep_cells and all(_GFM_SEP_CELL_RE.match(c) for c in sep_cells):
                gfm_tables += 1
                hdr_cells = _split_gfm_row_cells(ln)
                if len(hdr_cells) != len(sep_cells):
                    gfm_ok = False
                    errors.append(f"gfm_header_sep_col_mismatch_line_{i + 1}:{len(hdr_cells)}!={len(sep_cells)}")
                expected = len(sep_cells)
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    row_cells = _split_gfm_row_cells(lines[j])
                    if len(row_cells) != expected:
                        gfm_ok = False
                        errors.append(f"gfm_row_col_mismatch_line_{j + 1}:{len(row_cells)}!={expected}")
                    j += 1
                i = j
                continue
        i += 1

    # Validate HTML table tag balance
    html_tables = 0
    html_ok = True
    stack: list[str] = []
    for is_close, tag in _HTML_TAG_PAIR_RE.findall(text):
        tl = tag.lower()
        if not is_close:
            if tl == "table":
                html_tables += 1
            stack.append(tl)
        else:
            if not stack or stack[-1] != tl:
                html_ok = False
                errors.append(f"html_unbalanced_close_{tl}")
                break
            stack.pop()
    if stack:
        html_ok = False
        errors.append(f"html_unclosed_tags:{','.join(stack)}")
    if html_tables > 0 and not allow_html_tables:
        errors.append("unexpected_raw_html_table")

    return MarkdownValidation(
        valid=len(errors) == 0,
        table_count=gfm_tables + html_tables,
        gfm_table_count=gfm_tables,
        html_table_count=html_tables,
        fence_balanced=fence_ok,
        math_balanced=math_ok,
        html_tables_balanced=html_ok,
        gfm_tables_rectangular=gfm_ok,
        errors=tuple(errors),
    )


def build_gfm_table_lock_lark(num_cols: int, *, include_separator: bool = True) -> str:
    """Build a Lark CFG locking GFM separator + every subsequent table row to exactly `num_cols` cells."""
    if num_cols < 2 or num_cols > 32:
        raise ValueError(f"num_cols must be in [2, 32], got {num_cols}")
    sep_row = "| " + " | ".join(["---"] * num_cols) + " |\\n"
    cell_seq = " ".join('CELL "|"' for _ in range(num_cols))
    sep_rule = f'SEP: "{sep_row}"\n' if include_separator else 'SEP: ""\n'
    return (
        'start: SEP row+ TERM?\n'
        + sep_rule
        + f'row: "|" {cell_seq} "\\n"\n'
        + 'TERM: "\\n"\n'
        + 'CELL: /[^|\\r\\n]*/\n'
    )


class DynamicTableRowGrammar:
    """Opt-in JIT `llguidance` Lark table lock: locks column count `C` once a GFM header row completes.

    Also provides 0-forward-pass separator insertion via `matcher.compute_ff_tokens()`.
    """

    def __init__(self, llg_tokenizer: Any) -> None:
        import llguidance

        self._llg = llguidance
        self._llg_tok = llg_tokenizer
        self._matcher: Any | None = None
        self._line_buf: str = ""
        self._locked_cols: int | None = None
        self._open_fence: tuple[str, int] | None = None

    @property
    def is_active(self) -> bool:
        return self._matcher is not None and not self._matcher.is_stopped()

    @property
    def locked_cols(self) -> int | None:
        return self._locked_cols

    def reset(self) -> None:
        self._matcher = None
        self._line_buf = ""
        self._locked_cols = None
        self._open_fence = None

    def prepare_mask(self, *, vocab_size: int) -> np.ndarray | None:
        if self._matcher is None or self._matcher.is_stopped():
            return None
        raw_mask = self._matcher.compute_bitmask()
        words = np.frombuffer(raw_mask, dtype=np.uint32)
        bits = np.unpackbits(words.view(np.uint8), bitorder="little")
        n = int(vocab_size)
        if bits.size >= n:
            return bits[:n].astype(bool)
        padded = np.zeros(n, dtype=bool)
        padded[: bits.size] = bits.astype(bool)
        return padded

    def advance_text_and_maybe_lock(self, delta_text: str, tok_id: int) -> list[int]:
        """Track newly emitted token/text, activate C-column Lark lock after a GFM header line, and return any FF tokens."""
        ff_tokens: list[int] = []
        if self._matcher is not None:
            self._matcher.consume_token(int(tok_id))
            if not self._matcher.is_stopped():
                ff = [int(t) for t in self._matcher.compute_ff_tokens()]
                if ff:
                    self._matcher.consume_tokens(ff)
                    ff_tokens.extend(ff)
            if self._matcher.is_stopped():
                self._matcher = None
                self._locked_cols = None
                self._line_buf = ""
            return ff_tokens

        for ch in delta_text:
            if ch == "\n":
                line = self._line_buf.strip()
                self._line_buf = ""
                m_f = _FENCE_RE.match(line)
                if m_f:
                    marker, rest = m_f.group(1), m_f.group(2)
                    if self._open_fence is None:
                        self._open_fence = (marker[0], len(marker))
                    elif marker[0] == self._open_fence[0] and len(marker) >= self._open_fence[1] and not rest.strip():
                        self._open_fence = None
                    continue
                if self._open_fence is None and line.startswith("|") and _ends_with_unescaped_pipe(line) and len(line) >= 5:
                    cells = _split_gfm_row_cells(line)
                    # Trigger only on non-separator header rows with 2..24 columns
                    if 2 <= len(cells) <= 24 and not all(_GFM_SEP_CELL_RE.match(c) for c in cells):
                        self._activate_lock(len(cells))
                        if self._matcher is not None:
                            ff = [int(t) for t in self._matcher.compute_ff_tokens()]
                            if ff:
                                self._matcher.consume_tokens(ff)
                                ff_tokens.extend(ff)
            else:
                self._line_buf += ch
        return ff_tokens

    def _activate_lock(self, num_cols: int) -> None:
        lark = build_gfm_table_lock_lark(num_cols, include_separator=True)
        grammar = self._llg.grammar_from("lark", lark)
        self._matcher = self._llg.LLMatcher(self._llg_tok, grammar)
        self._locked_cols = int(num_cols)

    def clone(self) -> DynamicTableRowGrammar:
        """Return an independent snapshot of buffered line state and any active `LLMatcher`."""
        cloned = DynamicTableRowGrammar(self._llg_tok)
        cloned._matcher = self._matcher.deep_copy() if self._matcher is not None else None
        cloned._line_buf = self._line_buf
        cloned._locked_cols = self._locked_cols
        cloned._open_fence = self._open_fence
        return cloned
