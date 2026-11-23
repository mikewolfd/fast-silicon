"""Digit-stem filtered ANE-vs-VLM missing-number detection and targeted crop box localization."""

from __future__ import annotations

from collections import Counter
import re
from typing import Sequence

from .preprocess import rotate_box_normalized
from .types import OCRLine

_NUMBER_TOKEN_RE = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?%?")
_NON_DIGIT_RE = re.compile(r"\D+")
_STRIP_NUM_PUNCT_RE = re.compile(r"[,\$\s]+")


def extract_normalized_numbers(text: str) -> list[str]:
    """Extract normalized numeric/currency/percent tokens (`$1,250.00` -> `1250.00`)."""
    out: list[str] = []
    for match in _NUMBER_TOKEN_RE.findall(text):
        norm = match.replace(",", "").replace("$", "").strip()
        if norm and any("0" <= ch <= "9" for ch in norm):
            out.append(norm)
    return out


def _digit_stem(tok: str) -> tuple[str, str]:
    clean = tok.lstrip("-").rstrip("%")
    digits = _NON_DIGIT_RE.sub("", clean)
    return clean, digits


def _text_numeric_signature(text: str) -> tuple[Counter[str], str, str]:
    counts = Counter(extract_normalized_numbers(text))
    flat = _STRIP_NUM_PUNCT_RE.sub("", text)
    digits_flat = _NON_DIGIT_RE.sub("", text)
    return counts, flat, digits_flat


def _is_single_digit_substitution(
    clean: str,
    digits: str,
    cand_stems: Sequence[tuple[str, str]],
) -> bool:
    """Return True if a candidate token of length >= 3 digits differs by at most 1 char."""
    if len(digits) < 3:
        return False
    for c_clean, c_digits in cand_stems:
        if len(c_digits) != len(digits) or len(c_clean) != len(clean):
            continue
        if sum(a != b for a, b in zip(clean, c_clean)) <= 1:
            return True
    return False


def find_filtered_missing_numbers(vision_text: str, candidate_text: str) -> list[str]:
    """Return multi-digit ANE numeric tokens missing from `candidate_text` after filtering formatting noise.

    Ignores 1-digit tokens, comma-vs-dot decimal separator mismatches (`2,016` vs `2.016`),
    substrings embedded in unspaced digit runs (`19.29` matching `1929`), and likely ANE
    single-digit substitution hallucinations where a candidate number of the same normalized
    length (`>= 3` digits) matches at Hamming distance `<= 1` (`445` vs `448`, `310` vs `314`).
    """
    if not vision_text:
        return []
    vis_counts = Counter(extract_normalized_numbers(vision_text))
    cand_counts, cand_flat, cand_digits_flat = _text_numeric_signature(candidate_text)
    cand_stems = [_digit_stem(c) for c in cand_counts]
    kept: list[str] = []
    for tok in (vis_counts - cand_counts).elements():
        clean, digits = _digit_stem(tok)
        if len(digits) < 2:
            continue
        if clean in cand_flat or digits in cand_digits_flat:
            continue
        if _is_single_digit_substitution(clean, digits, cand_stems):
            continue
        kept.append(tok)
    return kept


def find_recovered_numbers(missing_numbers: Sequence[str], zoom_text: str) -> list[str]:
    """Return the subset of `missing_numbers` confirmed present in a targeted zoom pass."""
    if not missing_numbers or not zoom_text:
        return []
    zoom_counts, zoom_flat, zoom_digits_flat = _text_numeric_signature(zoom_text)
    recovered: list[str] = []
    for tok in missing_numbers:
        if zoom_counts.get(tok, 0) > 0:
            zoom_counts[tok] -= 1
            recovered.append(tok)
            continue
        clean, digits = _digit_stem(tok)
        if len(digits) >= 2 and (clean in zoom_flat or digits in zoom_digits_flat):
            recovered.append(tok)
    return recovered


_ALIGN_SPACE_RE = re.compile(r"\s+")
_ALIGN_COMPACT_RE = re.compile(r"[^0-9a-zA-Z]+")
_ALIGN_TOKEN_RE = re.compile(r"[0-9a-zA-Z]{2,}")


def _interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, max(a0, b0) - min(a1, b1))


def _union_norm_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _norm_box_to_px(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: float,
) -> tuple[int, int, int, int] | None:
    left = max(0, int((box[0] - pad) * width))
    top = max(0, int((box[1] - pad) * height))
    right = min(width, int((box[2] + pad) * width))
    bottom = min(height, int((box[3] + pad) * height))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def compute_missing_numeric_crop_boxes_px(
    lines: Sequence[OCRLine],
    missing_numbers: Sequence[str],
    image_size: tuple[int, int],
    *,
    rotation_cw: int = 0,
    pad_frac: float = 0.04,
    max_crops: int = 2,
    max_union_area_frac: float = 0.28,
    max_union_height_frac: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    """Compute tight padded `(left, top, right, bottom)` pixel crop(s) around ANE missing numbers.

    When matched lines stay compact, returns a single union crop. When isolated outlier lines
    would inflate the union across distant bands, prunes lone low-confidence outlier clusters and
    splits into ranked tight band crops up to `max_crops`.
    """
    if not lines or not missing_numbers:
        return []
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return []
    miss_set = set(missing_numbers)
    rot = int(rotation_cw) % 360
    pad = max(0.0, float(pad_frac))

    items: list[tuple[tuple[float, float, float, float], float, frozenset[str]]] = []
    for ln in lines:
        hits = frozenset(extract_normalized_numbers(ln.text)) & miss_set
        if not hits:
            continue
        raw_box = (ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1)
        rot_box = rotate_box_normalized(raw_box, rot) if rot != 0 else raw_box
        items.append((rot_box, float(getattr(ln, "confidence", 1.0)), hits))
    if not items:
        return []

    full_union = _union_norm_boxes([it[0] for it in items])
    full_area = max(0.0, full_union[2] - full_union[0]) * max(0.0, full_union[3] - full_union[1])
    full_height = max(0.0, full_union[3] - full_union[1])
    if len(items) == 1 or (
        full_area <= float(max_union_area_frac) and full_height <= float(max_union_height_frac)
    ):
        px = _norm_box_to_px(full_union, w, h, pad)
        return [px] if px is not None else []

    # Connected components by local vertical/horizontal proximity avoid merging distant bands.
    n = len(items)
    visited = [False] * n
    clusters: list[list[tuple[tuple[float, float, float, float], float, frozenset[str]]]] = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        comp: list[tuple[tuple[float, float, float, float], float, frozenset[str]]] = []
        while stack:
            idx = stack.pop()
            comp.append(items[idx])
            b_i = items[idx][0]
            for j in range(n):
                if visited[j]:
                    continue
                b_j = items[j][0]
                if (
                    _interval_gap(b_i[1], b_i[3], b_j[1], b_j[3]) <= 0.14
                    and _interval_gap(b_i[0], b_i[2], b_j[0], b_j[2]) <= 0.35
                ):
                    visited[j] = True
                    stack.append(j)
        clusters.append(comp)

    # Drop isolated lone low-confidence outlier clusters if stronger clusters remain.
    strong = [c for c in clusters if len(c) >= 2 or max(it[1] for it in c) >= 0.5]
    active = strong if strong else clusters

    rem_boxes = [it[0] for c in active for it in c]
    rem_union = _union_norm_boxes(rem_boxes)
    rem_area = max(0.0, rem_union[2] - rem_union[0]) * max(0.0, rem_union[3] - rem_union[1])
    rem_height = max(0.0, rem_union[3] - rem_union[1])
    if len(active) <= 1 or (
        rem_area <= float(max_union_area_frac) and rem_height <= float(max_union_height_frac)
    ):
        px = _norm_box_to_px(rem_union, w, h, pad)
        return [px] if px is not None else []

    ranked: list[tuple[tuple[int, int, float, float], tuple[int, int, int, int]]] = []
    for comp in active:
        comp_union = _union_norm_boxes([it[0] for it in comp])
        px = _norm_box_to_px(comp_union, w, h, pad)
        if px is None:
            continue
        uniq_hits = len(frozenset().union(*(it[2] for it in comp)))
        sum_conf = sum(it[1] for it in comp)
        comp_area = max(0.0, comp_union[2] - comp_union[0]) * max(0.0, comp_union[3] - comp_union[1])
        ranked.append(((uniq_hits, len(comp), sum_conf, -comp_area), px))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [px for _, px in ranked[: max(1, int(max_crops))]]


def compute_missing_numeric_crop_box_px(
    lines: Sequence[OCRLine],
    missing_numbers: Sequence[str],
    image_size: tuple[int, int],
    *,
    rotation_cw: int = 0,
    pad_frac: float = 0.04,
) -> tuple[int, int, int, int] | None:
    """Compute a padded `(left, top, right, bottom)` pixel crop covering ANE lines containing `missing_numbers`."""
    boxes = compute_missing_numeric_crop_boxes_px(
        lines,
        missing_numbers,
        image_size,
        rotation_cw=rotation_cw,
        pad_frac=pad_frac,
        max_crops=1,
    )
    return boxes[0] if boxes else None


def _parse_json_for_alignment(raw_text: str) -> object | None:
    import json

    txt = raw_text.strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = txt.find(open_ch)
        end = txt.rfind(close_ch)
        if start >= 0 and end > start:
            try:
                return json.loads(txt[start : end + 1])
            except Exception:
                continue
    return None


def _flatten_json_leaves(
    obj: object,
    path: str = "",
    array_group: str = "",
) -> list[tuple[int, str, str, str, bool]]:
    leaves: list[tuple[int, str, str, str, bool]] = []

    def _walk(node: object, cur_path: str, cur_group: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                key_str = str(k)
                nxt = f"{cur_path}.{key_str}" if cur_path else key_str
                _walk(v, nxt, cur_group)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                nxt = f"{cur_path}[{idx}]"
                _walk(item, nxt, nxt)
        elif node is None or isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            if cur_path:
                leaves.append((len(leaves), cur_path, cur_group, str(node), True))
        else:
            val_s = str(node).strip()
            if val_s and cur_path:
                num_toks = extract_normalized_numbers(val_s)
                compact = _ALIGN_COMPACT_RE.sub("", val_s)
                is_num = bool(
                    num_toks
                    and len(num_toks) == 1
                    and _digit_stem(num_toks[0])[1] == compact
                )
                leaves.append((len(leaves), cur_path, cur_group, val_s, is_num))

    _walk(obj, path, array_group)
    return leaves


def align_json_values_to_lines(
    raw_text: str,
    lines: Sequence[OCRLine],
    *,
    image_size: tuple[int, int] | None = None,
    rotation_cw: int = 0,
) -> dict[str, list[float]]:
    """Align extracted leaf JSON values back to `vision_res.lines` normalized `[x0, y0, x1, y1]` boxes.

    Zero-VLM CPU alignment: matches distinctive string leaves first to anchor enclosing array rows,
    then resolves numeric and repeated short tokens using row proximity and box orientation.
    """
    _ = image_size
    if not raw_text or not lines:
        return {}
    parsed = _parse_json_for_alignment(raw_text)
    if parsed is None:
        return {}
    leaves = _flatten_json_leaves(parsed)
    if not leaves:
        return {}

    rot = int(rotation_cw) % 360
    line_entries: list[
        tuple[int, list[float], float, str, str, frozenset[str], set[str], frozenset[str]]
    ] = []
    for idx, ln in enumerate(lines):
        raw_box = (ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1)
        rb = rotate_box_normalized(raw_box, rot) if rot != 0 else raw_box
        box_list = [round(float(rb[0]), 6), round(float(rb[1]), 6), round(float(rb[2]), 6), round(float(rb[3]), 6)]
        y_center = (box_list[1] + box_list[3]) * 0.5
        ln_norm = _ALIGN_SPACE_RE.sub(" ", ln.text.strip().lower())
        ln_compact = _ALIGN_COMPACT_RE.sub("", ln_norm)
        if not ln_norm and not ln_compact:
            continue
        ln_tokens = frozenset(_ALIGN_TOKEN_RE.findall(ln_norm))
        ln_nums = set(extract_normalized_numbers(ln.text))
        ln_digits = frozenset(_digit_stem(n)[1] for n in ln_nums if _digit_stem(n)[1])
        line_entries.append((idx, box_list, y_center, ln_norm, ln_compact, ln_tokens, ln_nums, ln_digits))
    if not line_entries:
        return {}

    group_anchors: dict[str, list[float]] = {}
    used_leaf_keys: set[tuple[int, str]] = set()
    matched: dict[int, tuple[str, list[float]]] = {}

    def _best_line_for_leaf(
        val_str: str,
        is_num: bool,
        y_anchor: float | None,
    ) -> tuple[int, float, list[float], float] | None:
        val_norm = _ALIGN_SPACE_RE.sub(" ", val_str.strip().lower())
        val_compact = _ALIGN_COMPACT_RE.sub("", val_norm)
        val_tokens = frozenset(_ALIGN_TOKEN_RE.findall(val_norm))
        val_nums = extract_normalized_numbers(val_str) if is_num else []
        val_num_primary = val_nums[0] if len(val_nums) == 1 else ""
        val_digits = _digit_stem(val_num_primary)[1] if val_num_primary else ""

        best_idx = -1
        best_score = 0.0
        best_base = 0.0
        best_box: list[float] = []
        best_yc = 0.0

        for idx, box_list, yc, ln_norm, ln_compact, ln_tokens, ln_nums, ln_digits in line_entries:
            base = 0.0
            if val_norm == ln_norm or (val_compact and val_compact == ln_compact):
                base = 1.0
            elif is_num and val_num_primary:
                if val_num_primary in ln_nums:
                    extra_nums = max(0, len(ln_nums) - 1)
                    base = 0.92 - min(0.12, 0.04 * extra_nums)
                elif len(val_digits) >= 3 and val_digits in ln_digits:
                    base = 0.80
            else:
                if val_compact and len(val_compact) >= 2 and (
                    val_norm in ln_norm or val_compact in ln_compact
                ):
                    ratio = min(1.0, len(val_compact) / max(1, len(ln_compact)))
                    base = 0.78 + 0.15 * ratio
                elif len(ln_compact) >= 3 and ln_compact in val_compact:
                    base = 0.65 * (len(ln_compact) / max(1, len(val_compact)))
                elif val_tokens and ln_tokens:
                    overlap = len(val_tokens & ln_tokens)
                    recall = overlap / len(val_tokens)
                    if recall >= 0.5:
                        base = 0.45 + 0.30 * recall

            if base < 0.45:
                continue

            score = base
            if y_anchor is not None:
                dy = abs(yc - y_anchor)
                if dy <= 0.04:
                    score += 0.25
                else:
                    score -= min(0.45, dy * 1.5)
            key = (idx, val_compact or val_norm)
            if key not in used_leaf_keys:
                score += 0.03

            if score > best_score:
                best_score = score
                best_base = base
                best_idx = idx
                best_box = box_list
                best_yc = yc

        if best_idx < 0:
            return None
        return best_idx, best_base, best_box, best_yc

    # Pass 1: distinctive non-numeric leaves establish array-row vertical anchors.
    pass1_leaves = sorted(
        (lf for lf in leaves if not lf[4]),
        key=lambda lf: (-len(_ALIGN_COMPACT_RE.sub("", lf[3])), lf[0]),
    )
    for order_idx, path, group, val_str, is_num in pass1_leaves:
        anchor = (
            sum(group_anchors[group]) / len(group_anchors[group])
            if group and group in group_anchors
            else None
        )
        res = _best_line_for_leaf(val_str, is_num, anchor)
        if res is None:
            continue
        line_idx, base, box_list, yc = res
        matched[order_idx] = (path, box_list)
        key = (line_idx, _ALIGN_COMPACT_RE.sub("", val_str.strip().lower()) or val_str.strip().lower())
        used_leaf_keys.add(key)
        if group and base >= 0.65:
            group_anchors.setdefault(group, []).append(yc)

    # Pass 2: numeric leaves (and any unaligned leaves) match with row anchors applied.
    for order_idx, path, group, val_str, is_num in leaves:
        if order_idx in matched:
            continue
        anchor = (
            sum(group_anchors[group]) / len(group_anchors[group])
            if group and group in group_anchors
            else None
        )
        res = _best_line_for_leaf(val_str, is_num, anchor)
        if res is None:
            continue
        line_idx, base, box_list, yc = res
        matched[order_idx] = (path, box_list)
        key = (line_idx, _ALIGN_COMPACT_RE.sub("", val_str.strip().lower()) or val_str.strip().lower())
        used_leaf_keys.add(key)
        if group and base >= 0.70:
            group_anchors.setdefault(group, []).append(yc)

    return {matched[idx][0]: matched[idx][1] for idx in sorted(matched)}
