"""Category 4 deterministic scorers: Hungarian grounded_field_f1, BFCL v3 AST + abstention, JSONSchemaBench compiler."""

from __future__ import annotations

import ast
from collections import Counter
import json
import math
import re
import time
from typing import Any, Mapping, Sequence
import unicodedata

import numpy as np

from .chat_reasoning import repeat_4gram_rate

# ---------------------------------------------------------------------------
# Pure-NumPy Exact Rectangular Hungarian Linear Sum Assignment (0 SciPy)
# ---------------------------------------------------------------------------


def hungarian_min_cost(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact Kuhn-Munkres / Jonker-Volgenant minimum-cost assignment for rectangular ``[R, C]`` matrices.

    Pure NumPy / Python implementation (`O(min(R, C)^2 * max(R, C))`), zero SciPy dependency.
    """
    c = np.asarray(cost_matrix, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"Expected 2D cost matrix, got shape {c.shape}")
    if c.size > 0 and not bool(np.all(np.isfinite(c))):
        raise ValueError("cost_matrix must contain only finite values")
    n_rows, n_cols = int(c.shape[0]), int(c.shape[1])
    if n_rows == 0 or n_cols == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    transposed = False
    if n_rows > n_cols:
        c = c.T
        n_rows, n_cols = n_cols, n_rows
        transposed = True

    # 1-indexed Python lists for scalar inner loop (faster than NumPy scalar indexing, 0 guard warnings).
    c_rows: list[list[float]] = c.tolist()
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row_cost = c_rows[i0 - 1]
            u_i0 = u[i0]
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = row_cost[j - 1] - u_i0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col = np.full(n_rows, -1, dtype=np.int64)
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            row_to_col[p[j] - 1] = j - 1

    rows = np.arange(n_rows, dtype=np.int64)
    cols = row_to_col
    if transposed:
        order = np.argsort(cols, kind="mergesort")
        return cols[order], rows[order]
    return rows, cols


def linear_sum_assignment_max(weight_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact maximum-weight bipartite assignment for rectangular ``[R, C]`` similarity matrices."""
    w = np.asarray(weight_matrix, dtype=np.float64)
    if w.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    max_w = float(np.max(w))
    return hungarian_min_cost(max_w - w)


# ---------------------------------------------------------------------------
# Schema-Validated Recursive `grounded_field_f1`
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})(?:[T\s].*)?$")
_DMY_DATE_RE = re.compile(r"^(\d{1,2})[\./-](\d{1,2})[\./-](\d{4})$")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _normalize_date_str(val: str) -> str | None:
    s = unicodedata.normalize("NFKC", val).strip()
    m = _ISO_DATE_RE.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m2 = _DMY_DATE_RE.match(s)
    if m2:
        a, b, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if 1 <= b <= 12 and 1 <= a <= 31:
            return f"{y:04d}-{b:02d}-{a:02d}"
    return None


def _norm_text(val: str) -> str:
    s = unicodedata.normalize("NFKC", val)
    return re.sub(r"\s+", " ", s).strip().lower()


def _resolve_subschema(sch: Any, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(sch, Mapping):
        return {}
    cur: Mapping[str, Any] = sch
    seen_refs: set[str] = set()
    while "$ref" in cur and isinstance(cur["$ref"], str):
        ref = cur["$ref"]
        if ref in seen_refs:
            break
        seen_refs.add(ref)
        if ref.startswith("#/$defs/"):
            target = root.get("$defs", {}).get(ref[len("#/$defs/") :])
        elif ref.startswith("#/definitions/"):
            target = root.get("definitions", {}).get(ref[len("#/definitions/") :])
        else:
            break
        if not isinstance(target, Mapping):
            break
        cur = target
    for union_k in ("anyOf", "oneOf"):
        branches = cur.get(union_k)
        if isinstance(branches, list) and branches:
            non_null = [
                _resolve_subschema(b, root)
                for b in branches
                if isinstance(b, Mapping) and b.get("type") != "null"
            ]
            if len(non_null) == 1:
                merged = dict(cur)
                merged.pop(union_k, None)
                merged.update(non_null[0])
                return merged
    return cur


def _score_string_leaf(pred: str, gold: str, subschema: Mapping[str, Any]) -> float:
    is_date_hint = str(subschema.get("format") or "").lower() in ("date", "date-time")
    g_date = _normalize_date_str(gold)
    p_date = _normalize_date_str(pred)
    if is_date_hint or (g_date is not None and p_date is not None):
        if g_date is not None and p_date is not None:
            return 1.0 if p_date == g_date else 0.0

    pn = _norm_text(pred)
    gn = _norm_text(gold)
    if pn == gn:
        return 1.0
    g_toks = _TOKEN_RE.findall(gn)
    if len(g_toks) >= 3:
        p_toks = _TOKEN_RE.findall(pn)
        if not p_toks:
            return 0.0
        cg = Counter(g_toks)
        cp = Counter(p_toks)
        inter = sum((cg & cp).values())
        return (2.0 * float(inter)) / float(len(p_toks) + len(g_toks))
    return 0.0


def _score_node(pred: Any, gold: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> float:
    sub = _resolve_subschema(schema, root)

    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0

    # Enum / boolean exact checks.
    if "enum" in sub:
        return 1.0 if pred == gold else 0.0
    if isinstance(gold, bool) or isinstance(pred, bool):
        return 1.0 if (isinstance(pred, bool) and isinstance(gold, bool) and pred == gold) else 0.0

    # Object recursive scoring (penalizing hallucinated non-null keys on gold null fields!).
    if isinstance(gold, Mapping) and isinstance(pred, Mapping):
        props = sub.get("properties") if isinstance(sub.get("properties"), Mapping) else {}
        add_props = sub.get("additionalProperties")
        default_prop_sch = add_props if isinstance(add_props, Mapping) else {}
        active_keys = sorted(
            {str(k) for k, v in gold.items() if v is not None}
            | {str(k) for k, v in pred.items() if v is not None}
        )
        if not active_keys:
            return 1.0
        total = 0.0
        for k in active_keys:
            gv = gold.get(k)
            pv = pred.get(k)
            if gv is None or pv is None:
                total += 0.0
            else:
                child_sch = props.get(k, default_prop_sch)
                total += _score_node(pv, gv, child_sch, root)
        return total / float(len(active_keys))

    # Array recursive scoring (multiset F1 for primitives; exact Hungarian F1 for objects).
    if isinstance(gold, list) and isinstance(pred, list):
        if len(gold) == 0 and len(pred) == 0:
            return 1.0
        if len(gold) == 0 or len(pred) == 0:
            return 0.0
        item_sch = sub.get("items") if isinstance(sub.get("items"), Mapping) else {}
        is_obj_array = any(isinstance(x, Mapping) for x in gold) or any(isinstance(x, Mapping) for x in pred)
        if is_obj_array or (len(pred) <= 64 and len(gold) <= 64):
            sim = np.zeros((len(pred), len(gold)), dtype=np.float64)
            for r, pe in enumerate(pred):
                for c, ge in enumerate(gold):
                    sim[r, c] = _score_node(pe, ge, item_sch, root)
            r_ind, c_ind = linear_sum_assignment_max(sim)
            matched_weight = float(np.sum(sim[r_ind, c_ind]))
            return (2.0 * matched_weight) / float(len(pred) + len(gold))
        # Fast multiset F1 for large primitive/non-object arrays (>64 items), safe for nested lists/dicts.
        def _hashable_key(val: Any) -> Any:
            if isinstance(val, str):
                return ("s", _norm_text(val))
            if isinstance(val, bool):
                return ("b", val)
            if isinstance(val, (int, float)):
                return ("n", float(val))
            if val is None:
                return ("null", None)
            return ("j", json.dumps(val, sort_keys=True, ensure_ascii=False))

        cp = Counter(_hashable_key(x) for x in pred)
        cg = Counter(_hashable_key(x) for x in gold)
        inter = sum((cp & cg).values())
        return (2.0 * float(inter)) / float(len(pred) + len(gold))

    # Integer vs floating number leaves.
    s_type = sub.get("type")
    if s_type == "integer" or (isinstance(gold, int) and isinstance(pred, int)):
        if isinstance(pred, int) and isinstance(gold, int):
            return 1.0 if pred == gold else 0.0
        if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
            return 1.0 if abs(float(pred) - float(gold)) <= 1e-9 else 0.0
        return 0.0

    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        pf, gf = float(pred), float(gold)
        if not (math.isfinite(pf) and math.isfinite(gf)):
            return 0.0
        return 1.0 if abs(pf - gf) <= 1e-5 * max(1.0, abs(gf)) else 0.0

    if isinstance(gold, str) and isinstance(pred, str):
        return _score_string_leaf(pred, gold, sub)

    return 1.0 if pred == gold else 0.0


def _json_equal(lhs: Any, rhs: Any) -> bool:
    """JSON value equality distinguishing booleans from numeric 0/1."""
    if isinstance(lhs, bool) != isinstance(rhs, bool):
        return False
    if isinstance(lhs, list) and isinstance(rhs, list):
        return len(lhs) == len(rhs) and all(_json_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        return set(lhs.keys()) == set(rhs.keys()) and all(_json_equal(lhs[k], rhs[k]) for k in lhs)
    return bool(lhs == rhs)


def _matches_json_type(val: Any, type_name: str) -> bool:
    if type_name == "null":
        return val is None
    if type_name == "boolean":
        return isinstance(val, bool)
    if type_name == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if type_name == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))
    if type_name == "string":
        return isinstance(val, str)
    if type_name == "array":
        return isinstance(val, list)
    if type_name == "object":
        return isinstance(val, Mapping)
    return False


def _resolve_json_pointer(ref: str, root: Mapping[str, Any]) -> tuple[bool, Any]:
    if ref == "#":
        return True, root
    if not ref.startswith("#/"):
        return False, None
    cur: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def _validate_jsonschema_fallback(
    val: Any,
    node: Any,
    root: Mapping[str, Any],
    *,
    _depth: int = 0,
) -> bool:
    """Pure-Python Draft-7 / Draft-2020-12 JSON Schema validator when optional `jsonschema` is absent."""
    if _depth > 128:
        return False
    if isinstance(node, bool):
        return node
    if not isinstance(node, Mapping):
        return True

    if "$ref" in node and isinstance(node["$ref"], str):
        ok_ref, target = _resolve_json_pointer(node["$ref"], root)
        if not ok_ref or not _validate_jsonschema_fallback(val, target, root, _depth=_depth + 1):
            return False
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        if siblings:
            return _validate_jsonschema_fallback(val, siblings, root, _depth=_depth + 1)
        return True

    if "allOf" in node:
        subs = node["allOf"]
        if not isinstance(subs, list) or not all(
            _validate_jsonschema_fallback(val, s, root, _depth=_depth + 1) for s in subs
        ):
            return False
    if "anyOf" in node:
        subs = node["anyOf"]
        if not isinstance(subs, list) or not any(
            _validate_jsonschema_fallback(val, s, root, _depth=_depth + 1) for s in subs
        ):
            return False
    if "oneOf" in node:
        subs = node["oneOf"]
        if not isinstance(subs, list):
            return False
        matches = sum(1 for s in subs if _validate_jsonschema_fallback(val, s, root, _depth=_depth + 1))
        if matches != 1:
            return False
    if "not" in node:
        if _validate_jsonschema_fallback(val, node["not"], root, _depth=_depth + 1):
            return False

    if "const" in node and not _json_equal(val, node["const"]):
        return False
    if "enum" in node:
        opts = node["enum"]
        if not isinstance(opts, list) or not any(_json_equal(val, cand) for cand in opts):
            return False

    if "type" in node:
        t_spec = node["type"]
        if isinstance(t_spec, str):
            if not _matches_json_type(val, t_spec):
                return False
        elif isinstance(t_spec, list):
            if not any(_matches_json_type(val, str(t)) for t in t_spec):
                return False
        else:
            return False

    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if not math.isfinite(float(val)):
            return False
        if "minimum" in node and val < node["minimum"]:
            return False
        if "maximum" in node and val > node["maximum"]:
            return False
        if isinstance(node.get("exclusiveMinimum"), (int, float)) and val <= node["exclusiveMinimum"]:
            return False
        if isinstance(node.get("exclusiveMaximum"), (int, float)) and val >= node["exclusiveMaximum"]:
            return False

    if isinstance(val, str):
        if "minLength" in node and len(val) < int(node["minLength"]):
            return False
        if "maxLength" in node and len(val) > int(node["maxLength"]):
            return False
        if "pattern" in node and not re.search(str(node["pattern"]), val):
            return False

    if isinstance(val, list):
        if "minItems" in node and len(val) < int(node["minItems"]):
            return False
        if "maxItems" in node and len(val) > int(node["maxItems"]):
            return False
        if "items" in node:
            it_spec = node["items"]
            if isinstance(it_spec, (Mapping, bool)):
                if not all(_validate_jsonschema_fallback(el, it_spec, root, _depth=_depth + 1) for el in val):
                    return False
            elif isinstance(it_spec, list):
                for el, sub_s in zip(val, it_spec):
                    if not _validate_jsonschema_fallback(el, sub_s, root, _depth=_depth + 1):
                        return False

    if isinstance(val, Mapping):
        req = node.get("required")
        if isinstance(req, list):
            for rk in req:
                if rk not in val:
                    return False
        if "minProperties" in node and len(val) < int(node["minProperties"]):
            return False
        if "maxProperties" in node and len(val) > int(node["maxProperties"]):
            return False
        props = node.get("properties")
        prop_map = props if isinstance(props, Mapping) else {}
        has_ap = "additionalProperties" in node
        ap_spec = node.get("additionalProperties", True)
        for k, v in val.items():
            if k in prop_map:
                if not _validate_jsonschema_fallback(v, prop_map[k], root, _depth=_depth + 1):
                    return False
            elif has_ap:
                if ap_spec is False:
                    return False
                if isinstance(ap_spec, (Mapping, bool)) and not _validate_jsonschema_fallback(
                    v, ap_spec, root, _depth=_depth + 1
                ):
                    return False
    return True


def validate_against_jsonschema(obj: Any, schema_obj: Mapping[str, Any]) -> bool:
    """Validate ``obj`` against ``schema_obj`` via ``jsonschema`` or pure-Python Draft-7/2020-12 fallback."""
    try:
        import jsonschema as _js_mod
    except ImportError:
        _js_mod = None

    sch = dict(schema_obj)
    if _js_mod is not None:
        try:
            _js_mod.validate(instance=obj, schema=sch)
            return True
        except Exception:
            return False
    try:
        return _validate_jsonschema_fallback(obj, sch, sch)
    except Exception:
        return False


def grounded_field_f1(
    pred_obj: Any,
    gold_obj: Any,
    schema_obj: Mapping[str, Any] | None,
    *,
    require_schema_valid: bool = True,
) -> float:
    """Schema-validated recursive Grounded Field-F1 with Hungarian object-array assignment.

    Returns ``0.0`` immediately if ``require_schema_valid`` is True and ``pred_obj`` fails ``jsonschema.validate``.
    """
    sch = dict(schema_obj) if isinstance(schema_obj, Mapping) else {}
    if require_schema_valid and sch:
        if not validate_against_jsonschema(pred_obj, sch):
            return 0.0
    return float(_score_node(pred_obj, gold_obj, sch, sch))


def _strip_json_fences(raw: str) -> str:
    s = (raw or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s


def evaluate_structured_extraction(
    items: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
    schema_key: str = "schema",
    gold_key: str = "extraction",
) -> dict[str, Any]:
    """Evaluate ``LiquidAI/structured-extraction`` or ``ExtractBench`` outputs via ``grounded_field_f1``."""
    n = len(items)
    if len(raw_outputs) != n:
        raise ValueError(f"Mismatched items={n} vs raw_outputs={len(raw_outputs)}")

    per_valid: list[bool] = []
    per_scores: list[float] = []
    valid_json_cnt = 0
    valid_schema_cnt = 0
    repeat_rates: list[float] = []

    for i, item in enumerate(items):
        raw = raw_outputs[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        tids = token_ids_list[i] if token_ids_list is not None else None
        rep = repeat_4gram_rate(tids if tids is not None else raw)
        repeat_rates.append(rep)

        sch = item.get(schema_key)
        if isinstance(sch, str):
            sch = json.loads(sch)
        gold = item.get(gold_key)
        if isinstance(gold, str):
            gold = json.loads(gold)

        parsed = None
        is_json = False
        try:
            parsed = json.loads(_strip_json_fences(raw))
            is_json = True
            valid_json_cnt += 1
        except Exception:
            pass

        is_schema = bool(is_json and isinstance(sch, Mapping) and validate_against_jsonschema(parsed, sch))
        if is_schema:
            valid_schema_cnt += 1

        is_valid = bool(is_json and is_schema and str(fr) != "length")
        score = grounded_field_f1(parsed, gold, sch, require_schema_valid=True) if is_valid else 0.0
        per_valid.append(is_valid)
        per_scores.append(score)

    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "valid_json_rate": round(float(valid_json_cnt) / float(max(1, n)), 6),
        "valid_schema_rate": round(float(valid_schema_cnt) / float(max(1, n)), 6),
        "grounded_field_f1": round(float(np.mean(per_scores)) if per_scores else 0.0, 6),
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


# ---------------------------------------------------------------------------
# BFCL v3 AST + Irrelevance Abstention Checker
# ---------------------------------------------------------------------------

_BFCL_TYPE_MAP: dict[str, str] = {
    "dict": "object",
    "float": "number",
    "int": "integer",
    "integer": "integer",
    "number": "number",
    "string": "string",
    "str": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "tuple": "array",
    "object": "object",
    "any": "string",
}


def _sanitize_bfcl_schema(node: Any) -> dict[str, Any]:
    """Convert BFCL Gorilla tool parameter definitions into standard JSON Schema for ``llguidance``."""
    if not isinstance(node, Mapping):
        return {"type": "string"}
    raw_t = str(node.get("type") or "object").strip().lower()
    mapped_t = _BFCL_TYPE_MAP.get(raw_t, "string")
    out: dict[str, Any] = {"type": mapped_t}
    if "enum" in node and isinstance(node["enum"], list) and node["enum"]:
        out["enum"] = list(node["enum"])
    if mapped_t == "object":
        props = node.get("properties")
        if isinstance(props, Mapping):
            out["properties"] = {str(k): _sanitize_bfcl_schema(v) for k, v in props.items()}
            req = node.get("required")
            if isinstance(req, list):
                out["required"] = [str(r) for r in req if str(r) in out["properties"]]
            out["additionalProperties"] = False
    elif mapped_t == "array":
        items = node.get("items")
        if isinstance(items, Mapping):
            out["items"] = _sanitize_bfcl_schema(items)
    return out


def build_bfcl_response_schema(functions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build an ``llguidance``-compatible JSON Schema with explicit empty ``tool_calls: []`` abstention branch."""
    branches: list[dict[str, Any]] = []
    for fn in functions:
        fname = str(fn.get("name") or "")
        params = _sanitize_bfcl_schema(fn.get("parameters") or {"type": "object", "properties": {}})
        branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": fname},
                    "arguments": params,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    if not branches:
        item_schema: dict[str, Any] = {"type": "object"}
    elif len(branches) == 1:
        item_schema = branches[0]
    else:
        item_schema = {"anyOf": branches}

    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": item_schema,
                "maxItems": 8,
            }
        },
        "required": ["tool_calls"],
        "additionalProperties": False,
    }


def _ast_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            low = node.id.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low in ("none", "null"):
                return None
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = _ast_literal(node.operand)
            return -val if isinstance(val, (int, float)) else str(val)
        return ast.unparse(node) if hasattr(ast, "unparse") else ""


def _parse_python_call_expr(expr: ast.AST) -> list[tuple[str, dict[str, Any]]] | None:
    if isinstance(expr, ast.List) or isinstance(expr, ast.Tuple):
        calls: list[tuple[str, dict[str, Any]]] = []
        for elt in expr.elts:
            sub = _parse_python_call_expr(elt)
            if sub is None:
                return None
            calls.extend(sub)
        return calls
    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name):
            fname = expr.func.id
        elif isinstance(expr.func, ast.Attribute):
            parts: list[str] = []
            cur: ast.AST = expr.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            fname = ".".join(reversed(parts))
        else:
            return None
        kwargs: dict[str, Any] = {}
        for kw in expr.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = _ast_literal(kw.value)
        return [(fname, kwargs)]
    return None


def _normalize_call_list(obj: Any) -> tuple[bool, list[tuple[str, dict[str, Any]]]]:
    if isinstance(obj, Mapping):
        if "tool_calls" in obj:
            tc = obj["tool_calls"]
            if not isinstance(tc, list):
                return False, []
            return _normalize_call_list(tc)
        if "name" in obj and isinstance(obj["name"], str):
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return False, []
            if not isinstance(args, Mapping):
                return False, []
            return True, [(str(obj["name"]), dict(args))]
        # Single Gorilla dict format: {"func_name": {"arg": val}}
        if len(obj) == 1:
            k, v = next(iter(obj.items()))
            if isinstance(v, Mapping):
                return True, [(str(k), dict(v))]
        return False, []

    if isinstance(obj, list):
        calls: list[tuple[str, dict[str, Any]]] = []
        for elem in obj:
            ok, sub = _normalize_call_list(elem)
            if not ok:
                return False, []
            calls.extend(sub)
        return True, calls
    return False, []


def parse_bfcl_tool_calls(raw_output: str) -> tuple[bool, list[tuple[str, dict[str, Any]]]]:
    """Parse JSON or Python AST tool calls from model output; returns ``(parse_ok, [(name, kwargs), ...])``."""
    s = _strip_json_fences(raw_output)
    if not s:
        return True, []

    # 1. Try JSON directly.
    try:
        obj = json.loads(s)
        return _normalize_call_list(obj)
    except Exception:
        pass

    # 2. Try Python AST call list `[fn(a=1), fn2(b="x")]` or single call `fn(a=1)`.
    try:
        tree = ast.parse(s, mode="eval")
        parsed_py = _parse_python_call_expr(tree.body)
        if parsed_py is not None:
            return True, parsed_py
    except SyntaxError:
        pass

    # 3. Embedded JSON block inside prose (unconstrained model output).
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            ok, calls = _normalize_call_list(obj)
            if ok:
                return True, calls
        except Exception:
            pass

    # Plain-text refusal with no call syntax counts as empty call list (abstention).
    if "(" not in s and "{" not in s:
        return True, []
    return False, []


def _bfcl_norm_str(val: str) -> str:
    s = unicodedata.normalize("NFKC", val).strip().lower()
    s = re.sub(r"[\s,./\-_]+", "", s)
    return s


def _bfcl_value_matches(pred_val: Any, allowed_vals: Sequence[Any]) -> bool:
    for cand in allowed_vals:
        if cand is None and pred_val is None:
            return True
        if isinstance(cand, bool):
            if isinstance(pred_val, bool) and pred_val == cand:
                return True
            continue
        if isinstance(cand, (int, float)) and not isinstance(cand, bool):
            if isinstance(pred_val, (int, float)) and not isinstance(pred_val, bool):
                if abs(float(pred_val) - float(cand)) <= 1e-5 * max(1.0, abs(float(cand))):
                    return True
            continue
        if isinstance(cand, str):
            if isinstance(pred_val, str) and (pred_val == cand or _bfcl_norm_str(pred_val) == _bfcl_norm_str(cand)):
                return True
            continue
        if isinstance(cand, list) and isinstance(pred_val, list):
            if len(cand) == len(pred_val) and all(_bfcl_value_matches(pv, [cv]) for pv, cv in zip(pred_val, cand)):
                return True
            continue
        if isinstance(cand, Mapping) and isinstance(pred_val, Mapping):
            # BFCL v3 possible_answer represents nested dict parameters as {key: [allowed_vals...]}.
            if not set(pred_val.keys()).issubset(set(cand.keys())):
                continue
            nested_ok = True
            for k, raw_v in cand.items():
                nested_allowed = list(raw_v) if isinstance(raw_v, list) else [raw_v]
                optional = ("" in nested_allowed) or (None in nested_allowed)
                if k not in pred_val or pred_val[k] is None:
                    if not optional:
                        nested_ok = False
                        break
                    continue
                if not _bfcl_value_matches(pred_val[k], nested_allowed):
                    nested_ok = False
                    break
            if nested_ok:
                return True
            continue
        if pred_val == cand:
            return True
    return False


def _bfcl_func_name_matches(
    pred_name: str,
    gold_name: str,
    declared_names: Sequence[str],
) -> bool:
    """Check BFCL v3 function name equivalence (exact, '.' vs '_', and unique qualified-suffix match)."""
    if pred_name == gold_name:
        return True
    p_norm = pred_name.replace("_", ".")
    g_norm = gold_name.replace("_", ".")
    if p_norm == g_norm:
        return True
    # Handle upstream BFCL entries (e.g. simple_363) where ground_truth omits the namespace prefix
    # present in the declared function schema (`restaurant_search.find_closest` vs `find_closest`).
    if "." not in gold_name and declared_names:
        suffix_matches = [
            dn
            for dn in declared_names
            if dn == gold_name
            or dn.split(".")[-1] == gold_name
            or dn.replace("_", ".").endswith("." + g_norm)
        ]
        if len(suffix_matches) == 1:
            target = suffix_matches[0]
            if pred_name == target or p_norm == target.replace("_", "."):
                return True
    return False


def _bfcl_single_call_matches(
    pred_name: str,
    pred_kwargs: Mapping[str, Any],
    gold_entry: Mapping[str, Any],
    allowed_params_by_func: Mapping[str, set[str]],
    declared_names: Sequence[str] = (),
) -> bool:
    if len(gold_entry) != 1:
        return False
    gold_name, gold_params = next(iter(gold_entry.items()))
    if not isinstance(gold_params, Mapping):
        return False
    if not _bfcl_func_name_matches(pred_name, gold_name, declared_names):
        return False

    schema_params = (
        allowed_params_by_func.get(gold_name)
        or allowed_params_by_func.get(pred_name)
        or allowed_params_by_func.get(pred_name.replace("_", "."))
        or allowed_params_by_func.get(gold_name.replace("_", "."))
    )
    if schema_params is not None:
        extra_keys = set(pred_kwargs.keys()) - schema_params - set(gold_params.keys())
        if extra_keys:
            return False

    for param, allowed in gold_params.items():
        allowed_list = list(allowed) if isinstance(allowed, list) else [allowed]
        optional = ("" in allowed_list) or (None in allowed_list)
        if param not in pred_kwargs or pred_kwargs[param] is None:
            if not optional:
                return False
            continue
        if not _bfcl_value_matches(pred_kwargs[param], allowed_list):
            return False
    return True


def check_bfcl_ast_match(
    pred_calls: Sequence[tuple[str, dict[str, Any]]],
    ground_truth: Sequence[Mapping[str, Any]],
    functions: Sequence[Mapping[str, Any]] | None = None,
    *,
    is_irrelevance: bool = False,
) -> bool:
    """Verify BFCL v3 AST call equivalence (unordered parallel calls + ``irrelevance`` abstention)."""
    if is_irrelevance:
        return len(pred_calls) == 0
    if len(pred_calls) != len(ground_truth):
        return False
    if len(ground_truth) == 0:
        return True

    allowed_params: dict[str, set[str]] = {}
    declared_names: list[str] = []
    for fn in functions or ():
        fname = str(fn.get("name") or "")
        if fname:
            declared_names.append(fname)
        props = (fn.get("parameters") or {}).get("properties") or {}
        if isinstance(props, Mapping):
            pset = set(str(k) for k in props.keys())
            allowed_params[fname] = pset
            allowed_params[fname.replace("_", ".")] = pset

    m = len(ground_truth)
    compat = np.zeros((m, m), dtype=np.float64)
    for r, (pn, pk) in enumerate(pred_calls):
        for c, g_entry in enumerate(ground_truth):
            if _bfcl_single_call_matches(pn, pk, g_entry, allowed_params, declared_names=declared_names):
                compat[r, c] = 1.0
    r_ind, c_ind = linear_sum_assignment_max(compat)
    return int(round(float(np.sum(compat[r_ind, c_ind])))) == m


def evaluate_bfcl_v3(
    items: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate BFCL v3 AST + ``irrelevance`` abstention across all splits."""
    n = len(items)
    if len(raw_outputs) != n:
        raise ValueError(f"Mismatched items={n} vs raw_outputs={len(raw_outputs)}")

    per_valid: list[bool] = []
    per_scores: list[float] = []
    by_split: dict[str, list[float]] = {}
    repeat_rates: list[float] = []

    for i, item in enumerate(items):
        raw = raw_outputs[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        tids = token_ids_list[i] if token_ids_list is not None else None
        rep = repeat_4gram_rate(tids if tids is not None else raw)
        repeat_rates.append(rep)

        parse_ok, calls = parse_bfcl_tool_calls(raw)
        is_valid = bool(parse_ok and str(fr) != "length")
        is_irrel = bool(item.get("is_irrelevance", item.get("split") == "irrelevance"))
        matched = (
            check_bfcl_ast_match(
                calls,
                item.get("ground_truth") or [],
                item.get("functions") or [],
                is_irrelevance=is_irrel,
            )
            if is_valid
            else False
        )
        sc = 1.0 if matched else 0.0
        per_valid.append(is_valid)
        per_scores.append(sc)
        by_split.setdefault(str(item.get("split") or "unknown"), []).append(sc)

    split_acc = {k: round(float(np.mean(v)), 6) for k, v in sorted(by_split.items())}
    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "overall_ast_acc": round(float(np.mean(per_scores)) if per_scores else 0.0, 6),
        "abstention_acc": split_acc.get("irrelevance"),
        "split_accuracy": split_acc,
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


# ---------------------------------------------------------------------------
# JSONSchemaBench Auxiliary `llguidance` Compiler & Forced-Jump Diagnostic
# ---------------------------------------------------------------------------


def _inspect_schema_keywords(schema_str: str) -> dict[str, bool]:
    return {
        "has_ref": '"$ref"' in schema_str,
        "has_anyof": '"anyOf"' in schema_str,
        "has_allof": '"allOf"' in schema_str,
        "has_additional_false": '"additionalProperties": false' in schema_str
        or '"additionalProperties":false' in schema_str,
        "has_pattern": '"pattern"' in schema_str,
    }


def evaluate_jsonschemabench_compiler(
    items: Sequence[Mapping[str, Any]],
    *,
    llguidance_tokenizer: Any | None = None,
    whitespace_flexible: bool = False,
    coerce_one_of: bool = True,
) -> dict[str, Any]:
    """Measure ``llguidance`` Grammar Compilation Time (GCT p50/p99 ms), keyword coverage, and ``ff_q`` jumps."""
    import llguidance

    compiler = llguidance.JsonCompiler(
        separators=(",", ":"),
        whitespace_flexible=whitespace_flexible,
        coerce_one_of=bool(coerce_one_of),
    )
    gct_ms: list[float] = []
    compile_ok = 0
    upfront_ff_counts: list[int] = []
    kw_totals: dict[str, int] = {"has_ref": 0, "has_anyof": 0, "has_allof": 0, "has_additional_false": 0, "has_pattern": 0}
    kw_compiled: dict[str, int] = dict(kw_totals)
    errors: list[dict[str, str]] = []

    for item in items:
        s_raw = item.get("json_schema")
        s_str = s_raw if isinstance(s_raw, str) else json.dumps(s_raw)
        flags = _inspect_schema_keywords(s_str)
        for k, present in flags.items():
            if present:
                kw_totals[k] += 1
        t0 = time.perf_counter()
        try:
            gram = compiler.compile(s_str)
            dt_ms = (time.perf_counter() - t0) * 1e3
            gct_ms.append(dt_ms)
            compile_ok += 1
            for k, present in flags.items():
                if present:
                    kw_compiled[k] += 1
            if llguidance_tokenizer is not None:
                matcher = llguidance.LLMatcher(llguidance_tokenizer, gram)
                ff = matcher.compute_ff_tokens()
                upfront_ff_counts.append(len(ff))
        except Exception as exc:
            if len(errors) < 10:
                errors.append({"id": str(item.get("id") or ""), "error": f"{type(exc).__name__}: {str(exc)[:120]}"})

    n = len(items)
    gct_arr = np.asarray(gct_ms, dtype=np.float64) if gct_ms else np.zeros((1,), dtype=np.float64)
    return {
        "n_schemas": n,
        "compiled_ok": compile_ok,
        "compile_success_rate": round(float(compile_ok) / float(max(1, n)), 6),
        "gct_p50_ms": round(float(np.percentile(gct_arr, 50)), 3) if gct_ms else 0.0,
        "gct_p99_ms": round(float(np.percentile(gct_arr, 99)), 3) if gct_ms else 0.0,
        "mean_upfront_ff_tokens": round(float(np.mean(upfront_ff_counts)), 3) if upfront_ff_counts else None,
        "keyword_coverage": {
            k: {
                "present": kw_totals[k],
                "compiled": kw_compiled[k],
                "rate": round(float(kw_compiled[k]) / float(max(1, kw_totals[k])), 4) if kw_totals[k] else 1.0,
            }
            for k in kw_totals
        },
        "sample_errors": errors,
    }


__all__ = [
    "build_bfcl_response_schema",
    "check_bfcl_ast_match",
    "evaluate_bfcl_v3",
    "evaluate_jsonschemabench_compiler",
    "evaluate_structured_extraction",
    "grounded_field_f1",
    "hungarian_min_cost",
    "linear_sum_assignment_max",
    "parse_bfcl_tool_calls",
    "validate_against_jsonschema",
]
