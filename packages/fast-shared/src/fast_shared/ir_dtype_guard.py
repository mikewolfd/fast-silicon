"""Static MLX Python / Metal kernel and CoreML MIL Program IR dtype & regression guard.

Targets known Apple Silicon M4 Pro silent performance / accuracy regressions
documented in `M4_PRO_MAX_HW_LESSONS_LEARNED.md`:
  1. Integer (`int32`/`int64`/`bool`) or default `float32` SDPA masks causing silent FP32 upcasts.
  2. Mixed `bfloat16` x `float16` or `bf16/fp16` x `float32` scale/weight implicit upcasts.
  3. Unintended `mx.float32` outputs (or `device float*` buffers) from fused Metal kernels.
  4. `0D` scalar arrays (`reshape(())`, scalar `mx.array(x)`) bound to Metal kernel array pointers.
  5. `mx.compile` inside loops or synchronous offset/layer `mx.eval` inside tight loops.
  6. CoreML MIL `Program` `cast(..., dtype="fp32")`, `common::add_fp16_cast` pass artifacts,
     non-FP16 I/O, FP32 compute ops, and non-FP16 SDPA masks.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Literal, Sequence

Severity = Literal["error", "warning"]

_ALLOWED_FP32_STATS_OUTPUTS = frozenset(
    {
        "ml_part",
        "lse",
        "lse_part",
        "m_l",
        "stats",
        "acc_f32",
    }
)

_INT_DTYPES = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_LOW_FLOAT_DTYPES = frozenset({"float16", "bfloat16"})
_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_DEVICE_FP32_PTR_RE = re.compile(r"\bdevice\s+(?:const\s+)?(?:packed_)?float(?:[234])?\s*\*")
_DEVICE_BF16_PTR_RE = re.compile(r"\bdevice\s+(?:const\s+)?(?:packed_)?bfloat(?:[234])?\s*\*")
_DEVICE_FP16_PTR_RE = re.compile(r"\bdevice\s+(?:const\s+)?(?:packed_)?half(?:[234])?\s*\*")


def _is_const_qualified_ptr(src_text: str, device_pos: int, match_text: str) -> bool:
    """Return True only for genuinely read-only `const device T*` / `device const T*`.

    `const_cast<device float*>(...)` strips `const` and MUST NOT be ignored.
    """
    prefix = src_text[max(0, device_pos - 24) : device_pos]
    if re.search(r"\bconst_cast\s*<\s*$", prefix):
        return False
    if re.search(r"\bconst\s+$", prefix):
        return True
    if re.search(r"\bdevice\s+const\s+", match_text):
        return True
    return False


@dataclass(frozen=True)
class GuardFinding:
    rule_id: str
    severity: Severity
    file: str
    line: int
    symbol: str
    message: str

    def format(self) -> str:
        return f"[{self.severity.upper()}] {self.rule_id} {self.file}:{self.line} ({self.symbol}): {self.message}"


@dataclass
class GuardReport:
    findings: list[GuardFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[GuardFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[GuardFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def extend(self, other: "GuardReport") -> None:
        self.findings.extend(other.findings)


def _attr_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _mx_dtype_from_node(node: ast.AST | None, sym_dtypes: dict[str, str] | None = None) -> str | None:
    if node is None:
        return None
    chain = _attr_chain(node)
    if chain is not None:
        mapping = {
            "mx.float16": "float16",
            "mx.bfloat16": "bfloat16",
            "mx.float32": "float32",
            "mx.int8": "int8",
            "mx.int16": "int16",
            "mx.int32": "int32",
            "mx.int64": "int64",
            "mx.uint8": "uint8",
            "mx.uint16": "uint16",
            "mx.uint32": "uint32",
            "mx.uint64": "uint64",
            "mx.bool_": "bool",
            "np.float16": "float16",
            "np.float32": "float32",
            "np.int32": "int32",
            "np.int64": "int64",
            "np.bool_": "bool",
        }
        if chain in mapping:
            return mapping[chain]
        if sym_dtypes and chain in sym_dtypes:
            return sym_dtypes[chain]
        if chain.endswith(".dtype") and sym_dtypes:
            base = chain[: -len(".dtype")]
            return sym_dtypes.get(base)
    if isinstance(node, ast.Name) and sym_dtypes:
        return sym_dtypes.get(node.id)
    return None


def _kw_arg(call: ast.Call, name: str, pos_idx: int | None = None) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    if pos_idx is not None and pos_idx < len(call.args):
        return call.args[pos_idx]
    return None


def _is_explicit_cast(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "astype"


def _extract_const_str(node: ast.AST, str_consts: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in str_consts:
        return str_consts[node.id]
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                parts.append(val.value)
            elif isinstance(val, ast.FormattedValue):
                parts.append("{expr}")
        return "".join(parts)
    return None


def _extract_str_list(node: ast.AST | None, str_consts: dict[str, str]) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    out: list[str] = []
    for elt in node.elts:
        s = _extract_const_str(elt, str_consts)
        if s is not None:
            out.append(s)
    return out


class _MlxAstChecker(ast.NodeVisitor):
    def __init__(
        self,
        *,
        filename: str,
        source: str,
        flag_bool_sdpa_as_error: bool = False,
        check_loop_hazards: bool = True,
        expected_act_dtype: str | None = None,
    ) -> None:
        self.filename = filename
        self.source = source
        self.lines = source.splitlines()
        self.flag_bool_sdpa_as_error = flag_bool_sdpa_as_error
        self.check_loop_hazards = check_loop_hazards
        self.expected_act_dtype = expected_act_dtype
        self.findings: list[GuardFinding] = []

        self.scope_stack: list[str] = ["<module>"]
        self.dtype_stack: list[dict[str, str]] = [{}]
        self.class_attrs_stack: list[dict[str, str]] = []
        self.zero_d_stack: list[set[str]] = [set()]
        self.str_consts: dict[str, str] = {}
        self.metal_kernels: dict[str, dict[str, Any]] = {}
        self.metal_kernel_factories: dict[str, dict[str, Any]] = {}
        self._checked_metal_defs: set[int] = set()
        self.loop_stack: list[ast.AST] = []

    @property
    def _cur_symbol(self) -> str:
        return ".".join(self.scope_stack)

    @property
    def _dtypes(self) -> dict[str, str]:
        return self.dtype_stack[-1]

    @property
    def _zero_d(self) -> set[str]:
        return self.zero_d_stack[-1]

    def _cur_sym_dtypes(self) -> dict[str, str]:
        merged = dict(self._dtypes)
        if self.class_attrs_stack:
            merged.update(self.class_attrs_stack[-1])
        return merged

    def _cur_act_dtype(self) -> str | None:
        if self.class_attrs_stack and "self.act_dtype" in self.class_attrs_stack[-1]:
            return self.class_attrs_stack[-1]["self.act_dtype"]
        return self.expected_act_dtype

    def _line_has_ignore(self, lineno: int, tag: str) -> bool:
        if 1 <= lineno <= len(self.lines):
            txt = self.lines[lineno - 1]
            return ("guard: allow-" + tag) in txt or "guard: ignore" in txt
        return False

    def _emit(self, rule_id: str, severity: Severity, lineno: int, message: str) -> None:
        self.findings.append(
            GuardFinding(
                rule_id=rule_id,
                severity=severity,
                file=self.filename,
                line=lineno,
                symbol=self._cur_symbol,
                message=message,
            )
        )

    def _infer_expr_dtype(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self._dtypes.get(node.id)
        if isinstance(node, ast.Attribute):
            chain = _attr_chain(node)
            if chain:
                if self.class_attrs_stack and chain in self.class_attrs_stack[-1]:
                    return self.class_attrs_stack[-1][chain]
                if chain in self._dtypes:
                    return self._dtypes[chain]
            return None
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript)):
                return self._infer_expr_dtype(node.value)
            return None
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Invert):
                return "bool"
            return self._infer_expr_dtype(node.operand)
        if isinstance(node, ast.Compare):
            return "bool"
        if isinstance(node, ast.BinOp):
            ld = self._infer_expr_dtype(node.left)
            rd = self._infer_expr_dtype(node.right)
            if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)) and (ld == "bool" or rd == "bool"):
                return "bool"
            if ld is not None and ld == rd:
                return ld
            if ld in _FLOAT_DTYPES and rd in _FLOAT_DTYPES:
                if "float32" in (ld, rd) or ({ld, rd} == {"bfloat16", "float16"}):
                    return "float32"
            return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "astype":
                dt_node = _kw_arg(node, "dtype", 0)
                return _mx_dtype_from_node(dt_node, self._cur_sym_dtypes())
            chain = _attr_chain(node.func)
            if chain in ("mx.array", "mx.ones", "mx.zeros", "mx.full", "mx.empty", "mx.arange", "np.array", "np.ones", "np.zeros", "np.full", "np.arange"):
                dt_node = _kw_arg(node, "dtype")
                if dt_node is not None:
                    return _mx_dtype_from_node(dt_node, self._cur_sym_dtypes())
                if chain in ("mx.arange", "np.arange"):
                    return "int32"
                if chain in ("mx.array", "np.array") and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, float):
                        return "float32"
                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, int) and not isinstance(arg0.value, bool):
                        return "int32"
                return None
            if chain in ("mx.where", "np.where"):
                if len(node.args) >= 3:
                    dt1 = self._infer_expr_dtype(node.args[1])
                    dt2 = self._infer_expr_dtype(node.args[2])
                    if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, float):
                        dt1 = dt1 or "float32"
                    if isinstance(node.args[2], ast.Constant) and isinstance(node.args[2].value, float):
                        dt2 = dt2 or "float32"
                    if isinstance(node.args[1], ast.UnaryOp) and isinstance(node.args[1].operand, ast.Constant):
                        if isinstance(node.args[1].operand.value, float):
                            dt1 = dt1 or "float32"
                    if isinstance(node.args[2], ast.UnaryOp) and isinstance(node.args[2].operand, ast.Constant):
                        if isinstance(node.args[2].operand.value, float):
                            dt2 = dt2 or "float32"
                    if dt1 == dt2:
                        return dt1
                    if "float32" in (dt1, dt2):
                        return "float32"
            if chain == "mx.contiguous" and node.args:
                return self._infer_expr_dtype(node.args[0])
            if chain in ("mx.matmul", "mx.addmm"):
                args = list(node.args) + [
                    kw.value for kw in node.keywords if kw.arg in ("a", "b", "c")
                ]
                known = [self._infer_expr_dtype(a) for a in args]
                if known and all(dt is not None for dt in known):
                    if len(set(known)) == 1:
                        return known[0]
                    if all(dt in _FLOAT_DTYPES for dt in known) and (
                        "float32" in known or set(known) == {"bfloat16", "float16"}
                    ):
                        return "float32"
                return None
            if isinstance(node.func, ast.Attribute) and node.func.attr == "reshape":
                return self._infer_expr_dtype(node.func.value)
        return None

    def _is_zero_d_expr(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id in self._zero_d:
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "reshape" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Tuple) and len(arg0.elts) == 0:
                    return True
            chain = _attr_chain(node.func)
            if chain == "mx.array" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, (int, float)):
                    return True
                if isinstance(arg0, ast.UnaryOp) and isinstance(arg0.operand, ast.Constant):
                    return True
            if chain == "mx.contiguous" and node.args:
                return self._is_zero_d_expr(node.args[0])
        return False

    def _record_metal_kernel_def(self, target_name: str, call: ast.Call) -> None:
        in_names = _extract_str_list(_kw_arg(call, "input_names"), self.str_consts)
        out_names = _extract_str_list(_kw_arg(call, "output_names"), self.str_consts)
        src_node = _kw_arg(call, "source")
        src_text = _extract_const_str(src_node, self.str_consts) if src_node is not None else ""
        info = {
            "input_names": in_names,
            "output_names": out_names,
            "source": src_text or "",
            "lineno": call.lineno,
        }
        self.metal_kernels[target_name] = info
        if len(self.scope_stack) >= 2:
            self.metal_kernel_factories[self.scope_stack[-1]] = info
        self._checked_metal_defs.add(id(call))
        self._check_metal_kernel_source(target_name, out_names, src_text or "", call.lineno)

    def _check_metal_kernel_source(
        self,
        kern_name: str,
        out_names: list[str],
        src_text: str,
        lineno: int,
    ) -> None:
        if not src_text or self._line_has_ignore(lineno, "fp32"):
            return
        # Flag mutable `device float*` / `device float4*` / `const_cast<device float*>` pointers unless allowed LSE/stats
        allowed_non_stats = [o for o in out_names if o not in _ALLOWED_FP32_STATS_OUTPUTS]
        if allowed_non_stats:
            for m in _DEVICE_FP32_PTR_RE.finditer(src_text):
                if not _is_const_qualified_ptr(src_text, m.start(), m.group(0)):
                    self._emit(
                        "METAL_FP32_OUTPUT_PTR",
                        "error",
                        lineno,
                        f"Metal kernel `{kern_name}` uses mutable `{m.group(0).strip()}` pointer for non-stats outputs {allowed_non_stats}; expected `half`/`bfloat`.",
                    )
                    break

        act_dt = self._cur_act_dtype()
        if act_dt == "float16" and not self._line_has_ignore(lineno, "bf16-ptr"):
            for m in _DEVICE_BF16_PTR_RE.finditer(src_text):
                if not _is_const_qualified_ptr(src_text, m.start(), m.group(0)):
                    self._emit(
                        "METAL_BF16_PTR_IN_FP16_KERNEL",
                        "error",
                        lineno,
                        f"Metal kernel `{kern_name}` uses `{m.group(0).strip()}` pointer inside FP16 (`float16`) context; expected `half`/`half4`.",
                    )
                    break

    def _check_metal_kernel_call(self, call: ast.Call, kern_info: dict[str, Any] | None) -> None:
        out_dtypes_node = _kw_arg(call, "output_dtypes")
        out_names = kern_info["output_names"] if kern_info else []
        in_names = kern_info["input_names"] if kern_info else []
        src_text = kern_info["source"] if kern_info else ""
        sym_dtypes = self._cur_sym_dtypes()

        call_out_dts: list[str | None] = []
        if isinstance(out_dtypes_node, (ast.List, ast.Tuple)):
            for idx, elt in enumerate(out_dtypes_node.elts):
                dt = _mx_dtype_from_node(elt, sym_dtypes)
                call_out_dts.append(dt)
                if dt == "float32" and not self._line_has_ignore(call.lineno, "fp32"):
                    oname = out_names[idx] if idx < len(out_names) else f"out[{idx}]"
                    if oname not in _ALLOWED_FP32_STATS_OUTPUTS:
                        self._emit(
                            "MLX_METAL_FP32_OUTPUT",
                            "error",
                            call.lineno,
                            f"Custom Metal kernel output `{oname}` uses `mx.float32`; use `mx.float16`/`mx.bfloat16` (only log-sum-exp stats outputs like `ml_part` may use `float32`).",
                        )

        # Cross-check kernel Metal pointer precision against explicit call `output_dtypes`
        if src_text and call_out_dts and not self._line_has_ignore(call.lineno, "ptr-mismatch"):
            if "float16" in call_out_dts and "bfloat16" not in call_out_dts and self._cur_act_dtype() != "float16":
                for m in _DEVICE_BF16_PTR_RE.finditer(src_text):
                    if not _is_const_qualified_ptr(src_text, m.start(), m.group(0)):
                        self._emit(
                            "METAL_BF16_PTR_IN_FP16_KERNEL",
                            "error",
                            call.lineno,
                            f"Metal kernel outputs `float16` `{call_out_dts}` but source uses `{m.group(0).strip()}` pointer.",
                        )
                        break
            elif "bfloat16" in call_out_dts and "float16" not in call_out_dts:
                for m in _DEVICE_FP16_PTR_RE.finditer(src_text):
                    if not _is_const_qualified_ptr(src_text, m.start(), m.group(0)):
                        self._emit(
                            "METAL_FP16_PTR_IN_BF16_KERNEL",
                            "error",
                            call.lineno,
                            f"Metal kernel outputs `bfloat16` `{call_out_dts}` but source uses `{m.group(0).strip()}` pointer.",
                        )
                        break

        inputs_node = _kw_arg(call, "inputs")
        if isinstance(inputs_node, (ast.List, ast.Tuple)):
            for idx, elt in enumerate(inputs_node.elts):
                if self._is_zero_d_expr(elt):
                    iname = in_names[idx] if idx < len(in_names) else f"input[{idx}]"
                    if not src_text or f"{iname}[0]" in src_text or f"{iname}[" in src_text or "*" in src_text:
                        self._emit(
                            "MLX_METAL_0D_SCALAR",
                            "error",
                            call.lineno,
                            f"0D scalar passed as `{iname}` to Metal kernel; MLX compiles 0D arrays as value scalars rather than `const device T*` (use `.reshape((1,))`).",
                        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope_stack.append(node.name)
        parent_dtypes = dict(self._dtypes)
        self.dtype_stack.append(parent_dtypes)
        self.zero_d_stack.append(set())
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.arg in ("dt", "dtype") and arg.annotation:
                dt = _mx_dtype_from_node(arg.annotation, parent_dtypes)
                if dt:
                    self._dtypes[arg.arg] = dt
        self.generic_visit(node)
        self.zero_d_stack.pop()
        self.dtype_stack.pop()
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope_stack.append(node.name)
        self.dtype_stack.append(dict(self._dtypes))
        self.zero_d_stack.append(set())
        self.generic_visit(node)
        self.zero_d_stack.pop()
        self.dtype_stack.pop()
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_stack.append(node.name)
        cls_attrs: dict[str, str] = {}
        if self.expected_act_dtype is not None:
            cls_attrs["self.act_dtype"] = self.expected_act_dtype
        # Pre-seed explicit `self.act_dtype = ...` from `__init__` scoped only to this class
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Attribute):
                        chain = _attr_chain(sub.targets[0])
                        if chain and chain.startswith("self."):
                            dt = _mx_dtype_from_node(sub.value, cls_attrs)
                            if dt is not None:
                                cls_attrs[chain] = dt
        self.class_attrs_stack.append(cls_attrs)
        self.generic_visit(node)
        self.class_attrs_stack.pop()
        self.scope_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _attr_chain(node.value.func) == "mx.fast.metal_kernel"
        ):
            self._record_metal_kernel_def(node.targets[0].id, node.value)
        self.generic_visit(node)
        if len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Call):
                fchain = _attr_chain(node.value.func)
                if fchain and fchain in self.metal_kernel_factories:
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            self.metal_kernels[elt.id] = self.metal_kernel_factories[fchain]
                return
            if isinstance(tgt, ast.Name):
                sval = _extract_const_str(node.value, self.str_consts)
                if sval is not None and len(self.scope_stack) == 1:
                    self.str_consts[tgt.id] = sval
                dt = self._infer_expr_dtype(node.value)
                if dt is not None:
                    self._dtypes[tgt.id] = dt
                elif tgt.id in self._dtypes:
                    del self._dtypes[tgt.id]
                if self._is_zero_d_expr(node.value):
                    self._zero_d.add(tgt.id)
                else:
                    self._zero_d.discard(tgt.id)
                if isinstance(node.value, ast.Call):
                    fchain = _attr_chain(node.value.func)
                    if fchain and fchain in self.metal_kernel_factories:
                        self.metal_kernels[tgt.id] = self.metal_kernel_factories[fchain]
            elif isinstance(tgt, ast.Attribute):
                chain = _attr_chain(tgt)
                if chain and chain.startswith("self.") and self.class_attrs_stack:
                    dt = _mx_dtype_from_node(node.value, self._cur_sym_dtypes()) or self._infer_expr_dtype(node.value)
                    if dt is not None:
                        self.class_attrs_stack[-1][chain] = dt

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if node.value is not None and isinstance(node.target, ast.Name):
            dt = self._infer_expr_dtype(node.value)
            if dt is not None:
                self._dtypes[node.target.id] = dt
            if self._is_zero_d_expr(node.value):
                self._zero_d.add(node.target.id)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.generic_visit(node)
        # Elementwise arithmetic & `@` matmul where operands have explicitly known float dtypes
        if isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.Div, ast.MatMult)):
            if self._line_has_ignore(node.lineno, "mixed-dtype"):
                return
            ld = self._infer_expr_dtype(node.left)
            rd = self._infer_expr_dtype(node.right)
            if ld not in _FLOAT_DTYPES or rd not in _FLOAT_DTYPES:
                return
            if {ld, rd} == {"bfloat16", "float16"}:
                self._emit(
                    "MLX_MIXED_BF16_FP16",
                    "error",
                    node.lineno,
                    "Mixed `bfloat16` and `float16` arithmetic implicitly upcasts to `float32` on Apple Silicon; align both operands to matching native dtype.",
                )
            elif (
                "float32" in (ld, rd)
                and (ld in _LOW_FLOAT_DTYPES or rd in _LOW_FLOAT_DTYPES)
                and not (_is_explicit_cast(node.left) or _is_explicit_cast(node.right))
            ):
                self._emit(
                    "MLX_IMPLICIT_FP32_SCALE_UPCAST",
                    "error",
                    node.lineno,
                    f"Mixing `{ld}` and `{rd}` in binary op causes silent `float32` promotion; cast scales/arrays to matching `{ld if ld in _LOW_FLOAT_DTYPES else rd}`.",
                )

    def _is_layer_loop(self, loop_node: ast.AST) -> bool:
        if not isinstance(loop_node, ast.For):
            return False
        iter_src = ast.get_source_segment(self.source, loop_node.iter) or ""
        return any(tok in iter_src for tok in ("layers", "NUM_LAYERS", "num_layers", "first_kv_shared"))

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        chain = _attr_chain(node.func)

        # 1. SDPA mask checks
        if chain in ("mx.fast.scaled_dot_product_attention", "scaled_dot_product_attention"):
            mask_node = _kw_arg(node, "mask", 3)
            if mask_node is not None and not (
                isinstance(mask_node, ast.Constant) and (mask_node.value is None or mask_node.value == "causal")
            ):
                mdt = self._infer_expr_dtype(mask_node)
                if mdt in _INT_DTYPES and not self._line_has_ignore(node.lineno, "int-mask"):
                    self._emit(
                        "MLX_SDPA_INT_MASK",
                        "error",
                        node.lineno,
                        f"Integer SDPA mask (`{mdt}`) triggers silent `FP32` SDPA upcast (-36% throughput); use additive FP16 (`-65504.0`) or explicit FP16/BF16 mask.",
                    )
                elif mdt == "float32" and not self._line_has_ignore(node.lineno, "fp32-mask"):
                    self._emit(
                        "MLX_SDPA_FP32_MASK",
                        "error",
                        node.lineno,
                        "FP32 SDPA mask triggers silent `FP32` attention upcast; cast additive mask to `float16` (`-65504.0`) or `bfloat16`.",
                    )
                elif mdt == "bool" and not self._line_has_ignore(node.lineno, "bool-mask"):
                    sev: Severity = "error" if self.flag_bool_sdpa_as_error else "warning"
                    self._emit(
                        "MLX_SDPA_BOOL_MASK",
                        sev,
                        node.lineno,
                        "Boolean SDPA mask detected; prefer additive FP16 (`-65504.0`) or explicit FP16 mask on dense encoder paths if mask cast overhead appears.",
                    )

        # 2. Metal kernel definitions and invocations
        if chain == "mx.fast.metal_kernel" and _kw_arg(node, "output_dtypes") is None and id(node) not in self._checked_metal_defs:
            # Standalone definition (non-assigned) still gets source checked
            out_names = _extract_str_list(_kw_arg(node, "output_names"), self.str_consts)
            src_node = _kw_arg(node, "source")
            src_text = _extract_const_str(src_node, self.str_consts) if src_node is not None else ""
            if src_text and isinstance(node.func, ast.Attribute):
                self._checked_metal_defs.add(id(node))
                self._check_metal_kernel_source("<inline>", out_names, src_text, node.lineno)

        if _kw_arg(node, "output_dtypes") is not None and _kw_arg(node, "inputs") is not None:
            kern_info = None
            if isinstance(node.func, ast.Name):
                kern_info = self.metal_kernels.get(node.func.id)
            elif isinstance(node.func, ast.Call) and _attr_chain(node.func.func) == "mx.fast.metal_kernel":
                kern_info = {
                    "input_names": _extract_str_list(_kw_arg(node.func, "input_names"), self.str_consts),
                    "output_names": _extract_str_list(_kw_arg(node.func, "output_names"), self.str_consts),
                    "source": _extract_const_str(_kw_arg(node.func, "source"), self.str_consts) or "",
                }
            self._check_metal_kernel_call(node, kern_info)

        # 3. Quantized matmul / gather_qmm mixed dtype & missing lhs_indices traps
        if chain in ("mx.gather_qmm", "mx.quantized_matmul"):
            x_dt = self._infer_expr_dtype(_kw_arg(node, "x", 0))
            s_dt = self._infer_expr_dtype(_kw_arg(node, "scales", 2))
            if {x_dt, s_dt} == {"bfloat16", "float16"} or (
                "float32" in (x_dt, s_dt) and (x_dt in _LOW_FLOAT_DTYPES or s_dt in _LOW_FLOAT_DTYPES)
            ):
                self._emit(
                    "MLX_IMPLICIT_FP32_SCALE_UPCAST",
                    "error",
                    node.lineno,
                    f"`{chain}` mixes activation `{x_dt}` and scales `{s_dt}`; align scales to activation dtype to prevent silent `FP32` promotion.",
                )
            if chain == "mx.gather_qmm" and _kw_arg(node, "rhs_indices") is not None and _kw_arg(node, "lhs_indices") is None:
                if not self._line_has_ignore(node.lineno, "lhs-indices"):
                    self._emit(
                        "MLX_GATHER_QMM_MISSING_LHS",
                        "error",
                        node.lineno,
                        "`mx.gather_qmm` with `rhs_indices` and missing `lhs_indices` silently broadcasts every token across all experts.",
                    )

        # 3b. Matrix multiplies (`mx.matmul`, `mx.addmm`) with conflicting known float dtypes
        if chain in ("mx.matmul", "mx.addmm") and not self._line_has_ignore(node.lineno, "mixed-dtype"):
            op_nodes = (
                [_kw_arg(node, "a", 0), _kw_arg(node, "b", 1)]
                if chain == "mx.matmul"
                else [_kw_arg(node, "c", 0), _kw_arg(node, "a", 1), _kw_arg(node, "b", 2)]
            )
            op_dts = [self._infer_expr_dtype(n) for n in op_nodes if n is not None]
            if len(op_dts) >= 2 and all(dt in _FLOAT_DTYPES for dt in op_dts):
                unique_dts = set(op_dts)
                if {"bfloat16", "float16"}.issubset(unique_dts):
                    self._emit(
                        "MLX_MIXED_BF16_FP16",
                        "error",
                        node.lineno,
                        f"`{chain}` mixes `bfloat16` and `float16` operands ({sorted(unique_dts)}), causing silent `FP32` GEMM upcast.",
                    )
                elif "float32" in unique_dts and any(dt in _LOW_FLOAT_DTYPES for dt in unique_dts):
                    # Allow only if EVERY float32 operand was explicitly `.astype(mx.float32)`
                    f32_explicit = all(
                        _is_explicit_cast(n)
                        for n, dt in zip(op_nodes, op_dts)
                        if dt == "float32" and n is not None
                    )
                    if not f32_explicit:
                        target_dt = next(dt for dt in op_dts if dt in _LOW_FLOAT_DTYPES)
                        self._emit(
                            "MLX_IMPLICIT_FP32_SCALE_UPCAST",
                            "error",
                            node.lineno,
                            f"`{chain}` mixes `float32` and `{target_dt}` operands ({sorted(unique_dts)}); cast operands to matching `{target_dt}`.",
                        )

        # 4. Loop hazards (`mx.compile` inside loop; scalar/layer sync `mx.eval` inside tight loops)
        if self.check_loop_hazards and self.loop_stack:
            if chain == "mx.compile" and not self._line_has_ignore(node.lineno, "compile"):
                self._emit(
                    "MLX_COMPILE_IN_LOOP",
                    "error",
                    node.lineno,
                    "`mx.compile` called inside a loop triggers repeated graph recompilation overhead; hoist outside the loop.",
                )
            elif chain == "mx.eval" and not self._line_has_ignore(node.lineno, "eval"):
                fn_name = self.scope_stack[-1]
                is_setup_or_bench = fn_name.startswith(
                    ("attach_", "__init__", "warmup", "make_", "build_", "export_", "main", "bench")
                )
                arg_names = [a.id for a in node.args if isinstance(a, ast.Name)]
                if any(a.startswith("off") or a in ("step", "i", "idx") for a in arg_names):
                    self._emit(
                        "MLX_SYNC_EVAL_OFFSET_IN_LOOP",
                        "error",
                        node.lineno,
                        f"Synchronous `mx.eval({', '.join(arg_names)})` on small scalar/offset inside loop introduces host-GPU sync stalls.",
                    )
                elif not is_setup_or_bench and any(self._is_layer_loop(lp) for lp in self.loop_stack):
                    self._emit(
                        "MLX_EVAL_IN_LAYER_LOOP",
                        "error",
                        node.lineno,
                        "Synchronous `mx.eval` inside a per-layer loop breaks graph fusion and introduces per-layer host-GPU sync stalls.",
                    )


def check_mlx_source(
    source: str,
    *,
    filename: str = "<memory>",
    flag_bool_sdpa_as_error: bool = False,
    check_loop_hazards: bool = True,
    expected_act_dtype: str | None = None,
) -> GuardReport:
    tree = ast.parse(source, filename=filename)
    checker = _MlxAstChecker(
        filename=filename,
        source=source,
        flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
        check_loop_hazards=check_loop_hazards,
        expected_act_dtype=expected_act_dtype,
    )
    checker.visit(tree)
    return GuardReport(findings=checker.findings)


def check_mlx_file(
    path: str | Path,
    *,
    flag_bool_sdpa_as_error: bool = False,
    check_loop_hazards: bool = True,
    expected_act_dtype: str | None = None,
) -> GuardReport:
    p = Path(path)
    return check_mlx_source(
        p.read_text(encoding="utf-8"),
        filename=str(p),
        flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
        check_loop_hazards=check_loop_hazards,
        expected_act_dtype=expected_act_dtype,
    )


def check_mlx_paths(
    paths: Sequence[str | Path],
    *,
    flag_bool_sdpa_as_error: bool = False,
    check_loop_hazards: bool = True,
    expected_act_dtype: str | None = None,
) -> GuardReport:
    report = GuardReport()
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for py in sorted(path.rglob("*.py")):
                report.extend(
                    check_mlx_file(
                        py,
                        flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
                        check_loop_hazards=check_loop_hazards,
                        expected_act_dtype=expected_act_dtype,
                    )
                )
        else:
            report.extend(
                check_mlx_file(
                    path,
                    flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
                    check_loop_hazards=check_loop_hazards,
                    expected_act_dtype=expected_act_dtype,
                )
            )
    return report


# ---------------------------------------------------------------------------
# CoreML MIL Program Static IR Inspector
# ---------------------------------------------------------------------------

def _mil_dtype_str(var: Any) -> str:
    from coremltools.converters.mil.mil import types

    if hasattr(var, "dtype"):
        return str(types.builtin_to_string(var.dtype))
    return ""


def _walk_mil_block(
    block: Any,
    *,
    fn_name: str,
    model_name: str,
    allow_fp32_ops: set[str],
    flag_add_fp16_cast: bool,
    findings: list[GuardFinding],
) -> None:
    for idx, op in enumerate(block.operations):
        op_type = str(getattr(op, "op_type", ""))
        op_name = str(getattr(op, "name", f"op_{idx}"))

        for sub_block in getattr(op, "blocks", ()) or ():
            _walk_mil_block(
                sub_block,
                fn_name=fn_name,
                model_name=model_name,
                allow_fp32_ops=allow_fp32_ops,
                flag_add_fp16_cast=flag_add_fp16_cast,
                findings=findings,
            )

        if op_type == "const":
            continue

        if flag_add_fp16_cast and (op_name.endswith("_cast_fp16") or any(str(o.name).endswith("_to_fp16") for o in op.outputs)):
            findings.append(
                GuardFinding(
                    rule_id="COREML_ADD_FP16_CAST_ARTIFACT",
                    severity="error",
                    file=model_name,
                    line=idx,
                    symbol=f"{fn_name}.{op_name}",
                    message=f"Op `{op_type}` (`{op_name}`) carries `common::add_fp16_cast` pass artifact; build MIL program natively in `types.fp16` or explicitly skip elementwise FP32 ops.",
                )
            )

        if op_type == "cast":
            out_dts = [_mil_dtype_str(o) for o in op.outputs]
            if "fp32" in out_dts:
                # Permit fp32 cast only if every consumer op is in allow_fp32_ops
                consumers_ok = bool(allow_fp32_ops) and all(
                    all(str(c_op.op_type) in allow_fp32_ops for c_op in getattr(o, "child_ops", ()))
                    and len(getattr(o, "child_ops", ())) > 0
                    for o in op.outputs
                )
                if not consumers_ok:
                    findings.append(
                        GuardFinding(
                            rule_id="COREML_UNINTENDED_FP32_CAST",
                            severity="error",
                            file=model_name,
                            line=idx,
                            symbol=f"{fn_name}.{op_name}",
                            message=f"Unexpected `cast(..., dtype='fp32')` on op `{op_name}` forces FP32 compute/CPU fallback.",
                        )
                    )
            elif flag_add_fp16_cast and "fp16" in out_dts and op_name.startswith("cast_"):
                findings.append(
                    GuardFinding(
                        rule_id="COREML_ADD_FP16_CAST_ENTRY",
                        severity="error",
                        file=model_name,
                        line=idx,
                        symbol=f"{fn_name}.{op_name}",
                        message=f"Auto-inserted entry `cast` `{op_name}` -> `fp16` indicates FP32 TensorSpec/program rewritten by `add_fp16_cast`.",
                    )
                )
            continue

        # Compute operations
        if op_type not in allow_fp32_ops:
            for o in op.outputs:
                odt = _mil_dtype_str(o)
                if odt == "fp32":
                    findings.append(
                        GuardFinding(
                            rule_id="COREML_FP32_COMPUTE_OP",
                            severity="error",
                            file=model_name,
                            line=idx,
                            symbol=f"{fn_name}.{op_name}",
                            message=f"Compute op `{op_type}` outputs `fp32` tensor `{o.name}` instead of `fp16`.",
                        )
                    )

        if op_type == "scaled_dot_product_attention" and "attn_mask" in op.inputs:
            mdt = _mil_dtype_str(op.inputs["attn_mask"])
            if mdt and mdt != "fp16":
                findings.append(
                    GuardFinding(
                        rule_id="COREML_SDPA_MASK_NOT_FP16",
                        severity="error",
                        file=model_name,
                        line=idx,
                        symbol=f"{fn_name}.{op_name}",
                        message=f"CoreML `scaled_dot_product_attention` `attn_mask` has dtype `{mdt}` instead of `fp16`.",
                    )
                )


def check_coreml_mil_program(
    program: Any,
    *,
    model_name: str = "<mil_program>",
    require_fp16_io: bool = True,
    allow_fp32_ops: Iterable[str] = (),
    flag_add_fp16_cast: bool = True,
) -> GuardReport:
    """Inspect a `coremltools.converters.mil.Program` for FP32 casts, `add_fp16_cast`, and non-FP16 ops."""
    findings: list[GuardFinding] = []
    allowed = frozenset(allow_fp32_ops)

    for fn_name, fn in program.functions.items():
        if require_fp16_io:
            for inp_name, inp_var in fn.inputs.items():
                dt = _mil_dtype_str(inp_var)
                if dt == "fp32":
                    findings.append(
                        GuardFinding(
                            rule_id="COREML_NON_FP16_INPUT",
                            severity="error",
                            file=model_name,
                            line=0,
                            symbol=f"{fn_name}.{inp_name}",
                            message=f"MIL function input `{inp_name}` has dtype `fp32`; expected `fp16`.",
                        )
                    )
            for out_var in fn.outputs:
                dt = _mil_dtype_str(out_var)
                if dt == "fp32":
                    findings.append(
                        GuardFinding(
                            rule_id="COREML_NON_FP16_OUTPUT",
                            severity="error",
                            file=model_name,
                            line=0,
                            symbol=f"{fn_name}.{out_var.name}",
                            message=f"MIL function output `{out_var.name}` has dtype `fp32`; expected `fp16`.",
                        )
                    )

        _walk_mil_block(
            fn,
            fn_name=str(fn_name),
            model_name=model_name,
            allow_fp32_ops=allowed,
            flag_add_fp16_cast=flag_add_fp16_cast,
            findings=findings,
        )

    return GuardReport(findings=findings)


def check_coreml_model(
    mlmodel: Any,
    *,
    model_name: str = "<mlmodel>",
    require_fp16_io: bool = True,
    allow_fp32_ops: Iterable[str] = (),
    flag_add_fp16_cast: bool = True,
) -> GuardReport:
    """Inspect a converted `ct.models.MLModel` (`_mil_program`) statically without loading onto ANE/GPU."""
    prog = getattr(mlmodel, "_mil_program", None)
    if prog is None:
        raise ValueError("MLModel has no `_mil_program`; convert with `convert_to='mlprogram'` and `skip_model_load=True`.")
    return check_coreml_mil_program(
        prog,
        model_name=model_name,
        require_fp16_io=require_fp16_io,
        allow_fp32_ops=allow_fp32_ops,
        flag_add_fp16_cast=flag_add_fp16_cast,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static MLX / Metal IR & dtype regression guard")
    parser.add_argument("paths", nargs="+", help="Python files or directories to scan")
    parser.add_argument("--strict-bool-mask", action="store_true", help="Treat bool SDPA masks as errors")
    parser.add_argument("--act-dtype", choices=("float16", "bfloat16"), default=None, help="Expected activation dtype")
    args = parser.parse_args(argv)

    rep = check_mlx_paths(
        args.paths,
        flag_bool_sdpa_as_error=args.strict_bool_mask,
        expected_act_dtype=args.act_dtype,
    )
    for f in rep.findings:
        print(f.format())
    print(f"Summary: {len(rep.errors)} errors, {len(rep.warnings)} warnings")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
