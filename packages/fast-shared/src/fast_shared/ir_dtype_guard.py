"""Static MLX Python / Metal 4 kernel and CoreML MIL Program IR dtype & regression guard.

Targets known Apple Silicon M4 (Apple9) / M5+ (Apple10/NAX) silent performance & accuracy regressions:
  1. Integer (`int32`/`int64`/`bool`) or `float32` SDPA masks and Q/K/V-vs-mask dtype mismatches.
  2. Mixed `bfloat16` x `float16` or `bf16/fp16` x `float32` scale/bias/weight implicit upcasts,
     including unpacked `mx.quantize(...)` tuples and `gather_qmm` missing `lhs_indices` / `sorted_indices=True`.
  3. Unintended `mx.float32` outputs, mutable `device float*` pointers, Metal 4 `tensor<device ...>`
     mismatches, and `const device` half/bfloat mismatches in custom Metal kernels.
  4. `0D` scalar arrays (`reshape(())`, `mx.zeros(())`, `mx.ones(())`, `mx.full((), ...)`, scalar `mx.array(x)`)
     and non-contiguous transposed views (`.T`, `mx.transpose`) passed to `mx.fast.metal_kernel`.
  5. `mx.compile` inside loops, synchronous offset/layer/while-loop `mx.eval`, and implicit host-GPU syncs.
  6. CoreML MIL `Program` `cast(..., dtype="fp32")`, `common::add_fp16_cast` pass artifacts,
     non-FP16 I/O, FP32 compute ops, non-FP16 SDPA masks, and `mb.matmul` ANE fallback warnings.
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

# Metal pointer patterns (Metal 3 & Metal 4)
_DEVICE_FP32_PTR_RE = re.compile(r"\bdevice\s+(?:const\s+)?(?:packed_)?float(?:[234])?\s*\*")
_DEVICE_BF16_PTR_RE = re.compile(r"\bdevice\s+(?:const\s+)?(?:packed_)?bfloat(?:[234])?\s*\*")
_DEVICE_FP16_PTR_RE = re.compile(r"\bdevice\s+(?:const\s+)?(?:packed_)?half(?:[234])?\s*\*")

# Metal 4 <metal_tensor> declarations
_TENSOR_FP32_RE = re.compile(r"\btensor\s*<\s*(?:const\s+)?device\s+(?:const\s+)?float\b")
_TENSOR_BF16_RE = re.compile(r"\btensor\s*<\s*(?:const\s+)?device\s+(?:const\s+)?bfloat\b")
_TENSOR_FP16_RE = re.compile(r"\btensor\s*<\s*(?:const\s+)?device\s+(?:const\s+)?half\b")


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


def _is_const_qualified_tensor(match_text: str) -> bool:
    """Return True when a Metal 4 `tensor<...>` declaration is explicitly const-qualified."""
    return bool(re.search(r"<\s*const\s+device\b|\bdevice\s+const\b", match_text))


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


def _extract_str_list(
    node: ast.AST | None,
    str_consts: dict[str, str],
    list_consts: dict[str, list[str]] | None = None,
) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Name) and list_consts and node.id in list_consts:
        return list(list_consts[node.id])
    if isinstance(node, ast.IfExp):
        b_list = _extract_str_list(node.body, str_consts, list_consts)
        o_list = _extract_str_list(node.orelse, str_consts, list_consts)
        return o_list if len(o_list) >= len(b_list) else b_list
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    out: list[str] = []
    for elt in node.elts:
        s = _extract_const_str(elt, str_consts)
        if s is not None:
            out.append(s)
    return out


def _is_empty_tuple(node: ast.AST) -> bool:
    return isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) == 0


class _MlxAstChecker(ast.NodeVisitor):
    def __init__(
        self,
        *,
        filename: str,
        source: str,
        flag_bool_sdpa_as_error: bool = False,
        check_loop_hazards: bool = True,
        expected_act_dtype: str | None = None,
        check_coreml_ast: bool = False,
    ) -> None:
        self.filename = filename
        self.source = source
        self.lines = source.splitlines()
        self.flag_bool_sdpa_as_error = flag_bool_sdpa_as_error
        self.check_loop_hazards = check_loop_hazards
        self.expected_act_dtype = expected_act_dtype
        self.check_coreml_ast = check_coreml_ast
        self.findings: list[GuardFinding] = []

        self.scope_stack: list[str] = ["<module>"]
        self.dtype_stack: list[dict[str, str]] = [{}]
        self.class_attrs_stack: list[dict[str, str]] = []
        self.zero_d_stack: list[set[str]] = [set()]
        self.noncontig_stack: list[set[str]] = [set()]
        self.list_consts_stack: list[dict[str, list[str]]] = [{}]
        self.str_consts: dict[str, str] = {}
        self.metal_kernels: dict[str, dict[str, Any]] = {}
        self.metal_kernel_factories: dict[str, dict[str, Any]] = {}
        self._checked_metal_defs: set[int] = set()
        self._assign_lhs_names: dict[int, list[str]] = {}
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

    @property
    def _noncontig(self) -> set[str]:
        return self.noncontig_stack[-1]

    @property
    def _list_consts(self) -> dict[str, list[str]]:
        return self.list_consts_stack[-1]

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
            if node.attr == "T":
                return self._infer_expr_dtype(node.value)
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
            if chain in ("mx.contiguous", "mx.transpose", "mx.swapaxes") and node.args:
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
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("reshape", "transpose", "squeeze"):
                return self._infer_expr_dtype(node.func.value)
        return None

    def _is_zero_d_expr(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id in self._zero_d:
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "reshape" and node.args:
                if _is_empty_tuple(node.args[0]):
                    return True
            chain = _attr_chain(node.func)
            if chain in ("mx.zeros", "mx.ones", "mx.empty", "mx.full") and node.args:
                if _is_empty_tuple(node.args[0]):
                    return True
            if chain == "mx.array" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, (int, float)):
                    return True
                if isinstance(arg0, ast.UnaryOp) and isinstance(arg0.operand, ast.Constant):
                    return True
            if chain == "mx.contiguous" and node.args:
                return self._is_zero_d_expr(node.args[0])
        return False

    def _is_noncontiguous_expr(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id in self._noncontig:
            return True
        if isinstance(node, ast.Attribute) and node.attr == "T":
            return True
        if isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            if chain == "mx.contiguous":
                return False
            if chain in ("mx.transpose", "mx.swapaxes"):
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("transpose", "swapaxes"):
                return True
        return False

    def _record_metal_kernel_def(self, target_name: str, call: ast.Call) -> None:
        in_names = _extract_str_list(_kw_arg(call, "input_names"), self.str_consts, self._list_consts)
        out_names = _extract_str_list(_kw_arg(call, "output_names"), self.str_consts, self._list_consts)
        src_node = _kw_arg(call, "source")
        hdr_node = _kw_arg(call, "header")
        src_text = _extract_const_str(src_node, self.str_consts) if src_node is not None else ""
        hdr_text = _extract_const_str(hdr_node, self.str_consts) if hdr_node is not None else ""
        info = {
            "name": target_name,
            "input_names": in_names,
            "output_names": out_names,
            "source": src_text or "",
            "header": hdr_text or "",
            "lineno": call.lineno,
        }
        self.metal_kernels[target_name] = info
        if len(self.scope_stack) >= 2:
            self.metal_kernel_factories[self.scope_stack[-1]] = info
        self._checked_metal_defs.add(id(call))
        self._check_metal_kernel_source(target_name, out_names, src_text or "", hdr_text or "", call.lineno)

    def _check_metal_kernel_source(
        self,
        kern_name: str,
        out_names: list[str],
        src_text: str,
        hdr_text: str,
        lineno: int,
    ) -> None:
        if not src_text or self._line_has_ignore(lineno, "fp32"):
            return
        combined = f"{hdr_text}\n{src_text}" if hdr_text else src_text

        # Check Metal 4 <metal_tensor> header completeness when tensor<device ...> is used
        if (
            ("tensor<device" in src_text or "tensor<const device" in src_text)
            and hdr_text
            and "<metal_tensor>" not in hdr_text
            and not self._line_has_ignore(lineno, "metal-tensor-header")
        ):
            self._emit(
                "METAL_TENSOR_MISSING_HEADER",
                "error",
                lineno,
                f"Metal kernel `{kern_name}` uses Metal 4 `tensor<device ...>` without `#include <metal_tensor>` in `header`.",
            )

        # Flag mutable `device float*` / `device float4*` / `const_cast<device float*>` / `tensor<device float, ...>`
        allowed_non_stats = [o for o in out_names if o not in _ALLOWED_FP32_STATS_OUTPUTS]
        if allowed_non_stats:
            emitted_fp32 = False
            for m in _DEVICE_FP32_PTR_RE.finditer(src_text):
                if not _is_const_qualified_ptr(src_text, m.start(), m.group(0)):
                    self._emit(
                        "METAL_FP32_OUTPUT_PTR",
                        "error",
                        lineno,
                        f"Metal kernel `{kern_name}` uses mutable `{m.group(0).strip()}` pointer for non-stats outputs {allowed_non_stats}; expected `half`/`bfloat`.",
                    )
                    emitted_fp32 = True
                    break
            if not emitted_fp32:
                for m in _TENSOR_FP32_RE.finditer(src_text):
                    if not _is_const_qualified_tensor(m.group(0)):
                        self._emit(
                            "METAL_FP32_OUTPUT_PTR",
                            "error",
                            lineno,
                            f"Metal kernel `{kern_name}` uses mutable Metal 4 `{m.group(0).strip()}` tensor for non-stats outputs {allowed_non_stats}; expected `half`/`bfloat`.",
                        )
                        break

        act_dt = self._cur_act_dtype()
        kname_low = kern_name.lower()
        is_explicit_conv = any(tok in kname_low for tok in ("bf16_to_f16", "f16_to_bf16", "to_f16", "to_fp16", "to_bf16", "cast"))
        if act_dt == "float16" and not is_explicit_conv and not self._line_has_ignore(lineno, "bf16-ptr"):
            emitted_bf16 = False
            for m in _DEVICE_BF16_PTR_RE.finditer(src_text):
                self._emit(
                    "METAL_BF16_PTR_IN_FP16_KERNEL",
                    "error",
                    lineno,
                    f"Metal kernel `{kern_name}` uses `{m.group(0).strip()}` pointer inside FP16 (`float16`) context; expected `half`/`half4`.",
                )
                emitted_bf16 = True
                break
            if not emitted_bf16:
                for m in _TENSOR_BF16_RE.finditer(combined):
                    self._emit(
                        "METAL_BF16_PTR_IN_FP16_KERNEL",
                        "error",
                        lineno,
                        f"Metal kernel `{kern_name}` uses Metal 4 `{m.group(0).strip()}` tensor inside FP16 (`float16`) context; expected `half`.",
                    )
                    break
        elif act_dt == "bfloat16" and not is_explicit_conv and not self._line_has_ignore(lineno, "fp16-ptr"):
            # Only flag explicit `tensor<device half` when no explicit f16 tag in kernel name
            if "f16" not in kname_low and "fp16" not in kname_low:
                for m in _TENSOR_FP16_RE.finditer(src_text):
                    self._emit(
                        "METAL_FP16_PTR_IN_BF16_KERNEL",
                        "error",
                        lineno,
                        f"Metal kernel `{kern_name}` uses Metal 4 `{m.group(0).strip()}` tensor inside BF16 (`bfloat16`) context; expected `bfloat`.",
                    )
                    break

    def _check_metal_kernel_call(self, call: ast.Call, kern_info: dict[str, Any] | None) -> None:
        out_dtypes_node = _kw_arg(call, "output_dtypes")
        out_names = kern_info["output_names"] if kern_info else []
        in_names = kern_info["input_names"] if kern_info else []
        src_text = kern_info["source"] if kern_info else ""
        kname = str(kern_info.get("name", "")) if kern_info else ""
        sym_dtypes = self._cur_sym_dtypes()

        lhs_names = self._assign_lhs_names.get(id(call), [])
        call_out_dts: list[str | None] = []
        if isinstance(out_dtypes_node, (ast.List, ast.Tuple)):
            for idx, elt in enumerate(out_dtypes_node.elts):
                dt = _mx_dtype_from_node(elt, sym_dtypes)
                call_out_dts.append(dt)
                if dt == "float32" and not self._line_has_ignore(call.lineno, "fp32"):
                    oname = (
                        out_names[idx]
                        if idx < len(out_names)
                        else (lhs_names[idx] if idx < len(lhs_names) else f"out[{idx}]")
                    )
                    if oname not in _ALLOWED_FP32_STATS_OUTPUTS:
                        self._emit(
                            "MLX_METAL_FP32_OUTPUT",
                            "error",
                            call.lineno,
                            f"Custom Metal kernel output `{oname}` uses `mx.float32`; use `mx.float16`/`mx.bfloat16` (only log-sum-exp stats outputs like `ml_part` may use `float32`).",
                        )

        inputs_node = _kw_arg(call, "inputs")
        call_in_dts: list[str | None] = []
        if isinstance(inputs_node, (ast.List, ast.Tuple)):
            for elt in inputs_node.elts:
                call_in_dts.append(self._infer_expr_dtype(elt))

        # Cross-check kernel Metal pointer & Metal 4 tensor precision against explicit call `output_dtypes`
        kname_low = kname.lower()
        is_explicit_conv = any(tok in kname_low for tok in ("bf16_to_f16", "f16_to_bf16", "to_f16", "to_fp16", "to_bf16", "cast"))
        if src_text and call_out_dts and not is_explicit_conv and not self._line_has_ignore(call.lineno, "ptr-mismatch"):
            if "float16" in call_out_dts and "bfloat16" not in call_out_dts and self._cur_act_dtype() != "float16":
                has_bf16_in = "bfloat16" in call_in_dts
                emitted = False
                for m in _DEVICE_BF16_PTR_RE.finditer(src_text):
                    if not has_bf16_in or not _is_const_qualified_ptr(src_text, m.start(), m.group(0)):
                        self._emit(
                            "METAL_BF16_PTR_IN_FP16_KERNEL",
                            "error",
                            call.lineno,
                            f"Metal kernel outputs `float16` `{call_out_dts}` but source uses `{m.group(0).strip()}` pointer.",
                        )
                        emitted = True
                        break
                if not emitted and not has_bf16_in:
                    for m in _TENSOR_BF16_RE.finditer(src_text):
                        self._emit(
                            "METAL_BF16_PTR_IN_FP16_KERNEL",
                            "error",
                            call.lineno,
                            f"Metal kernel outputs `float16` `{call_out_dts}` but source uses Metal 4 `{m.group(0).strip()}` tensor.",
                        )
                        break
            elif "bfloat16" in call_out_dts and "float16" not in call_out_dts:
                has_fp16_in = "float16" in call_in_dts
                emitted = False
                for m in _DEVICE_FP16_PTR_RE.finditer(src_text):
                    if not has_fp16_in or not _is_const_qualified_ptr(src_text, m.start(), m.group(0)):
                        self._emit(
                            "METAL_FP16_PTR_IN_BF16_KERNEL",
                            "error",
                            call.lineno,
                            f"Metal kernel outputs `bfloat16` `{call_out_dts}` but source uses `{m.group(0).strip()}` pointer.",
                        )
                        emitted = True
                        break
                if not emitted and not has_fp16_in:
                    for m in _TENSOR_FP16_RE.finditer(src_text):
                        self._emit(
                            "METAL_FP16_PTR_IN_BF16_KERNEL",
                            "error",
                            call.lineno,
                            f"Metal kernel outputs `bfloat16` `{call_out_dts}` but source uses Metal 4 `{m.group(0).strip()}` tensor.",
                        )
                        break

        if isinstance(inputs_node, (ast.List, ast.Tuple)):
            for idx, elt in enumerate(inputs_node.elts):
                iname = in_names[idx] if idx < len(in_names) else f"input[{idx}]"
                if self._is_zero_d_expr(elt):
                    if not src_text or f"{iname}[0]" in src_text or f"{iname}[" in src_text or "*" in src_text:
                        self._emit(
                            "MLX_METAL_0D_SCALAR",
                            "error",
                            call.lineno,
                            f"0D scalar passed as `{iname}` to Metal kernel; MLX compiles 0D arrays as value scalars rather than `const device T*` (use `.reshape((1,))`).",
                        )
                if self._is_noncontiguous_expr(elt) and not self._line_has_ignore(call.lineno, "noncontiguous"):
                    self._emit(
                        "MLX_METAL_NONCONTIGUOUS_INPUT",
                        "error",
                        call.lineno,
                        f"Transposed/non-contiguous view passed as `{iname}` to `mx.fast.metal_kernel`; materialize with `mx.contiguous(...)` first to avoid stride/copy traps.",
                    )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope_stack.append(node.name)
        parent_dtypes = dict(self._dtypes)
        self.dtype_stack.append(parent_dtypes)
        self.zero_d_stack.append(set())
        self.noncontig_stack.append(set())
        self.list_consts_stack.append(dict(self._list_consts))
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.arg in ("dt", "dtype") and arg.annotation:
                dt = _mx_dtype_from_node(arg.annotation, parent_dtypes)
                if dt:
                    self._dtypes[arg.arg] = dt
        self.generic_visit(node)
        self.list_consts_stack.pop()
        self.noncontig_stack.pop()
        self.zero_d_stack.pop()
        self.dtype_stack.pop()
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope_stack.append(node.name)
        self.dtype_stack.append(dict(self._dtypes))
        self.zero_d_stack.append(set())
        self.noncontig_stack.append(set())
        self.list_consts_stack.append(dict(self._list_consts))
        self.generic_visit(node)
        self.list_consts_stack.pop()
        self.noncontig_stack.pop()
        self.zero_d_stack.pop()
        self.dtype_stack.pop()
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_stack.append(node.name)
        cls_attrs: dict[str, str] = {}
        if self.expected_act_dtype is not None:
            cls_attrs["self.act_dtype"] = self.expected_act_dtype
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
        if len(node.targets) == 1:
            tgt0 = node.targets[0]
            if isinstance(tgt0, (ast.Tuple, ast.List)) and isinstance(node.value, ast.Call):
                self._assign_lhs_names[id(node.value)] = [
                    elt.id if isinstance(elt, ast.Name) else "" for elt in tgt0.elts
                ]
            elif isinstance(tgt0, ast.Name):
                sl = _extract_str_list(node.value, self.str_consts, self._list_consts)
                if sl:
                    self._list_consts[tgt0.id] = sl
                if isinstance(node.value, ast.Call) and _attr_chain(node.value.func) == "mx.fast.metal_kernel":
                    self._record_metal_kernel_def(tgt0.id, node.value)
        self.generic_visit(node)
        if len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(node.value, ast.Call):
                fchain = _attr_chain(node.value.func)
                if fchain == "mx.quantize":
                    # Track unpacked `(wq, scales[, biases]) = mx.quantize(w, ...)` dtypes accurately!
                    w_node = _kw_arg(node.value, "w", 0)
                    w_dt = self._infer_expr_dtype(w_node)
                    mode_node = _kw_arg(node.value, "mode")
                    mode_str = _extract_const_str(mode_node, self.str_consts) if mode_node is not None else None
                    is_e8m0 = bool(mode_str and any(tag in mode_str.lower() for tag in ("mxfp", "nvfp")))
                    for idx, elt in enumerate(tgt.elts):
                        if not isinstance(elt, ast.Name):
                            continue
                        if idx == 0:
                            self._dtypes[elt.id] = "uint32"
                        elif idx == 1:
                            if is_e8m0:
                                self._dtypes[elt.id] = "uint8"
                            elif w_dt is not None:
                                self._dtypes[elt.id] = w_dt
                        elif idx == 2 and w_dt is not None:
                            self._dtypes[elt.id] = w_dt
                    return
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
                if self._is_noncontiguous_expr(node.value):
                    self._noncontig.add(tgt.id)
                else:
                    self._noncontig.discard(tgt.id)
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
            if self._is_noncontiguous_expr(node.value):
                self._noncontig.add(node.target.id)
            else:
                self._noncontig.discard(node.target.id)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.generic_visit(node)
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

    def _is_setup_or_bench_fn(self) -> bool:
        fn_name = self.scope_stack[-1]
        return fn_name.startswith(
            ("attach_", "__init__", "warmup", "make_", "build_", "export_", "main", "bench", "_bench", "measure_", "test_")
        )

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        chain = _attr_chain(node.func)

        # 0. Optional CoreML MIL Python builder AST check (`mb.matmul` ANE fallback warning)
        if self.check_coreml_ast and chain == "mb.matmul" and not self._line_has_ignore(node.lineno, "coreml-matmul"):
            if not Path(self.filename).name.startswith("test_"):
                self._emit(
                    "COREML_PY_MATMUL_ANE",
                    "warning",
                    node.lineno,
                    "`mb.matmul` frequently falls back or runs below peak on ANE; prefer 4D NCHW `mb.conv` (1x1) on static buckets for ANE saturation.",
                )

        # 1. SDPA mask & Q/K/V dtype checks
        if chain in ("mx.fast.scaled_dot_product_attention", "scaled_dot_product_attention"):
            q_dt = self._infer_expr_dtype(_kw_arg(node, "q", 0))
            k_dt = self._infer_expr_dtype(_kw_arg(node, "k", 1))
            v_dt = self._infer_expr_dtype(_kw_arg(node, "v", 2))
            qkv_dts = [d for d in (q_dt, k_dt, v_dt) if d in _FLOAT_DTYPES]
            if len(set(qkv_dts)) > 1 and not self._line_has_ignore(node.lineno, "sdpa-dtype"):
                self._emit(
                    "MLX_SDPA_DTYPE_MISMATCH",
                    "error",
                    node.lineno,
                    f"SDPA Q/K/V have mismatched float dtypes ({q_dt}, {k_dt}, {v_dt}), forcing implicit FP32 promotion.",
                )

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
                        "Boolean SDPA mask detected; prefer additive FP16 (`-65504.0`) or explicit FP16/BF16 mask (verify MLX 0.32.3+ M5 NAX array-mask behavior before keeping bool).",
                    )
                elif (
                    mdt in _LOW_FLOAT_DTYPES
                    and q_dt in _LOW_FLOAT_DTYPES
                    and mdt != q_dt
                    and not self._line_has_ignore(node.lineno, "sdpa-dtype")
                ):
                    self._emit(
                        "MLX_SDPA_DTYPE_MISMATCH",
                        "error",
                        node.lineno,
                        f"SDPA mask dtype (`{mdt}`) mismatches query dtype (`{q_dt}`), causing silent `FP32` upcast.",
                    )

        # 2. Metal kernel definitions and invocations
        if chain == "mx.fast.metal_kernel" and _kw_arg(node, "output_dtypes") is None and id(node) not in self._checked_metal_defs:
            out_names = _extract_str_list(_kw_arg(node, "output_names"), self.str_consts)
            src_node = _kw_arg(node, "source")
            hdr_node = _kw_arg(node, "header")
            src_text = _extract_const_str(src_node, self.str_consts) if src_node is not None else ""
            hdr_text = _extract_const_str(hdr_node, self.str_consts) if hdr_node is not None else ""
            if src_text and isinstance(node.func, ast.Attribute):
                self._checked_metal_defs.add(id(node))
                self._check_metal_kernel_source("<inline>", out_names, src_text, hdr_text or "", node.lineno)

        if _kw_arg(node, "output_dtypes") is not None and _kw_arg(node, "inputs") is not None:
            kern_info = None
            if isinstance(node.func, ast.Name):
                kern_info = self.metal_kernels.get(node.func.id)
            elif isinstance(node.func, ast.Call) and _attr_chain(node.func.func) == "mx.fast.metal_kernel":
                kern_info = {
                    "name": "<inline>",
                    "input_names": _extract_str_list(_kw_arg(node.func, "input_names"), self.str_consts),
                    "output_names": _extract_str_list(_kw_arg(node.func, "output_names"), self.str_consts),
                    "source": _extract_const_str(_kw_arg(node.func, "source"), self.str_consts) or "",
                    "header": _extract_const_str(_kw_arg(node.func, "header"), self.str_consts) or "",
                }
            self._check_metal_kernel_call(node, kern_info)

        # 3. Quantized matmul / gather_qmm mixed dtype (`scales` & `biases`), missing `lhs_indices`, & `sorted_indices`
        if chain in ("mx.gather_qmm", "mx.quantized_matmul"):
            x_dt = self._infer_expr_dtype(_kw_arg(node, "x", 0))
            s_dt = self._infer_expr_dtype(_kw_arg(node, "scales", 2))
            b_dt = self._infer_expr_dtype(_kw_arg(node, "biases", 3))
            if not self._line_has_ignore(node.lineno, "mixed-dtype"):
                for param_label, p_dt in (("scales", s_dt), ("biases", b_dt)):
                    if {x_dt, p_dt} == {"bfloat16", "float16"} or (
                        "float32" in (x_dt, p_dt) and (x_dt in _LOW_FLOAT_DTYPES or p_dt in _LOW_FLOAT_DTYPES)
                    ):
                        self._emit(
                            "MLX_IMPLICIT_FP32_SCALE_UPCAST",
                            "error",
                            node.lineno,
                            f"`{chain}` mixes activation `{x_dt}` and `{param_label}` `{p_dt}`; align scales/biases to activation dtype to prevent silent `FP32` promotion.",
                        )
                        break
            if chain == "mx.gather_qmm" and _kw_arg(node, "rhs_indices") is not None:
                if _kw_arg(node, "lhs_indices") is None and not self._line_has_ignore(node.lineno, "lhs-indices"):
                    self._emit(
                        "MLX_GATHER_QMM_MISSING_LHS",
                        "error",
                        node.lineno,
                        "`mx.gather_qmm` with `rhs_indices` and missing `lhs_indices` silently broadcasts every token across all experts.",
                    )
                sorted_node = _kw_arg(node, "sorted_indices")
                is_unsorted = sorted_node is None or (
                    isinstance(sorted_node, ast.Constant) and sorted_node.value is False
                )
                if is_unsorted and not self._line_has_ignore(node.lineno, "unsorted-qmm"):
                    self._emit(
                        "MLX_GATHER_QMM_UNSORTED_INDICES",
                        "warning",
                        node.lineno,
                        "`mx.gather_qmm` omits `sorted_indices=True`; for M >= 2 batched MoE routing, sort expert indices with `mx.argsort` and pass `sorted_indices=True` (+15-27%).",
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

        # 4. Loop hazards (`mx.compile` inside loop; sync `mx.eval` / implicit host syncs inside loops)
        if self.check_loop_hazards and self.loop_stack:
            is_setup_or_bench = self._is_setup_or_bench_fn()
            if chain == "mx.compile" and not self._line_has_ignore(node.lineno, "compile"):
                self._emit(
                    "MLX_COMPILE_IN_LOOP",
                    "error",
                    node.lineno,
                    "`mx.compile` called inside a loop triggers repeated graph recompilation overhead; hoist outside the loop.",
                )
            elif chain == "mx.eval" and not self._line_has_ignore(node.lineno, "eval"):
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
                elif not is_setup_or_bench and any(isinstance(lp, ast.While) for lp in self.loop_stack):
                    self._emit(
                        "MLX_SYNC_EVAL_IN_WHILE_LOOP",
                        "warning",
                        node.lineno,
                        "Synchronous `mx.eval` inside `while` loop can stall device pipeline; consider `mx.async_eval` for pipelined decode steps.",
                    )
            elif not is_setup_or_bench and not self._line_has_ignore(node.lineno, "sync"):
                in_hot_loop = any(self._is_layer_loop(lp) or isinstance(lp, ast.While) for lp in self.loop_stack)
                if in_hot_loop:
                    # Implicit host-GPU sync on tracked MLX arrays (`.item()`, `.tolist()`, `float(x)`, `int(x)`, `np.asarray(x)`)
                    sync_target: ast.AST | None = None
                    sync_desc = ""
                    if isinstance(node.func, ast.Attribute) and node.func.attr in ("item", "tolist") and not node.args:
                        sync_target = node.func.value
                        sync_desc = f".{node.func.attr}()"
                    elif isinstance(node.func, ast.Name) and node.func.id in ("float", "int") and len(node.args) == 1:
                        sync_target = node.args[0]
                        sync_desc = f"{node.func.id}(...)"
                    elif chain in ("np.asarray", "np.array") and node.args:
                        sync_target = node.args[0]
                        sync_desc = f"{chain}(...)"
                    if sync_target is not None and self._infer_expr_dtype(sync_target) is not None:
                        self._emit(
                            "MLX_IMPLICIT_HOST_SYNC_IN_LOOP",
                            "warning",
                            node.lineno,
                            f"Implicit host-GPU synchronization via `{sync_desc}` on tracked MLX array inside tight loop stalls GPU pipeline.",
                        )


def check_mlx_source(
    source: str,
    *,
    filename: str = "<memory>",
    flag_bool_sdpa_as_error: bool = False,
    check_loop_hazards: bool = True,
    expected_act_dtype: str | None = None,
    check_coreml_ast: bool = False,
) -> GuardReport:
    tree = ast.parse(source, filename=filename)
    checker = _MlxAstChecker(
        filename=filename,
        source=source,
        flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
        check_loop_hazards=check_loop_hazards,
        expected_act_dtype=expected_act_dtype,
        check_coreml_ast=check_coreml_ast,
    )
    checker.visit(tree)
    return GuardReport(findings=checker.findings)


def check_mlx_file(
    path: str | Path,
    *,
    flag_bool_sdpa_as_error: bool = False,
    check_loop_hazards: bool = True,
    expected_act_dtype: str | None = None,
    check_coreml_ast: bool = False,
) -> GuardReport:
    p = Path(path)
    return check_mlx_source(
        p.read_text(encoding="utf-8"),
        filename=str(p),
        flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
        check_loop_hazards=check_loop_hazards,
        expected_act_dtype=expected_act_dtype,
        check_coreml_ast=check_coreml_ast,
    )


def check_mlx_paths(
    paths: Sequence[str | Path],
    *,
    flag_bool_sdpa_as_error: bool = False,
    check_loop_hazards: bool = True,
    expected_act_dtype: str | None = None,
    check_coreml_ast: bool = False,
) -> GuardReport:
    report = GuardReport()
    for p in paths:
        path = Path(p)
        if path.suffix == ".mlpackage":
            continue
        if path.is_dir():
            for py in sorted(path.rglob("*.py")):
                report.extend(
                    check_mlx_file(
                        py,
                        flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
                        check_loop_hazards=check_loop_hazards,
                        expected_act_dtype=expected_act_dtype,
                        check_coreml_ast=check_coreml_ast,
                    )
                )
        else:
            report.extend(
                check_mlx_file(
                    path,
                    flag_bool_sdpa_as_error=flag_bool_sdpa_as_error,
                    check_loop_hazards=check_loop_hazards,
                    expected_act_dtype=expected_act_dtype,
                    check_coreml_ast=check_coreml_ast,
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
    warn_matmul_ane: bool,
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
                warn_matmul_ane=warn_matmul_ane,
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

        if warn_matmul_ane and op_type == "matmul":
            findings.append(
                GuardFinding(
                    rule_id="COREML_MATMUL_ANE_FALLBACK",
                    severity="warning",
                    file=model_name,
                    line=idx,
                    symbol=f"{fn_name}.{op_name}",
                    message=f"MIL op `matmul` (`{op_name}`) underperforms or falls off ANE; replace with 4D NCHW `conv` (1x1) on static buckets for peak ANE throughput.",
                )
            )

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
                            message=f"Compute op `{op_type}` outputs `fp32` tensor `{o.name}` instead of `fp16`/`bf16`.",
                        )
                    )

        if op_type == "scaled_dot_product_attention" and "attn_mask" in op.inputs:
            mdt = _mil_dtype_str(op.inputs["attn_mask"])
            if mdt and mdt not in ("fp16", "bf16"):
                findings.append(
                    GuardFinding(
                        rule_id="COREML_SDPA_MASK_NOT_FP16",
                        severity="error",
                        file=model_name,
                        line=idx,
                        symbol=f"{fn_name}.{op_name}",
                        message=f"CoreML `scaled_dot_product_attention` `attn_mask` has dtype `{mdt}` instead of `fp16`/`bf16`.",
                    )
                )


def check_coreml_mil_program(
    program: Any,
    *,
    model_name: str = "<mil_program>",
    require_fp16_io: bool = True,
    allow_fp32_ops: Iterable[str] = (),
    flag_add_fp16_cast: bool = True,
    warn_matmul_ane: bool = True,
) -> GuardReport:
    """Inspect a `coremltools.converters.mil.Program` for FP32 casts, `add_fp16_cast`, non-FP16 ops, and `matmul` ANE warnings."""
    findings: list[GuardFinding] = []
    allowed = set(allow_fp32_ops)

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
                            message=f"MIL function input `{inp_name}` has dtype `fp32`; expected `fp16`/`bf16`.",
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
                            message=f"MIL function output `{out_var.name}` has dtype `fp32`; expected `fp16`/`bf16`.",
                        )
                    )

        _walk_mil_block(
            fn,
            fn_name=str(fn_name),
            model_name=model_name,
            allow_fp32_ops=allowed,
            flag_add_fp16_cast=flag_add_fp16_cast,
            warn_matmul_ane=warn_matmul_ane,
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
    warn_matmul_ane: bool = True,
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
        warn_matmul_ane=warn_matmul_ane,
    )


def check_coreml_paths(
    paths: Sequence[str | Path],
    *,
    require_fp16_io: bool = True,
    allow_fp32_ops: Iterable[str] = (),
    flag_add_fp16_cast: bool = True,
    warn_matmul_ane: bool = True,
) -> GuardReport:
    """Scan `.mlpackage` files (or directories containing them) with `check_coreml_model`."""
    report = GuardReport()
    mlpackages: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.suffix == ".mlpackage" and path.exists():
            mlpackages.append(path)
        elif path.is_dir():
            mlpackages.extend(sorted(path.rglob("*.mlpackage")))
    if not mlpackages:
        return report

    try:
        import coremltools as ct
    except ImportError:
        for pkg in mlpackages:
            report.findings.append(
                GuardFinding(
                    rule_id="COREML_TOOLS_MISSING",
                    severity="warning",
                    file=str(pkg),
                    line=0,
                    symbol="<coreml>",
                    message="coremltools is not installed; skipping `.mlpackage` MIL inspection.",
                )
            )
        return report

    for pkg in mlpackages:
        try:
            mlmodel = ct.models.MLModel(str(pkg), skip_model_load=True)
            if getattr(mlmodel, "_mil_program", None) is not None:
                report.extend(
                    check_coreml_model(
                        mlmodel,
                        model_name=str(pkg),
                        require_fp16_io=require_fp16_io,
                        allow_fp32_ops=allow_fp32_ops,
                        flag_add_fp16_cast=flag_add_fp16_cast,
                        warn_matmul_ane=warn_matmul_ane,
                    )
                )
        except Exception as exc:
            report.findings.append(
                GuardFinding(
                    rule_id="COREML_LOAD_ERROR",
                    severity="error",
                    file=str(pkg),
                    line=0,
                    symbol="<coreml>",
                    message=f"Failed to inspect `.mlpackage`: {type(exc).__name__}: {exc}",
                )
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static MLX / Metal 4 / CoreML MIL IR & dtype regression guard")
    parser.add_argument("paths", nargs="+", help="Python files, directories, or `.mlpackage` bundles to scan")
    parser.add_argument("--strict-bool-mask", action="store_true", help="Treat bool SDPA masks as errors")
    parser.add_argument("--act-dtype", choices=("float16", "bfloat16"), default=None, help="Expected activation dtype")
    parser.add_argument(
        "--coreml",
        action="store_true",
        help="Enable CoreML MIL inspection on `.mlpackage` paths and `mb.matmul` ANE checks in Python builders",
    )
    args = parser.parse_args(argv)

    rep = check_mlx_paths(
        args.paths,
        flag_bool_sdpa_as_error=args.strict_bool_mask,
        expected_act_dtype=args.act_dtype,
        check_coreml_ast=args.coreml,
    )
    if args.coreml:
        rep.extend(check_coreml_paths(args.paths))

    for f in rep.findings:
        print(f.format())
    print(f"Summary: {len(rep.errors)} errors, {len(rep.warnings)} warnings")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
