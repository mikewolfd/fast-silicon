"""Model configuration for pplx-embed-v2-context-0.6b."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MODEL_NAME = "perplexity-ai/pplx-embed-v2-context-0.6b"

_DEFAULT_LAYER_TYPES: tuple[str, ...] = tuple(
    "linear_attention" if i % 2 == 0 else "full_attention" for i in range(12)
)


@dataclass(frozen=True)
class PplxConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    intermediate_size: int = 3584
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_dim: int = 64
    rope_theta: float = 10_000_000.0
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    eos_token_id: int = 248044
    pad_token_id: int = 248044
    query_prefix_token_id: int = 248077
    document_prefix_token_id: int = 248078
    chunk_sep_token_id: int = 248079
    query_prefix: str = "[Q] "
    document_prefix: str = "[D] "
    boundary_marker: str = "<|chunk_sep|>"
    embedding_dim: int = 1024
    query_length: int = 262144
    document_length: int = 262144
    full_attention_interval: int = 2
    layer_types: tuple[str, ...] = _DEFAULT_LAYER_TYPES

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) != num_hidden_layers ({self.num_hidden_layers})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must divide by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        expected_rope_dim = int(round(self.head_dim * self.partial_rotary_factor))
        if self.rope_dim != expected_rope_dim:
            raise ValueError(
                f"rope_dim ({self.rope_dim}) != round(head_dim * partial_rotary_factor) ({expected_rope_dim})"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PplxConfig:
        tc = data.get("text_config", data)
        num_layers = int(tc.get("num_hidden_layers", 12))
        default_types = tuple("linear_attention" if i % 2 == 0 else "full_attention" for i in range(num_layers))
        layer_types = tuple(tc.get("layer_types", default_types))

        # Upstream config.json sets full_attention_interval=4 even though layer_types alternates every 2 layers.
        # Derive the true interval whenever alternating layer_types are present.
        raw_interval = int(tc.get("full_attention_interval", 2))
        if len(layer_types) >= 2 and all(
            lt == ("full_attention" if idx % 2 == 1 else "linear_attention")
            for idx, lt in enumerate(layer_types)
        ):
            full_interval = 2
        else:
            full_interval = raw_interval

        rope_params = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        head_dim = int(tc.get("head_dim", 256))
        partial_rotary_factor = float(rope_params.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
        rope_theta = float(rope_params.get("rope_theta", tc.get("rope_theta", 10_000_000.0)))
        raw_mrope = rope_params.get("mrope_section") or (11, 11, 10)
        mrope_section = (int(raw_mrope[0]), int(raw_mrope[1]), int(raw_mrope[2]))
        rope_dim = int(round(head_dim * partial_rotary_factor))

        raw_eos = tc.get("eos_token_id")
        if raw_eos is None:
            raw_eos = data.get("eos_token_id")
        eos_token_id = 248044 if raw_eos is None else int(raw_eos)

        raw_pad = tc.get("pad_token_id")
        if raw_pad is None:
            raw_pad = data.get("pad_token_id")
        pad_token_id = eos_token_id if raw_pad is None else int(raw_pad)

        return cls(
            hidden_size=int(tc.get("hidden_size", 1024)),
            num_hidden_layers=num_layers,
            intermediate_size=int(tc.get("intermediate_size", 3584)),
            num_attention_heads=int(tc.get("num_attention_heads", 8)),
            num_key_value_heads=int(tc.get("num_key_value_heads", 2)),
            head_dim=head_dim,
            partial_rotary_factor=partial_rotary_factor,
            rope_dim=rope_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            linear_conv_kernel_dim=int(tc.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(tc.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(tc.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(tc.get("linear_num_key_heads", 16)),
            linear_num_value_heads=int(tc.get("linear_num_value_heads", 16)),
            rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
            vocab_size=int(tc.get("vocab_size", 248320)),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            query_prefix_token_id=int(data.get("query_prefix_token_id", 248077)),
            document_prefix_token_id=int(data.get("document_prefix_token_id", 248078)),
            chunk_sep_token_id=int(data.get("chunk_sep_token_id", 248079)),
            query_prefix=str(data.get("query_prefix", "[Q] ")),
            document_prefix=str(data.get("document_prefix", "[D] ")),
            boundary_marker=str(data.get("boundary_marker", "<|chunk_sep|>")),
            embedding_dim=int(data.get("embedding_dim", tc.get("hidden_size", 1024))),
            query_length=int(data.get("query_length", 262144)),
            document_length=int(data.get("document_length", 262144)),
            full_attention_interval=full_interval,
            layer_types=layer_types,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PplxConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


DEFAULT_PPLX_CONFIG = PplxConfig()

__all__ = ["DEFAULT_MODEL_NAME", "DEFAULT_PPLX_CONFIG", "PplxConfig"]
