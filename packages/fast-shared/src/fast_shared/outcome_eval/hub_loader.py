"""Pure-Arrow / JSONL HuggingFace dataset loader with immutable SHA pinning and SHA-256 stratified sampling."""

from __future__ import annotations

import csv
import gzip
import hashlib
import html as _html
import json
import os
from pathlib import Path
import re
import tarfile
from typing import Any, Callable, Mapping, Sequence
import urllib.parse
import urllib.request

import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

DEFAULT_CACHE_DIR = Path("~/.cache/fast-silicon/outcome-benchmarks")
DEFAULT_LOCK_FILENAME = "configs/outcome_benchmarks.lock.json"

# Hard license / subset safety exclusions enforced in code regardless of caller args.
FORBIDDEN_LEGALBENCH_SUBSETS: frozenset[str] = frozenset({"maud"})
ALLOWED_LEGALBENCH_SUBSETS: frozenset[str] = frozenset({"cuad", "contractnli", "privacy_qa"})
FORBIDDEN_LONGEMBED_SUBSETS: frozenset[str] = frozenset({"summ_screen_fd"})
ALLOWED_LONGEMBED_SUBSETS: frozenset[str] = frozenset({"2wikimqa", "passkey", "qmsum"})
ALLOWED_REWARDBENCH2_SUBSETS: tuple[str, ...] = ("Factuality", "Precise IF", "Math", "Safety", "Focus")

# Official Larson et al. (EMNLP 2019) 10-domain x 15-intent CLINC150 hierarchy (+ "oos").
CLINC150_DOMAINS: dict[str, tuple[str, ...]] = {
    "banking": (
        "account_blocked",
        "balance",
        "bill_balance",
        "bill_due",
        "freeze_account",
        "interest_rate",
        "min_payment",
        "order_checks",
        "pay_bill",
        "pin_change",
        "report_fraud",
        "routing",
        "spending_history",
        "transactions",
        "transfer",
    ),
    "credit_cards": (
        "application_status",
        "apr",
        "card_declined",
        "credit_limit",
        "credit_limit_change",
        "credit_score",
        "damaged_card",
        "expiration_date",
        "improve_credit_score",
        "international_fees",
        "new_card",
        "redeem_rewards",
        "replacement_card_duration",
        "report_lost_card",
        "rewards_balance",
    ),
    "kitchen_and_dining": (
        "accept_reservations",
        "calories",
        "cancel_reservation",
        "confirm_reservation",
        "cook_time",
        "food_last",
        "how_busy",
        "ingredient_substitution",
        "ingredients_list",
        "meal_suggestion",
        "nutrition_info",
        "recipe",
        "restaurant_reservation",
        "restaurant_reviews",
        "restaurant_suggestion",
    ),
    "home": (
        "calendar",
        "calendar_update",
        "next_song",
        "order",
        "order_status",
        "play_music",
        "reminder",
        "reminder_update",
        "shopping_list",
        "shopping_list_update",
        "smart_home",
        "todo_list",
        "todo_list_update",
        "update_playlist",
        "what_song",
    ),
    "auto_and_commute": (
        "current_location",
        "directions",
        "distance",
        "gas",
        "gas_type",
        "jump_start",
        "last_maintenance",
        "mpg",
        "oil_change_how",
        "oil_change_when",
        "schedule_maintenance",
        "tire_change",
        "tire_pressure",
        "traffic",
        "uber",
    ),
    "travel": (
        "book_flight",
        "book_hotel",
        "car_rental",
        "carry_on",
        "exchange_rate",
        "flight_status",
        "international_visa",
        "lost_luggage",
        "plug_type",
        "timezone",
        "translate",
        "travel_alert",
        "travel_notification",
        "travel_suggestion",
        "vaccines",
    ),
    "utility": (
        "alarm",
        "calculator",
        "date",
        "definition",
        "find_phone",
        "flip_coin",
        "make_call",
        "measurement_conversion",
        "roll_dice",
        "share_location",
        "spelling",
        "text",
        "time",
        "timer",
        "weather",
    ),
    "work": (
        "direct_deposit",
        "income",
        "insurance",
        "insurance_change",
        "meeting_schedule",
        "next_holiday",
        "payday",
        "pto_balance",
        "pto_request",
        "pto_request_status",
        "pto_used",
        "rollover_401k",
        "schedule_meeting",
        "taxes",
        "w2",
    ),
    "small_talk": (
        "are_you_a_bot",
        "do_you_have_pets",
        "fun_fact",
        "goodbye",
        "greeting",
        "how_old_are_you",
        "meaning_of_life",
        "tell_joke",
        "thank_you",
        "what_are_your_hobbies",
        "what_can_i_ask_you",
        "what_is_your_name",
        "where_are_you_from",
        "who_do_you_work_for",
        "who_made_you",
    ),
    "meta": (
        "cancel",
        "change_accent",
        "change_ai_name",
        "change_language",
        "change_speed",
        "change_user_name",
        "change_volume",
        "maybe",
        "no",
        "repeat",
        "reset_settings",
        "sync_device",
        "user_name",
        "whisper_mode",
        "yes",
    ),
}

CLINC150_INTENT_TO_DOMAIN: dict[str, str] = {
    intent: domain
    for domain, intents in CLINC150_DOMAINS.items()
    for intent in intents
}
CLINC150_INTENT_TO_DOMAIN["oos"] = "out_of_scope"


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve local staging directory for pinned outcome benchmark datasets."""
    raw = cache_dir or os.environ.get("FAST_SILICON_OUTCOME_CACHE") or DEFAULT_CACHE_DIR
    return Path(raw).expanduser().resolve()


def resolve_lockfile_path(lock_path: str | Path | None = None) -> Path:
    """Locate ``configs/outcome_benchmarks.lock.json`` from explicit path or ancestor repo root."""
    if lock_path is not None:
        p = Path(lock_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Outcome benchmark lockfile not found: {p}")
        return p
    env_p = os.environ.get("FAST_SILICON_OUTCOME_LOCKFILE")
    if env_p:
        p = Path(env_p).expanduser().resolve()
        if p.is_file():
            return p
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / DEFAULT_LOCK_FILENAME,
        *[parent / DEFAULT_LOCK_FILENAME for parent in here.parents],
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Could not locate {DEFAULT_LOCK_FILENAME}; pass lock_path explicitly or set FAST_SILICON_OUTCOME_LOCKFILE."
    )


def load_lockfile(lock_path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """Load lockfile JSON and return ``(lock_dict, lockfile_sha256)``."""
    p = resolve_lockfile_path(lock_path)
    raw_bytes = p.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"Malformed outcome benchmark lockfile: {p}")
    return data, digest


def compute_file_sha256(path: str | Path) -> str:
    """Stream SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_sort_key(dataset_key: str, item_id: str | int) -> str:
    """Deterministic SHA-256 ordering key immune to Python RNG/hash-seed differences."""
    return hashlib.sha256(f"{dataset_key}:{item_id}".encode("utf-8")).hexdigest()


def stratified_sha256_sample(
    items: Sequence[Any],
    *,
    dataset_key: str,
    id_fn: Callable[[Any], str | int],
    stratum_fn: Callable[[Any], str] | None = None,
    total_n: int | None = None,
    per_stratum: int | None = None,
) -> list[Any]:
    """Deterministic SHA-256 stratified sampler across arbitrary item records.

    Sorts items within each stratum by ``sha256(f"{dataset_key}:{id}")``.
    - If ``per_stratum`` is specified: selects up to ``per_stratum`` items per stratum.
    - If ``total_n`` is specified: allocates proportionally across strata via largest-remainder
      (guaranteeing >=1 per non-empty stratum whenever ``total_n >= len(strata)``).
    - If neither is set: returns all items in deterministic stratified SHA-256 order.
    """
    if not items:
        return []
    groups: dict[str, list[Any]] = {}
    for item in items:
        s_key = str(stratum_fn(item)) if stratum_fn is not None else "all"
        groups.setdefault(s_key, []).append(item)

    sorted_groups: dict[str, list[Any]] = {}
    for s_key in sorted(groups):
        sorted_groups[s_key] = sorted(
            groups[s_key],
            key=lambda it: (sha256_sort_key(dataset_key, id_fn(it)), str(id_fn(it))),
        )

    if per_stratum is not None:
        k = max(0, int(per_stratum))
        out: list[Any] = []
        for s_key in sorted(sorted_groups):
            out.extend(sorted_groups[s_key][:k])
        return out

    if total_n is None or int(total_n) >= len(items):
        out = []
        for s_key in sorted(sorted_groups):
            out.extend(sorted_groups[s_key])
        return out

    target = max(0, int(total_n))
    if target == 0:
        return []

    strata = sorted(sorted_groups.keys())
    n_total = len(items)
    alloc: dict[str, int] = {s: 0 for s in strata}

    if target >= len(strata):
        for s in strata:
            alloc[s] = 1
        remaining = target - len(strata)
        capacities = {s: len(sorted_groups[s]) - 1 for s in strata}
        cap_sum = sum(capacities.values())
        if remaining > 0 and cap_sum > 0:
            raw_shares = {s: (remaining * capacities[s]) / float(cap_sum) for s in strata}
            floors = {s: min(capacities[s], int(raw_shares[s])) for s in strata}
            for s in strata:
                alloc[s] += floors[s]
            leftover = target - sum(alloc.values())
            order_rem = sorted(
                strata,
                key=lambda s: (-(raw_shares[s] - floors[s]), s),
            )
            idx = 0
            while leftover > 0 and idx < len(order_rem):
                s = order_rem[idx]
                if alloc[s] < len(sorted_groups[s]):
                    alloc[s] += 1
                    leftover -= 1
                idx += 1
    else:
        # Fewer requested items than strata: pick top strata by deterministic hash of stratum name.
        chosen_strata = sorted(strata, key=lambda s: (sha256_sort_key(dataset_key, f"stratum:{s}"), s))[:target]
        for s in chosen_strata:
            alloc[s] = 1

    out = []
    for s in strata:
        out.extend(sorted_groups[s][: alloc[s]])
    return out


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a Parquet file into Python dicts via PyArrow without touching ``datasets``."""
    tbl = pq.read_table(Path(path))
    cols = tbl.column_names
    py_cols = [tbl.column(c).to_pylist() for c in cols]
    return [dict(zip(cols, vals)) for vals in zip(*py_cols)]


def read_parquet_class_names(path: str | Path, column: str) -> list[str]:
    """Extract HuggingFace ``ClassLabel`` names embedded in PyArrow Parquet schema metadata."""
    sch = pq.read_schema(Path(path))
    if not sch.metadata or b"huggingface" not in sch.metadata:
        return []
    meta = json.loads(sch.metadata[b"huggingface"].decode("utf-8"))
    feat = meta.get("info", {}).get("features", {}).get(column, {})
    names = feat.get("names")
    return [str(x) for x in names] if isinstance(names, list) else []


def read_arrow_ipc_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a HuggingFace Arrow IPC ``.arrow`` shard directly via ``pyarrow.ipc``."""
    p = Path(path)
    with p.open("rb") as fp:
        try:
            tbl = pa_ipc.open_stream(fp).read_all()
        except Exception:
            fp.seek(0)
            tbl = pa_ipc.open_file(fp).read_all()
    cols = tbl.column_names
    py_cols = [tbl.column(c).to_pylist() for c in cols]
    return [dict(zip(cols, vals)) for vals in zip(*py_cols)]


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL (or JSON list) records."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        return list(parsed) if isinstance(parsed, list) else []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def download_pinned_file(
    dataset_key: str,
    filename: str,
    *,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    verify_checksum: bool = True,
) -> Path:
    """Download/resolve a single pinned file from HuggingFace Hub into ``cache_dir/<dataset_key>/<filename>``."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"][dataset_key]
    if ds.get("gated", False):
        raise PermissionError(f"Refusing gated dataset {dataset_key!r}")

    # Hard license safety checks on excluded subsets before any network call.
    fn_low = filename.lower()
    if dataset_key == "legalbench_rag" and ("maud" in fn_low.split("/") or fn_low.endswith("maud.json")):
        raise ValueError("Refusing excluded LegalBench-RAG subset 'maud' (unverified upstream license)")
    if dataset_key == "longembed" and "summ_screen_fd" in fn_low:
        raise ValueError("Refusing excluded LongEmbed subset 'summ_screen_fd' (CC-BY-SA-4.0)")

    base_cache = resolve_cache_dir(cache_dir)
    dest_root = base_cache / dataset_key
    local_candidate = dest_root / filename

    pinned_meta = ds.get("pinned_files", {}).get(filename, {})
    expected_sha = pinned_meta.get("sha256")

    if local_candidate.is_file():
        if verify_checksum and expected_sha:
            actual = compute_file_sha256(local_candidate)
            if actual != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for cached {dataset_key}/{filename}: got {actual}, expected {expected_sha}"
                )
        return local_candidate

    if str(ds.get("source") or "").strip().lower() == "github":
        if local_files_only:
            raise FileNotFoundError(
                f"Offline mode enabled and GitHub pinned file missing from cache: {local_candidate}"
            )
        raw_url = (
            f"https://raw.githubusercontent.com/{ds['repo_id']}/{ds['revision']}/"
            f"{urllib.parse.quote(filename.lstrip('/'), safe='/')}"
        )
        local_candidate.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_candidate.with_name(f".{local_candidate.name}.tmp.{os.getpid()}")
        req = urllib.request.Request(raw_url, headers={"User-Agent": "fast-silicon-outcome-eval/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, tmp_path.open("wb") as out_fp:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out_fp.write(chunk)
            tmp_path.replace(local_candidate)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        downloaded = local_candidate
    else:
        from huggingface_hub import hf_hub_download

        dest_root.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=str(ds["repo_id"]),
                filename=filename,
                repo_type="dataset",
                revision=str(ds["revision"]),
                local_dir=str(dest_root),
                local_files_only=local_files_only,
            )
        )
    if verify_checksum and expected_sha:
        actual = compute_file_sha256(downloaded)
        if actual != expected_sha:
            raise ValueError(
                f"Checksum mismatch for downloaded {dataset_key}/{filename}: got {actual}, expected {expected_sha}"
            )
    return downloaded


# ---------------------------------------------------------------------------
# Category 1 Loaders: NanoBEIR-en, LegalBench-RAG, LongEmbed
# ---------------------------------------------------------------------------


def load_nanobeir_subset(
    subset: str,
    *,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load one ``sentence-transformers/NanoBEIR-en`` subset (corpus, queries, qrels)."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    allowed = set(lock["datasets"]["nanobeir_en"]["tier2_subsets"])
    if subset not in allowed:
        raise ValueError(f"Unknown NanoBEIR-en subset {subset!r}; expected one of {sorted(allowed)}")

    p_corp = download_pinned_file(
        "nanobeir_en",
        f"corpus/{subset}-00000-of-00001.parquet",
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    p_qry = download_pinned_file(
        "nanobeir_en",
        f"queries/{subset}-00000-of-00001.parquet",
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    p_qrel = download_pinned_file(
        "nanobeir_en",
        f"qrels/{subset}-00000-of-00001.parquet",
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    corpus: dict[str, str] = {}
    for row in read_parquet_rows(p_corp):
        cid = str(row["_id"])
        corpus[cid] = str(row.get("text") or "")

    queries: dict[str, str] = {}
    for row in read_parquet_rows(p_qry):
        qid = str(row["_id"])
        queries[qid] = str(row.get("text") or "")

    qrels: dict[str, dict[str, float]] = {}
    for row in read_parquet_rows(p_qrel):
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = float(row.get("score", 1.0)) if row.get("score") is not None else 1.0
        qrels.setdefault(qid, {})[cid] = score

    return {"subset": subset, "corpus": corpus, "queries": queries, "qrels": qrels}


def filter_legalbench_subsets(subsets: Sequence[str]) -> list[str]:
    """Enforce permissive LegalBench-RAG subset allowlist (strictly rejecting ``maud``)."""
    out: list[str] = []
    for sub in subsets:
        s = str(sub).strip().lower()
        if s in FORBIDDEN_LEGALBENCH_SUBSETS:
            raise ValueError(f"LegalBench-RAG subset {sub!r} is strictly excluded due to unverified license")
        if s not in ALLOWED_LEGALBENCH_SUBSETS:
            raise ValueError(f"Unknown LegalBench-RAG subset {sub!r}; allowed: {sorted(ALLOWED_LEGALBENCH_SUBSETS)}")
        if s not in out:
            out.append(s)
    return out


def load_legalbench_rag(
    *,
    tier: str = "tier1",
    subsets: Sequence[str] | None = None,
    max_docs: int | None = None,
    max_queries: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load LegalBench-RAG (strictly excluding ``maud``) with deterministic doc/query selection."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["legalbench_rag"]
    t_cfg = ds.get(tier, ds["tier1"])
    eff_subsets = filter_legalbench_subsets(subsets if subsets is not None else t_cfg["subsets"])
    eff_max_docs = int(t_cfg["max_docs"] if max_docs is None else max_docs)
    eff_max_queries = int(t_cfg["max_queries"] if max_queries is None else max_queries)

    all_tests: list[dict[str, Any]] = []
    doc_to_subset: dict[str, str] = {}
    for sub in eff_subsets:
        bp = download_pinned_file(
            "legalbench_rag",
            f"benchmarks/{sub}.json",
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        raw = json.loads(bp.read_text(encoding="utf-8"))
        for idx, item in enumerate(raw.get("tests", [])):
            snippets = []
            for sn in item.get("snippets", []):
                fp = str(sn.get("file_path") or "")
                if not fp or fp.lower().startswith("maud/"):
                    continue
                span = sn.get("span")
                if not isinstance(span, list) or len(span) != 2:
                    continue
                a, b = int(span[0]), int(span[1])
                if b <= a:
                    continue
                snippets.append((fp, a, b))
                doc_to_subset.setdefault(fp, sub)
            if snippets:
                all_tests.append(
                    {
                        "id": f"{sub}:{idx}",
                        "subset": sub,
                        "query": str(item.get("query") or ""),
                        "gold_spans": snippets,
                    }
                )

    # Deterministically select documents stratified across allowed subsets.
    doc_items = [{"doc_id": fp, "subset": sub} for fp, sub in doc_to_subset.items()]
    selected_docs = stratified_sha256_sample(
        doc_items,
        dataset_key=f"legalbench_rag:{tier}:docs",
        id_fn=lambda r: r["doc_id"],
        stratum_fn=lambda r: r["subset"],
        total_n=eff_max_docs,
    )
    selected_doc_set = {d["doc_id"] for d in selected_docs}

    # Retain queries whose gold spans fall within the selected documents.
    eligible_queries: list[dict[str, Any]] = []
    for q in all_tests:
        kept_spans = [sp for sp in q["gold_spans"] if sp[0] in selected_doc_set]
        if kept_spans:
            eligible_queries.append({**q, "gold_spans": kept_spans})

    selected_queries = stratified_sha256_sample(
        eligible_queries,
        dataset_key=f"legalbench_rag:{tier}:queries",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["subset"],
        total_n=eff_max_queries,
    )

    corpus_texts: dict[str, str] = {}
    for doc_id in sorted(selected_doc_set):
        cp = download_pinned_file(
            "legalbench_rag",
            f"corpus/{doc_id}",
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            verify_checksum=False,
        )
        corpus_texts[doc_id] = cp.read_text(encoding="utf-8", errors="replace")

    return {
        "subsets": eff_subsets,
        "corpus_texts": corpus_texts,
        "queries": selected_queries,
    }


def filter_longembed_subsets(subsets: Sequence[str]) -> list[str]:
    """Enforce permissive LongEmbed subset allowlist (strictly rejecting ``summ_screen_fd``)."""
    out: list[str] = []
    for sub in subsets:
        s = str(sub).strip().lower()
        if s in FORBIDDEN_LONGEMBED_SUBSETS:
            raise ValueError(f"LongEmbed subset {sub!r} is strictly excluded (CC-BY-SA-4.0)")
        if s not in ALLOWED_LONGEMBED_SUBSETS:
            raise ValueError(f"Unknown LongEmbed subset {sub!r}; allowed: {sorted(ALLOWED_LONGEMBED_SUBSETS)}")
        if s not in out:
            out.append(s)
    return out


def load_longembed_subset(
    subset: str,
    *,
    context_lengths: Sequence[int] | None = None,
    max_queries: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load one allowed ``dwzhu/LongEmbed`` subset (strictly forbidding ``summ_screen_fd``)."""
    eff_sub = filter_longembed_subsets([subset])[0]
    if lock is None:
        lock, _ = load_lockfile(lock_path)

    p_corp = download_pinned_file(
        "longembed", f"{eff_sub}/corpus.jsonl", lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    p_qry = download_pinned_file(
        "longembed", f"{eff_sub}/queries.jsonl", lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    p_qrel = download_pinned_file(
        "longembed", f"{eff_sub}/qrels.jsonl", lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )

    ctx_filter = {int(c) for c in context_lengths} if context_lengths else None
    raw_queries = read_jsonl_rows(p_qry)
    if ctx_filter is not None:
        raw_queries = [q for q in raw_queries if int(q.get("context_length") or 0) in ctx_filter]

    if max_queries is not None and len(raw_queries) > int(max_queries):
        raw_queries = stratified_sha256_sample(
            raw_queries,
            dataset_key=f"longembed:{eff_sub}",
            id_fn=lambda r: str(r.get("qid") or r.get("_id") or ""),
            stratum_fn=lambda r: str(r.get("context_length", "all")),
            total_n=int(max_queries),
        )

    queries = {str(q.get("qid") or q.get("_id")): str(q.get("text") or "") for q in raw_queries}
    query_ctx = {
        str(q.get("qid") or q.get("_id")): int(q.get("context_length") or 0)
        for q in raw_queries
        if q.get("context_length") is not None
    }

    qrels: dict[str, dict[str, float]] = {}
    for row in read_jsonl_rows(p_qrel):
        qid = str(row.get("qid") or row.get("query-id") or "")
        if qid not in queries:
            continue
        did = str(row.get("doc_id") or row.get("corpus-id") or "")
        if did:
            qrels.setdefault(qid, {})[did] = float(row.get("score", 1.0) or 1.0)

    corpus: dict[str, str] = {}
    for row in read_jsonl_rows(p_corp):
        did = str(row.get("doc_id") or row.get("_id") or "")
        if ctx_filter is not None and row.get("context_length") is not None:
            if int(row["context_length"]) not in ctx_filter:
                continue
        if did:
            corpus[did] = str(row.get("text") or "")

    return {
        "subset": eff_sub,
        "corpus": corpus,
        "queries": queries,
        "query_context_lengths": query_ctx,
        "qrels": qrels,
    }


# ---------------------------------------------------------------------------
# Category 2 Loaders: RAGTruth, CLINC150, Banking77, RewardBench-2
# ---------------------------------------------------------------------------


def _ragtruth_is_faithful(row: Mapping[str, Any]) -> bool:
    hlp = row.get("hallucination_labels_processed")
    if isinstance(hlp, Mapping):
        ev = int(hlp.get("evident_conflict") or 0)
        bl = int(hlp.get("baseless_info") or 0)
        return (ev == 0) and (bl == 0)
    raw = str(row.get("hallucination_labels") or "").strip()
    return raw in ("", "[]", "null", "None")


def load_ragtruth(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``wandb/RAGTruth-processed`` ``test`` split with deterministic stratified sampling."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["ragtruth"]
    p = download_pinned_file(
        "ragtruth", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    rows = read_parquet_rows(p)
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        faithful = _ragtruth_is_faithful(r)
        items.append(
            {
                "id": str(r.get("id") or idx),
                "query": str(r.get("query") or ""),
                "context": str(r.get("context") or ""),
                "output": str(r.get("output") or ""),
                "task_type": str(r.get("task_type") or "unknown"),
                "is_faithful": bool(faithful),
                "label": 1 if faithful else 0,
            }
        )
    target_n = ds.get(tier, {}).get("n") if max_items is None else int(max_items)
    return stratified_sha256_sample(
        items,
        dataset_key=f"ragtruth:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: f"{r['task_type']}:{int(r['is_faithful'])}",
        total_n=target_n,
    )


def load_clinc_oos(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load ``clinc/clinc_oos`` ``plus/test`` with domain+intent labels for 2-stage K=11 -> K=15 routing."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["clinc_oos"]
    p = download_pinned_file(
        "clinc_oos", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    intent_names = read_parquet_class_names(p, "intent")
    rows = read_parquet_rows(p)
    in_scope_items: list[dict[str, Any]] = []
    oos_items: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        iid = int(r["intent"])
        name = intent_names[iid] if 0 <= iid < len(intent_names) else f"intent_{iid}"
        domain = CLINC150_INTENT_TO_DOMAIN.get(name, "out_of_scope")
        is_oos = name == "oos" or domain == "out_of_scope"
        rec = {
            "id": f"clinc:{idx}",
            "text": str(r.get("text") or ""),
            "intent_id": iid,
            "intent_name": name,
            "domain": domain,
            "is_oos": is_oos,
        }
        if is_oos:
            oos_items.append(rec)
        else:
            in_scope_items.append(rec)

    t_cfg = ds.get(tier, ds["tier1"])
    if max_items is not None:
        all_items = in_scope_items + oos_items
        selected = stratified_sha256_sample(
            all_items,
            dataset_key=f"clinc_oos:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["domain"],
            total_n=int(max_items),
        )
    else:
        n_is = int(t_cfg.get("in_scope", 450))
        n_oos = int(t_cfg.get("oos", 100))
        sel_is = stratified_sha256_sample(
            in_scope_items,
            dataset_key=f"clinc_oos:{tier}:is",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["domain"],
            total_n=n_is,
        )
        sel_oos = stratified_sha256_sample(
            oos_items,
            dataset_key=f"clinc_oos:{tier}:oos",
            id_fn=lambda r: r["id"],
            stratum_fn=None,
            total_n=n_oos,
        )
        selected = sel_is + sel_oos

    return {
        "items": selected,
        "intent_names": intent_names,
        "domains": CLINC150_DOMAINS,
    }


def load_banking77(
    *,
    tier: str = "tier2",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load ``PolyAI/banking77`` test split with all 77 intent class names."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["banking77"]
    p = download_pinned_file(
        "banking77", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    label_names = read_parquet_class_names(p, "label")
    rows = read_parquet_rows(p)
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        lid = int(r["label"])
        lname = label_names[lid] if 0 <= lid < len(label_names) else f"label_{lid}"
        items.append({"id": f"bank77:{idx}", "text": str(r.get("text") or ""), "label_id": lid, "label_name": lname})
    target_n = ds.get(tier, {}).get("n") if max_items is None else int(max_items)
    if target_n == 0:
        selected: list[dict[str, Any]] = []
    else:
        selected = stratified_sha256_sample(
            items,
            dataset_key=f"banking77:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: str(r["label_id"]),
            total_n=target_n,
        )
    return {"items": selected, "label_names": label_names}


def load_reward_bench_2(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``allenai/reward-bench-2`` Best-of-4 prompts (excluding ``Ties``) with deterministic candidate permutation."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["reward_bench_2"]
    p = download_pinned_file(
        "reward_bench_2", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    allowed = set(ALLOWED_REWARDBENCH2_SUBSETS)
    rows = read_parquet_rows(p)
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        subset = str(r.get("subset") or "")
        if subset not in allowed:
            continue
        chosen = r.get("chosen") or []
        rejected = r.get("rejected") or []
        if not isinstance(chosen, list) or not isinstance(rejected, list):
            continue
        if len(chosen) < 1 or len(rejected) < 3:
            continue
        item_id = str(r.get("id") if r.get("id") is not None else idx)
        # Deterministically permute candidate positions by SHA-256 to eliminate index-0 tie bias.
        gold_idx = int(sha256_sort_key("rb2_pos", item_id)[:8], 16) % 4
        cands = [str(rejected[0]), str(rejected[1]), str(rejected[2])]
        cands.insert(gold_idx, str(chosen[0]))
        items.append(
            {
                "id": item_id,
                "subset": subset,
                "prompt": str(r.get("prompt") or ""),
                "candidates": cands,
                "gold_idx": gold_idx,
            }
        )

    t_cfg = ds.get(tier, ds["tier1"])
    if max_items is not None:
        return stratified_sha256_sample(
            items,
            dataset_key=f"reward_bench_2:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["subset"],
            total_n=int(max_items),
        )
    if "per_subset" in t_cfg and t_cfg["per_subset"] is not None:
        return stratified_sha256_sample(
            items,
            dataset_key=f"reward_bench_2:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["subset"],
            per_stratum=int(t_cfg["per_subset"]),
        )
    return stratified_sha256_sample(
        items,
        dataset_key=f"reward_bench_2:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["subset"],
        total_n=t_cfg.get("n"),
    )


# ---------------------------------------------------------------------------
# Category 3 Loaders: IFBench_test, IFEval, MMLU-Redux 2.0, MATH-500, LiveBench
# ---------------------------------------------------------------------------


def load_ifbench_test(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``allenai/IFBench_test`` rows with non-null instruction kwargs normalized."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["ifbench_test"]
    p = download_pinned_file(
        "ifbench_test", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(read_parquet_rows(p)):
        iids = [str(x) for x in (r.get("instruction_id_list") or [])]
        raw_kws = r.get("kwargs") or []
        clean_kws = [
            {k: v for k, v in (kw or {}).items() if v is not None} if isinstance(kw, Mapping) else {}
            for kw in raw_kws
        ]
        fam = iids[0].split(":")[0] if iids else "unknown"
        items.append(
            {
                "id": str(r.get("key") if r.get("key") is not None else idx),
                "prompt": str(r.get("prompt") or ""),
                "instruction_id_list": iids,
                "kwargs": clean_kws,
                "primary_family": fam,
            }
        )
    target_n = ds.get(tier, {}).get("n") if max_items is None else int(max_items)
    return stratified_sha256_sample(
        items,
        dataset_key=f"ifbench_test:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["primary_family"],
        total_n=target_n,
    )


def load_ifeval(
    *,
    tier: str = "tier2",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``google/IFEval`` rows."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["ifeval"]
    p = download_pinned_file(
        "ifeval", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(read_jsonl_rows(p)):
        iids = [str(x) for x in (r.get("instruction_id_list") or [])]
        raw_kws = r.get("kwargs") or []
        clean_kws = [
            {k: v for k, v in (kw or {}).items() if v is not None} if isinstance(kw, Mapping) else {}
            for kw in raw_kws
        ]
        fam = iids[0].split(":")[0] if iids else "unknown"
        items.append(
            {
                "id": str(r.get("key") if r.get("key") is not None else idx),
                "prompt": str(r.get("prompt") or ""),
                "instruction_id_list": iids,
                "kwargs": clean_kws,
                "primary_family": fam,
            }
        )
    target_n = ds.get(tier, {}).get("n") if max_items is None else int(max_items)
    if target_n == 0:
        return []
    return stratified_sha256_sample(
        items,
        dataset_key=f"ifeval:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["primary_family"],
        total_n=target_n,
    )


def load_mmlu_redux(
    *,
    tier: str = "tier1",
    subjects: Sequence[str] | None = None,
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``edinburgh-dawg/mmlu-redux-2.0`` strictly filtering ``error_type == 'ok'`` across Arrow IPC shards."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["mmlu_redux_2"]
    all_subjects = list(ds.get("subjects", []))
    eff_subjects = list(subjects) if subjects is not None else all_subjects
    err_filter = str(ds.get("error_type_filter", "ok"))

    letters = ("A", "B", "C", "D")
    items: list[dict[str, Any]] = []
    for sub in eff_subjects:
        p = download_pinned_file(
            "mmlu_redux_2",
            f"{sub}/data-00000-of-00001.arrow",
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        for idx, r in enumerate(read_arrow_ipc_rows(p)):
            if str(r.get("error_type") or "").strip().lower() != err_filter:
                continue
            ans_idx = int(r["answer"])
            choices = [str(c) for c in (r.get("choices") or [])]
            if not (0 <= ans_idx < min(len(choices), 4)):
                continue
            items.append(
                {
                    "id": f"{sub}:{idx}",
                    "subject": sub,
                    "question": str(r.get("question") or ""),
                    "choices": choices[:4],
                    "answer_idx": ans_idx,
                    "answer_letter": letters[ans_idx],
                }
            )

    t_cfg = ds.get(tier, ds["tier1"])
    if max_items is not None:
        return stratified_sha256_sample(
            items,
            dataset_key=f"mmlu_redux_2:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["subject"],
            total_n=int(max_items),
        )
    if t_cfg.get("per_subject") is not None:
        return stratified_sha256_sample(
            items,
            dataset_key=f"mmlu_redux_2:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["subject"],
            per_stratum=int(t_cfg["per_subject"]),
        )
    return stratified_sha256_sample(
        items,
        dataset_key=f"mmlu_redux_2:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["subject"],
        total_n=t_cfg.get("n"),
    )


def load_math_500(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``HuggingFaceH4/MATH-500`` stratified by difficulty level (1..5)."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["math_500"]
    p = download_pinned_file(
        "math_500", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(read_jsonl_rows(p)):
        items.append(
            {
                "id": str(r.get("unique_id") or idx),
                "problem": str(r.get("problem") or ""),
                "answer": str(r.get("answer") or ""),
                "subject": str(r.get("subject") or ""),
                "level": int(r.get("level") or 0),
            }
        )
    t_cfg = ds.get(tier, ds["tier1"])
    if max_items is not None:
        return stratified_sha256_sample(
            items,
            dataset_key=f"math_500:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: str(r["level"]),
            total_n=int(max_items),
        )
    if t_cfg.get("per_level") is not None:
        return stratified_sha256_sample(
            items,
            dataset_key=f"math_500:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: str(r["level"]),
            per_stratum=int(t_cfg["per_level"]),
        )
    return stratified_sha256_sample(
        items,
        dataset_key=f"math_500:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: str(r["level"]),
        total_n=t_cfg.get("n"),
    )


def load_livebench_reasoning(
    *,
    tier: str = "tier2",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load commit-pinned ``livebench/reasoning`` tasks (``zebra_puzzle``, ``spatial``, ``web_of_lies_v2``)."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["livebench_reasoning"]
    p = download_pinned_file(
        "livebench_reasoning", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(read_parquet_rows(p)):
        turns = r.get("turns") or []
        prompt = "\n\n".join(str(t) for t in turns) if isinstance(turns, list) else str(turns)
        items.append(
            {
                "id": str(r.get("question_id") or idx),
                "task": str(r.get("task") or ""),
                "prompt": prompt,
                "ground_truth": str(r.get("ground_truth") or ""),
                "level": str(r.get("level") or ""),
            }
        )
    target_n = ds.get(tier, {}).get("n") if max_items is None else int(max_items)
    if target_n == 0:
        return []
    return stratified_sha256_sample(
        items,
        dataset_key=f"livebench_reasoning:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["task"],
        total_n=target_n,
    )


# ---------------------------------------------------------------------------
# Category 4 Loaders: LiquidAI, BFCL v3, JSONSchemaBench, ExtractBench
# ---------------------------------------------------------------------------


def load_liquidai_extraction(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``LiquidAI/structured-extraction`` ``test_unseen`` (+ ``test`` in Tier-2)."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["liquidai_extraction"]
    t_cfg = ds.get(tier, ds["tier1"])

    def _read_split(fn: str, split_name: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        p = download_pinned_file(
            "liquidai_extraction", fn, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
        )
        recs: list[dict[str, Any]] = []
        for idx, r in enumerate(read_parquet_rows(p)):
            raw_schema = r.get("schema")
            raw_ext = r.get("extraction")
            schema_obj = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
            ext_obj = json.loads(raw_ext) if isinstance(raw_ext, str) else raw_ext
            recs.append(
                {
                    "id": str(r.get("id") or f"{split_name}:{idx}"),
                    "split": split_name,
                    "doc_type": str(r.get("doc_type") or "unknown"),
                    "language": str(r.get("language") or "en"),
                    "document": str(r.get("document") or ""),
                    "schema": schema_obj,
                    "extraction": ext_obj,
                }
            )
        return stratified_sha256_sample(
            recs,
            dataset_key=f"liquidai_extraction:{tier}:{split_name}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: f"{r['doc_type']}:{r['language']}",
            total_n=limit,
        )

    if max_items is not None:
        return _read_split("test_unseen.parquet", "test_unseen", int(max_items))

    out = _read_split("test_unseen.parquet", "test_unseen", int(t_cfg.get("test_unseen", 150)))
    out.extend(_read_split("test.parquet", "test", int(t_cfg.get("test", 0))))
    return out


def load_bfcl_v3(
    *,
    tier: str = "tier1",
    splits: Sequence[str] | None = None,
    per_split: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``BFCL v3`` single-turn AST + ``irrelevance`` splits paired with ``possible_answer/`` ground truth."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["bfcl_v3"]
    split_specs = ds["splits"]
    eff_splits = list(splits) if splits is not None else list(split_specs.keys())
    t_cfg = ds.get(tier, ds["tier1"])
    k = int(t_cfg.get("per_split", 30) if per_split is None else per_split)

    out: list[dict[str, Any]] = []
    for sp_name in eff_splits:
        spec = split_specs[sp_name]
        p_q = download_pinned_file(
            "bfcl_v3", spec["questions"], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
        )
        q_rows = read_jsonl_rows(p_q)
        ans_map: dict[str, Any] = {}
        if spec.get("answers"):
            p_a = download_pinned_file(
                "bfcl_v3", spec["answers"], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
            )
            for a_row in read_jsonl_rows(p_a):
                ans_map[str(a_row["id"])] = a_row.get("ground_truth")

        is_irrel = sp_name == "irrelevance"
        sp_items: list[dict[str, Any]] = []
        for r in q_rows:
            rid = str(r["id"])
            sp_items.append(
                {
                    "id": rid,
                    "split": sp_name,
                    "question": r.get("question") or [],
                    "functions": r.get("function") or [],
                    "ground_truth": [] if is_irrel else (ans_map.get(rid) or []),
                    "is_irrelevance": is_irrel,
                }
            )
        out.extend(
            stratified_sha256_sample(
                sp_items,
                dataset_key=f"bfcl_v3:{tier}:{sp_name}",
                id_fn=lambda r: r["id"],
                stratum_fn=None,
                total_n=k,
            )
        )
    return out


def load_jsonschemabench(
    *,
    tier: str = "tier2",
    per_subset: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``epfl-dlab/JSONSchemaBench`` ``Github_hard`` & ``Kubernetes`` schemas."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["jsonschemabench"]
    t_cfg = ds.get(tier, ds["tier2"])
    k = int(t_cfg.get("per_subset", 100) if per_subset is None else per_subset)

    out: list[dict[str, Any]] = []
    for fn in ds["files"]:
        sub = fn.split("/")[0]
        p = download_pinned_file(
            "jsonschemabench", fn, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
        )
        rows = read_parquet_rows(p)
        sub_items = [
            {
                "id": str(r.get("unique_id") or f"{sub}:{i}"),
                "subset": sub,
                "json_schema": str(r.get("json_schema") or ""),
            }
            for i, r in enumerate(rows)
        ]
        out.extend(
            stratified_sha256_sample(
                sub_items,
                dataset_key=f"jsonschemabench:{tier}:{sub}",
                id_fn=lambda r: r["id"],
                stratum_fn=None,
                total_n=k,
            )
        )
    return out


def load_extractbench(
    *,
    tier: str = "tier2",
    max_items: int | None = None,
    extract_pdf_text: bool = False,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``llamaindex/ExtractBench`` ``short`` L1 metadata (+ optional pypdfium2 UTF-8 page extraction)."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["extractbench"]
    t_cfg = ds.get(tier, ds["tier2"])
    target_n = int(t_cfg.get("n", 80) if max_items is None else max_items)
    if target_n <= 0:
        return []

    p = download_pinned_file(
        "extractbench", ds["files"][0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
    )
    items: list[dict[str, Any]] = []
    for r in read_jsonl_rows(p):
        raw_schema = r.get("data_schema")
        raw_exp = r.get("expected_output")
        schema_obj = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
        exp_obj = json.loads(raw_exp) if isinstance(raw_exp, str) else raw_exp
        tags = [str(t) for t in (r.get("tags") or [])]
        domain_tag = next((t for t in tags if t.startswith("domain:")), "domain:unknown")
        items.append(
            {
                "id": str(r.get("id") or ""),
                "category": str(r.get("category") or "short"),
                "pdf": str(r.get("pdf") or ""),
                "data_schema": schema_obj,
                "expected_output": exp_obj,
                "tags": tags,
                "domain_tag": domain_tag,
                "text": None,
                "page_offsets": None,
            }
        )

    selected = stratified_sha256_sample(
        items,
        dataset_key=f"extractbench:{tier}:short_l1",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["domain_tag"],
        total_n=target_n,
    )

    if extract_pdf_text:
        try:
            import pypdfium2 as pdfium  # type: ignore[import-not-found]
        except ImportError:
            pdfium = None
        if pdfium is not None:
            for item in selected:
                pdf_rel = item["pdf"]
                if not pdf_rel:
                    continue
                pdf_p = download_pinned_file(
                    "extractbench",
                    pdf_rel,
                    lock=lock,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    verify_checksum=False,
                )
                doc = pdfium.PdfDocument(str(pdf_p))
                pages: list[str] = []
                offsets: list[int] = []
                cursor = 0
                try:
                    for page in doc:
                        tp = page.get_textpage()
                        txt = tp.get_text_bounded() or ""
                        tp.close()
                        page.close()
                        offsets.append(cursor)
                        pages.append(txt)
                        cursor += len(txt.encode("utf-8")) + 1
                finally:
                    doc.close()
                item["text"] = "\n".join(pages)
                item["page_offsets"] = offsets
    return selected


# ---------------------------------------------------------------------------
# Category 5 Loaders: olmOCR-Bench, FinTabNet-OTSL, OmniDocBench-EN, CORD-v2
# ---------------------------------------------------------------------------

OLMOCR_BENCH_CATEGORIES: tuple[str, ...] = (
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
)

_CORD_MENU_KEYS: tuple[str, ...] = (
    "nm",
    "num",
    "cnt",
    "unitprice",
    "price",
    "discountprice",
    "itemsubtotal",
    "etc",
)
_CORD_SUBTOTAL_KEYS: tuple[str, ...] = (
    "subtotal_price",
    "discount_price",
    "service_price",
    "tax_price",
    "etc",
)
_CORD_TOTAL_KEYS: tuple[str, ...] = (
    "total_price",
    "cashprice",
    "changeprice",
    "creditcardprice",
    "emoneyprice",
    "menuqty_cnt",
    "menutype_cnt",
    "total_etc",
)


def _extract_image_bytes(img_cell: Any) -> bytes:
    """Extract raw image bytes from a HuggingFace Parquet image struct or raw bytes cell."""
    if isinstance(img_cell, (bytes, bytearray, memoryview)):
        return bytes(img_cell)
    if isinstance(img_cell, Mapping):
        raw = img_cell.get("bytes")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw)
    return b""


def reconstruct_fintabnet_html(html_tokens: Sequence[str], cells: Sequence[Any]) -> str:
    """Reconstruct complete HTML ``<table>`` ground truth by pairing ``html`` tokens with ``cells`` text."""
    if isinstance(cells, Sequence) and cells and isinstance(cells[0], list):
        cell_list: Sequence[Any] = cells[0]
    else:
        cell_list = cells

    cell_texts: list[str] = []
    for cell in cell_list:
        if isinstance(cell, Mapping):
            toks = cell.get("tokens")
            if isinstance(toks, Sequence) and not isinstance(toks, (str, bytes)):
                raw = "".join(str(t) for t in toks)
            else:
                raw = str(cell.get("text") or "")
        else:
            raw = str(cell or "")
        cell_texts.append(" ".join(raw.split()))

    out_parts: list[str] = []
    c_idx = 0
    for tok in html_tokens:
        s = str(tok)
        low = s.strip().lower()
        if low in ("</td>", "</th>"):
            if c_idx < len(cell_texts):
                txt = cell_texts[c_idx]
                c_idx += 1
                if txt:
                    out_parts.append(_html.escape(txt, quote=False))
            out_parts.append(s)
        else:
            out_parts.append(s)

    joined = "".join(out_parts).strip()
    if not joined:
        return "<table></table>"
    if not joined.lower().startswith("<table"):
        return f"<table>{joined}</table>"
    return joined


def _coerce_cord_scalar(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        s = " ".join(val.split()).strip()
        return s or None
    if isinstance(val, list):
        parts = [_coerce_cord_scalar(el) for el in val]
        kept = [p for p in parts if p]
        return " / ".join(kept) if kept else None
    if isinstance(val, Mapping):
        parts = [_coerce_cord_scalar(v) for v in val.values()]
        kept = [p for p in parts if p]
        return " ".join(kept) if kept else None
    s = " ".join(str(val).split()).strip()
    return s or None


def normalize_cord_v2_parse(gt_parse: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize CORD-v2 ``gt_parse`` into a uniform Draft-2020-12 receipt extraction object."""
    raw_menu = gt_parse.get("menu")
    if isinstance(raw_menu, Mapping):
        menu_seq: list[Any] = [raw_menu]
    elif isinstance(raw_menu, list):
        menu_seq = raw_menu
    else:
        menu_seq = []

    norm_menu: list[dict[str, str]] = []
    for entry in menu_seq:
        if not isinstance(entry, Mapping):
            continue
        item_obj: dict[str, str] = {}
        for k in _CORD_MENU_KEYS:
            val = _coerce_cord_scalar(entry.get(k))
            if val is not None:
                item_obj[k] = val
        if item_obj:
            norm_menu.append(item_obj)

    norm_sub: dict[str, str] = {}
    raw_sub = gt_parse.get("sub_total")
    if isinstance(raw_sub, Mapping):
        for k in _CORD_SUBTOTAL_KEYS:
            val = _coerce_cord_scalar(raw_sub.get(k))
            if val is not None:
                norm_sub[k] = val

    norm_total: dict[str, str] = {}
    raw_total = gt_parse.get("total")
    if isinstance(raw_total, Mapping):
        for k in _CORD_TOTAL_KEYS:
            val = _coerce_cord_scalar(raw_total.get(k))
            if val is not None:
                norm_total[k] = val

    out: dict[str, Any] = {"menu": norm_menu, "total": norm_total}
    if norm_sub:
        out["sub_total"] = norm_sub
    return out


def build_cord_v2_schema() -> dict[str, Any]:
    """Return canonical Draft-2020-12 JSON Schema for normalized CORD-v2 receipt extraction."""
    opt_str: dict[str, Any] = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "menu": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {k: dict(opt_str) for k in _CORD_MENU_KEYS},
                    "additionalProperties": False,
                },
            },
            "sub_total": {
                "type": ["object", "null"],
                "properties": {k: dict(opt_str) for k in _CORD_SUBTOTAL_KEYS},
                "additionalProperties": False,
            },
            "total": {
                "type": "object",
                "properties": {k: dict(opt_str) for k in _CORD_TOTAL_KEYS},
                "additionalProperties": False,
            },
        },
        "required": ["menu", "total"],
        "additionalProperties": False,
    }


def load_olmocr_bench(
    *,
    tier: str = "tier1",
    categories: Sequence[str] | None = None,
    max_pages: int | None = None,
    download_pdfs: bool = True,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``allenai/olmOCR-bench`` single-page PDFs grouped with their binary unit tests."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["olmocr_bench"]
    t_cfg = ds.get(tier, ds.get("tier1", {}))
    allowed_cats = list(ds.get("categories") or OLMOCR_BENCH_CATEGORIES)
    eff_cats = list(categories) if categories is not None else allowed_cats
    for cat in eff_cats:
        if cat not in allowed_cats:
            raise ValueError(f"Unknown olmOCR-bench category {cat!r}; expected one of {allowed_cats}")

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for cat in eff_cats:
        jl_path = download_pinned_file(
            "olmocr_bench",
            f"bench_data/{cat}.jsonl",
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        for row in read_jsonl_rows(jl_path):
            pdf_rel = str(row.get("pdf") or "").strip()
            if not pdf_rel:
                continue
            key = (cat, pdf_rel)
            rec = grouped.get(key)
            if rec is None:
                rec = {
                    "id": f"{cat}:{pdf_rel}",
                    "category": cat,
                    "pdf_rel": pdf_rel,
                    "pdf_repo_path": f"bench_data/pdfs/{pdf_rel}",
                    "pdf_path": None,
                    "tests": [],
                }
                grouped[key] = rec
            rec["tests"].append(dict(row))

    all_pages = [grouped[k] for k in sorted(grouped)]
    if max_pages is not None:
        selected = stratified_sha256_sample(
            all_pages,
            dataset_key=f"olmocr_bench:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["category"],
            total_n=int(max_pages),
        )
    else:
        per_cat = t_cfg.get("per_category")
        tot_n = t_cfg.get("n")
        selected = stratified_sha256_sample(
            all_pages,
            dataset_key=f"olmocr_bench:{tier}",
            id_fn=lambda r: r["id"],
            stratum_fn=lambda r: r["category"],
            per_stratum=int(per_cat) if per_cat is not None else None,
            total_n=int(tot_n) if (per_cat is None and tot_n is not None) else None,
        )

    if download_pdfs:
        for item in selected:
            pdf_p = download_pinned_file(
                "olmocr_bench",
                item["pdf_repo_path"],
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                verify_checksum=False,
            )
            item["pdf_path"] = str(pdf_p)
    return selected


def load_fintabnet_otsl(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``docling-project/FinTabNet_OTSL`` test tables with exact reconstructed HTML GT."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["fintabnet_otsl"]
    t_cfg = ds.get(tier, ds.get("tier1", {}))
    target_n = int(max_items) if max_items is not None else (int(t_cfg["n"]) if t_cfg.get("n") is not None else None)
    if target_n is not None and target_n <= 0:
        return []
    shard_files = list(t_cfg.get("files") or ds.get("files") or [])

    meta_cols = ["filename", "imgid", "cols", "rows", "otsl", "html", "cells"]
    shard_paths: list[Path] = []
    candidates: list[dict[str, Any]] = []
    for s_idx, fn in enumerate(shard_files):
        p = download_pinned_file(
            "fintabnet_otsl",
            fn,
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        shard_paths.append(p)
        tbl = pq.read_table(p, columns=meta_cols)
        filenames = tbl.column("filename").to_pylist()
        imgids = tbl.column("imgid").to_pylist()
        cols_list = tbl.column("cols").to_pylist()
        rows_list = tbl.column("rows").to_pylist()
        otsl_list = tbl.column("otsl").to_pylist()
        html_list = tbl.column("html").to_pylist()
        cells_list = tbl.column("cells").to_pylist()
        for r_idx in range(tbl.num_rows):
            n_rows = int(rows_list[r_idx] or 0)
            n_cols = int(cols_list[r_idx] or 0)
            otsl_toks = otsl_list[r_idx] or []
            has_span = any(str(tok) in ("lcel", "ucel", "xcel") for tok in otsl_toks)
            area = n_rows * n_cols
            size_bucket = "small" if area <= 24 else ("medium" if area <= 80 else "large")
            rec_id = str(filenames[r_idx] or imgids[r_idx] or f"{s_idx}:{r_idx}")
            candidates.append(
                {
                    "id": rec_id,
                    "filename": str(filenames[r_idx] or rec_id),
                    "imgid": imgids[r_idx],
                    "rows": n_rows,
                    "cols": n_cols,
                    "has_span": bool(has_span),
                    "size_bucket": size_bucket,
                    "stratum": f"{int(has_span)}:{size_bucket}",
                    "_shard_idx": s_idx,
                    "_row_idx": r_idx,
                    "_html": html_list[r_idx] or [],
                    "_cells": cells_list[r_idx] or [],
                }
            )

    selected = stratified_sha256_sample(
        candidates,
        dataset_key=f"fintabnet_otsl:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["stratum"],
        total_n=target_n,
    )

    # Read embedded image bytes strictly for selected rows to avoid loading 5K+ PNGs into RAM.
    rows_by_shard: dict[int, list[int]] = {}
    for item in selected:
        rows_by_shard.setdefault(int(item["_shard_idx"]), []).append(int(item["_row_idx"]))
    img_by_loc: dict[tuple[int, int], bytes] = {}
    for s_idx, r_idxs in rows_by_shard.items():
        img_col = pq.read_table(shard_paths[s_idx], columns=["image"]).column("image")
        taken = img_col.take(r_idxs).to_pylist()
        for r_idx, img_val in zip(r_idxs, taken):
            img_by_loc[(s_idx, r_idx)] = _extract_image_bytes(img_val)

    out: list[dict[str, Any]] = []
    for item in selected:
        s_idx = int(item["_shard_idx"])
        r_idx = int(item["_row_idx"])
        out.append(
            {
                "id": item["id"],
                "filename": item["filename"],
                "imgid": item["imgid"],
                "rows": item["rows"],
                "cols": item["cols"],
                "has_span": item["has_span"],
                "size_bucket": item["size_bucket"],
                "gold_html": reconstruct_fintabnet_html(item["_html"], item["_cells"]),
                "image_bytes": img_by_loc.get((s_idx, r_idx), b""),
            }
        )
    return out


def load_omnidocbench_en(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    download_images: bool = True,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load English subset of ``opendatalab/OmniDocBench`` with ordered layout/table/formula annotations."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["omnidocbench_en"]
    t_cfg = ds.get(tier, ds.get("tier1", {}))
    target_n = int(max_items) if max_items is not None else (int(t_cfg["n"]) if t_cfg.get("n") is not None else None)
    if target_n is not None and target_n <= 0:
        return []
    lang_filter = str(ds.get("language_filter") or "english").strip().lower()

    json_path = download_pinned_file(
        "omnidocbench_en",
        ds["files"][0],
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    raw_pages = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_pages, list):
        raise ValueError(f"Expected JSON list in {json_path}")

    items: list[dict[str, Any]] = []
    for idx, page in enumerate(raw_pages):
        pinfo = page.get("page_info") or {}
        pattr = pinfo.get("page_attribute") or {}
        lang = str(pattr.get("language") or pinfo.get("language") or "").strip().lower()
        if lang_filter and lang != lang_filter:
            continue
        img_name = str(pinfo.get("image_path") or "").strip()
        if not img_name:
            continue
        data_source = str(pattr.get("data_source") or pinfo.get("data_source") or "unknown").strip()

        raw_dets = [d for d in (page.get("layout_dets") or []) if isinstance(d, Mapping) and not d.get("ignore")]
        raw_dets.sort(key=lambda d: (int(d["order"]) if d.get("order") is not None else 10**9, str(d.get("anno_id") or "")))

        text_blocks: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        equations: list[dict[str, Any]] = []
        for d in raw_dets:
            ctype = str(d.get("category_type") or "")
            order_val = d.get("order")
            if ctype in ("text_block", "title", "reference", "figure_caption", "table_caption", "equation_caption", "code_txt"):
                txt = str(d.get("text") or "").strip()
                if txt:
                    text_blocks.append({"category_type": ctype, "order": order_val, "text": txt})
            elif ctype == "table":
                html_s = str(d.get("html") or d.get("text") or "").strip()
                if html_s:
                    tables.append({"order": order_val, "html": html_s})
            elif ctype in ("equation_isolated", "equation_semantic"):
                ltx = str(d.get("latex") or d.get("text") or "").strip()
                if ltx:
                    equations.append({"category_type": ctype, "order": order_val, "latex": ltx})

        if not (text_blocks or tables or equations):
            continue
        items.append(
            {
                "id": img_name,
                "page_no": int(pinfo.get("page_no") or idx),
                "image_name": img_name,
                "image_repo_path": f"images/{img_name}",
                "image_path": None,
                "data_source": data_source,
                "layout": str(pattr.get("layout") or ""),
                "subset": str(pattr.get("subset") or ""),
                "text_blocks": text_blocks,
                "tables": tables,
                "equations": equations,
            }
        )

    selected = stratified_sha256_sample(
        items,
        dataset_key=f"omnidocbench_en:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["data_source"],
        total_n=target_n,
    )

    if download_images:
        for item in selected:
            img_p = download_pinned_file(
                "omnidocbench_en",
                item["image_repo_path"],
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                verify_checksum=False,
            )
            item["image_path"] = str(img_p)
    return selected


def load_cord_v2(
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``naver-clova-ix/cord-v2`` test receipts normalized for ``grounded_field_f1`` + ANLS."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["cord_v2"]
    t_cfg = ds.get(tier, ds.get("tier1", {}))
    target_n = int(max_items) if max_items is not None else (int(t_cfg["n"]) if t_cfg.get("n") is not None else None)
    if target_n is not None and target_n <= 0:
        return []

    p = download_pinned_file(
        "cord_v2",
        ds["files"][0],
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    gt_col = pq.read_table(p, columns=["ground_truth"]).column("ground_truth").to_pylist()
    schema_obj = build_cord_v2_schema()
    candidates: list[dict[str, Any]] = []
    for idx, raw_gt in enumerate(gt_col):
        parsed = json.loads(raw_gt) if isinstance(raw_gt, str) else (raw_gt or {})
        gt_parse = parsed.get("gt_parse") if isinstance(parsed, Mapping) else {}
        norm = normalize_cord_v2_parse(gt_parse if isinstance(gt_parse, Mapping) else {})
        n_menu = len(norm.get("menu") or [])
        bucket = "1" if n_menu <= 1 else ("2_4" if n_menu <= 4 else "5plus")
        candidates.append(
            {
                "id": f"cord_test_{idx:04d}",
                "_row_idx": idx,
                "menu_bucket": bucket,
                "gt_parse": norm,
                "schema": schema_obj,
            }
        )

    selected = stratified_sha256_sample(
        candidates,
        dataset_key=f"cord_v2:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["menu_bucket"],
        total_n=target_n,
    )

    sel_idxs = [int(it["_row_idx"]) for it in selected]
    img_vals = pq.read_table(p, columns=["image"]).column("image").take(sel_idxs).to_pylist()
    out: list[dict[str, Any]] = []
    for item, img_val in zip(selected, img_vals):
        out.append(
            {
                "id": item["id"],
                "menu_bucket": item["menu_bucket"],
                "gt_parse": item["gt_parse"],
                "schema": item["schema"],
                "image_bytes": _extract_image_bytes(img_val),
            }
        )
    return out


DOCLAYNET_CLASS_LABELS: dict[int, str] = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}

_NON_LATIN_SKIP_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]")


def _is_latin_dominant(text: str, *, min_alpha: int = 12, min_ratio: float = 0.85) -> bool:
    if not text:
        return False
    alpha = [ch for ch in text if ch.isalpha()]
    if len(alpha) < min_alpha:
        return False
    if _NON_LATIN_SKIP_RE.search(text):
        latin = sum(1 for ch in alpha if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("\u00c0" <= ch <= "\u024f"))
        return (float(latin) / float(len(alpha))) >= min_ratio
    latin = sum(1 for ch in alpha if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("\u00c0" <= ch <= "\u024f"))
    return (float(latin) / float(len(alpha))) >= min_ratio


def build_vrdu_schema(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Build Draft-2020-12 JSON Schema for a VRDU corpus from ``main/meta.json``."""
    match_map = dict(meta.get("entity_name_to_match_func") or {})
    patterns = dict(meta.get("entity_appearance_pattern") or {})
    unrepeated_keys = [k for k in match_map if patterns.get(k, "unrepeated") != "line_item"]
    line_item_keys = [k for k in match_map if patterns.get(k) == "line_item"]

    opt_str: dict[str, Any] = {"type": ["string", "null"]}
    props: dict[str, Any] = {k: dict(opt_str) for k in unrepeated_keys}
    required = list(unrepeated_keys)
    if line_item_keys:
        props["line_items"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: dict(opt_str) for k in line_item_keys},
                "additionalProperties": False,
            },
        }
        required.append("line_items")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _clean_vrdu_mention(val: Any) -> str:
    return " ".join(str(val or "").split()).strip()


def _extract_vrdu_page_gt(
    annotations: Sequence[Any],
    meta: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Select the primary annotated page and build structured KIE ground truth visible on that page."""
    patterns = dict(meta.get("entity_appearance_pattern") or {})
    match_map = dict(meta.get("entity_name_to_match_func") or {})

    # Bucket mentions by page_idx so visual single-page OCR is scored strictly on visible fields.
    by_page: dict[int, dict[str, list[tuple[float, float, str]]]] = {}
    for entry in annotations or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        field_name = str(entry[0])
        if match_map and field_name not in match_map:
            continue
        mentions = entry[1] if isinstance(entry[1], Sequence) else []
        for m in mentions:
            if not isinstance(m, (list, tuple)) or not m:
                continue
            txt = _clean_vrdu_mention(m[0])
            if not txt:
                continue
            bbox = m[1] if len(m) > 1 and isinstance(m[1], (list, tuple)) and len(m[1]) >= 5 else [0, 0.0, 0.0, 0.0, 0.0]
            page_idx = int(bbox[0]) if bbox[0] is not None else 0
            y1 = float(bbox[2] or 0.0)
            x1 = float(bbox[1] or 0.0)
            by_page.setdefault(page_idx, {}).setdefault(field_name, []).append((y1, x1, txt))

    if not by_page:
        return 0, ({ "line_items": [] } if any(v == "line_item" for v in patterns.values()) else {})

    best_page = min(
        by_page.keys(),
        key=lambda p_idx: (-len(by_page[p_idx]), p_idx),
    )
    page_fields = by_page[best_page]

    gt_parse: dict[str, Any] = {}
    line_item_fields = [k for k in match_map if patterns.get(k) == "line_item"]
    for k in match_map:
        if patterns.get(k, "unrepeated") == "line_item":
            continue
        vals = sorted(page_fields.get(k) or [], key=lambda t: (t[0], t[1]))
        dedup: list[str] = []
        for _, _, v in vals:
            if v not in dedup:
                dedup.append(v)
        if dedup:
            gt_parse[k] = dedup[0]

    if line_item_fields:
        col_lists: dict[str, list[str]] = {}
        max_rows = 0
        for k in line_item_fields:
            vals = [t[2] for t in sorted(page_fields.get(k) or [], key=lambda t: (t[0], t[1]))]
            col_lists[k] = vals
            max_rows = max(max_rows, len(vals))
        rows: list[dict[str, str]] = []
        for i in range(max_rows):
            row_obj: dict[str, str] = {}
            for k in line_item_fields:
                if i < len(col_lists[k]):
                    row_obj[k] = col_lists[k][i]
            if row_obj:
                rows.append(row_obj)
        gt_parse["line_items"] = rows

    return best_page, gt_parse


def load_vrdu(
    dataset_key: str = "vrdu_ad_buy",
    *,
    tier: str = "tier1",
    max_items: int | None = None,
    download_pdfs: bool = True,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``google-research-datasets/vrdu`` (``vrdu_ad_buy`` or ``vrdu_registration``) KIE items."""
    if dataset_key not in ("vrdu_ad_buy", "vrdu_registration"):
        raise ValueError(f"Unsupported VRDU dataset_key={dataset_key!r}")
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"][dataset_key]
    t_cfg = ds.get(tier, ds.get("tier1", {}))
    target_n = int(max_items) if max_items is not None else (int(t_cfg["n"]) if t_cfg.get("n") is not None else None)
    if target_n is not None and target_n <= 0:
        return []

    subdir = str(ds.get("subdir") or ("ad-buy-form" if dataset_key == "vrdu_ad_buy" else "registration-form"))
    data_file = str(ds.get("files", [f"{subdir}/main/dataset.jsonl.gz"])[0])
    meta_file = str(ds.get("files", ["", f"{subdir}/main/meta.json"])[1])
    split_file = str(ds.get("split_file") or "")

    data_p = download_pinned_file(dataset_key, data_file, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    meta_p = download_pinned_file(dataset_key, meta_file, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    schema = build_vrdu_schema(meta)

    allowed_test_files: set[str] | None = None
    if split_file:
        split_p = download_pinned_file(dataset_key, split_file, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
        split_obj = json.loads(split_p.read_text(encoding="utf-8"))
        if isinstance(split_obj.get("test"), list) and split_obj["test"]:
            allowed_test_files = {str(x) for x in split_obj["test"]}

    raw_bytes = data_p.read_bytes()
    text_blob = (gzip.decompress(raw_bytes) if data_p.suffix == ".gz" else raw_bytes).decode("utf-8")

    candidates: list[dict[str, Any]] = []
    for idx, line in enumerate(text_blob.splitlines()):
        s = line.strip()
        if not s:
            continue
        rec = json.loads(s)
        fname = str(rec.get("filename") or "").strip()
        if not fname:
            continue
        if allowed_test_files is not None and fname not in allowed_test_files:
            continue
        page_idx, gt_parse = _extract_vrdu_page_gt(rec.get("annotations") or [], meta)
        if not gt_parse:
            continue
        n_fields = len([k for k, v in gt_parse.items() if k != "line_items" and v]) + len(gt_parse.get("line_items") or [])
        bucket = "rich" if n_fields >= 5 else "sparse"
        candidates.append(
            {
                "id": fname,
                "row_idx": idx,
                "filename": fname,
                "pdf_repo_path": f"{subdir}/main/pdfs/{fname}",
                "pdf_path": None,
                "page_index": page_idx,
                "bucket": bucket,
                "meta": meta,
                "schema": schema,
                "gt_parse": gt_parse,
            }
        )

    selected = stratified_sha256_sample(
        candidates,
        dataset_key=f"{dataset_key}:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["bucket"],
        total_n=target_n,
    )

    if download_pdfs:
        for item in selected:
            pdf_p = download_pinned_file(
                dataset_key,
                item["pdf_repo_path"],
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                verify_checksum=False,
            )
            item["pdf_path"] = str(pdf_p)
    return selected


def load_doclaynet_gov(
    *,
    tier: str = "tier2",
    max_items: int | None = None,
    english_only: bool | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``laws_and_regulations``, ``government_tenders``, and ``patents`` pages from ``docling-project/DocLayNet-v1.2``."""
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["doclaynet_gov"]
    t_cfg = ds.get(tier, ds.get("tier2", {}))
    target_n = int(max_items) if max_items is not None else (int(t_cfg["n"]) if t_cfg.get("n") is not None else None)
    if target_n is not None and target_n <= 0:
        return []

    use_en = bool(ds.get("english_only", True) if english_only is None else english_only)
    allowed_cats = set(str(c) for c in (ds.get("doc_categories") or ("laws_and_regulations", "government_tenders", "patents")))
    shard_files = list(t_cfg.get("files") or ds.get("files") or [])

    candidates: list[dict[str, Any]] = []
    shard_tables: dict[str, Path] = {}
    for rel_fn in shard_files:
        pq_path = download_pinned_file(
            "doclaynet_gov",
            rel_fn,
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        shard_tables[rel_fn] = pq_path
        meta_tbl = pq.read_table(pq_path, columns=["metadata", "category_id", "bboxes", "pdf_cells"])
        metas = meta_tbl.column("metadata").to_pylist()
        cids_col = meta_tbl.column("category_id").to_pylist()
        bboxes_col = meta_tbl.column("bboxes").to_pylist()
        cells_col = meta_tbl.column("pdf_cells").to_pylist()

        for row_idx, (meta, cids, bboxes, cell_groups) in enumerate(zip(metas, cids_col, bboxes_col, cells_col)):
            if not isinstance(meta, Mapping):
                continue
            doc_cat = str(meta.get("doc_category") or "")
            if doc_cat not in allowed_cats:
                continue
            collection = str(meta.get("collection") or doc_cat)
            text_blocks: list[dict[str, Any]] = []
            table_texts: list[str] = []
            header_footer_texts: list[str] = []
            all_parts: list[str] = []

            for blk_i, cid in enumerate(cids or []):
                cid_int = int(cid)
                label = DOCLAYNET_CLASS_LABELS.get(cid_int, "Text")
                cells = cell_groups[blk_i] if (cell_groups and blk_i < len(cell_groups)) else []
                blk_txt = " ".join(str(c.get("text") or "").strip() for c in (cells or []) if isinstance(c, Mapping) and str(c.get("text") or "").strip()).strip()
                if not blk_txt:
                    continue
                all_parts.append(blk_txt)
                bbox = list(bboxes[blk_i]) if (bboxes and blk_i < len(bboxes)) else []
                if cid_int in (5, 6):
                    header_footer_texts.append(blk_txt)
                elif cid_int == 9:
                    table_texts.append(blk_txt)
                elif cid_int != 7:
                    text_blocks.append({"category_type": label, "order": blk_i, "text": blk_txt, "bbox": bbox})

            joined_all = " ".join(all_parts)
            if not text_blocks and not table_texts:
                continue
            if use_en and not _is_latin_dominant(joined_all):
                continue

            img_id = meta.get("image_id")
            orig_name = str(meta.get("original_filename") or f"row_{row_idx}")
            page_no = int(meta.get("page_no") or 0)
            candidates.append(
                {
                    "id": f"{collection}:{orig_name}:p{page_no}:{img_id}",
                    "_shard": rel_fn,
                    "_row_idx": row_idx,
                    "doc_category": doc_cat,
                    "collection": collection,
                    "page_no": page_no,
                    "text_blocks": text_blocks,
                    "table_texts": table_texts,
                    "header_footer_texts": header_footer_texts,
                }
            )

    selected = stratified_sha256_sample(
        candidates,
        dataset_key=f"doclaynet_gov:{tier}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["collection"],
        total_n=target_n,
    )

    by_shard: dict[str, list[dict[str, Any]]] = {}
    for item in selected:
        by_shard.setdefault(item["_shard"], []).append(item)

    for rel_fn, shard_items in by_shard.items():
        pq_path = shard_tables[rel_fn]
        sch_names = set(pq.read_schema(pq_path).names)
        cols = [c for c in ("image", "pdf") if c in sch_names]
        if not cols:
            continue
        idxs = [int(it["_row_idx"]) for it in shard_items]
        taken = pq.read_table(pq_path, columns=cols).take(idxs)
        img_col = taken.column("image").to_pylist() if "image" in cols else [None] * len(shard_items)
        pdf_col = taken.column("pdf").to_pylist() if "pdf" in cols else [None] * len(shard_items)
        for it, img_val, pdf_val in zip(shard_items, img_col, pdf_col):
            it["image_bytes"] = _extract_image_bytes(img_val) if img_val is not None else b""
            it["pdf_bytes"] = bytes(pdf_val) if isinstance(pdf_val, (bytes, bytearray, memoryview)) else b""

    out: list[dict[str, Any]] = []
    for it in selected:
        out.append({k: v for k, v in it.items() if not k.startswith("_")})
    return out


def _puredocbench_is_english(lang_val: Any) -> bool:
    if isinstance(lang_val, list):
        return any("en" == str(x).strip().lower() or "english" == str(x).strip().lower() for x in lang_val)
    low = str(lang_val or "").strip().lower()
    return low in ("english", "en")


def load_puredocbench_legal_gov(
    *,
    track: str = "clean",
    max_items: int | None = None,
    english_only: bool | None = None,
    images_root: str | Path | None = None,
    lock: Mapping[str, Any] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Load ``03_legal_gov`` annotations from ``zhihengli-casia/puredocbench`` normalized for OmniDocBench scoring."""
    eff_track = str(track or "clean").strip().lower()
    if eff_track not in ("clean", "digital", "real"):
        raise ValueError(f"Unsupported PureDocBench track={track!r}; expected 'clean', 'digital', or 'real'")
    if lock is None:
        lock, _ = load_lockfile(lock_path)
    ds = lock["datasets"]["puredocbench_legal_gov"]
    use_en = bool(ds.get("english_only", True) if english_only is None else english_only)
    cat_filter = str(ds.get("category_filter") or "03_legal_gov")
    files = list(ds.get("files") or ["gt/puredocbench_gt_latest.tar.gz", "release_manifest_candidate_1475.csv"])

    gt_tar = download_pinned_file("puredocbench_legal_gov", files[0], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    manifest_p = download_pinned_file("puredocbench_legal_gov", files[1], lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)

    manifest_map: dict[str, dict[str, str]] = {}
    with manifest_p.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            gt_rel = str(row.get("gt_rel") or "").strip()
            if gt_rel:
                manifest_map[gt_rel] = dict(row)

    eff_img_root = Path(images_root).expanduser() if images_root is not None else (resolve_cache_dir(cache_dir) / "puredocbench_legal_gov" / "images")
    rel_col = f"{eff_track}_rel"

    items: list[dict[str, Any]] = []
    prefix = f"gt/{cat_filter}/"
    with tarfile.open(gt_tar, "r:gz") as tf:
        for member in tf.getmembers():
            if not (member.isfile() and member.name.startswith(prefix) and member.name.endswith(".json")):
                continue
            raw_fp = tf.extractfile(member)
            if raw_fp is None:
                continue
            page = json.load(raw_fp)
            pinfo = page.get("page_info") or {}
            pattr = pinfo.get("page_attribute") or {}
            if use_en and not _puredocbench_is_english(pattr.get("language")):
                continue

            gt_rel = member.name[len("gt/") :]
            m_row = manifest_map.get(gt_rel, {})
            img_rel = str(m_row.get(rel_col) or pinfo.get("image_path") or "").strip()
            img_path = (eff_img_root / img_rel) if img_rel else None

            raw_dets = [d for d in (page.get("layout_dets") or []) if isinstance(d, Mapping) and not d.get("ignore")]
            raw_dets.sort(key=lambda d: (int(d["order"]) if d.get("order") is not None else 10**9, str(d.get("anno_id") or "")))

            text_blocks: list[dict[str, Any]] = []
            tables: list[dict[str, Any]] = []
            equations: list[dict[str, Any]] = []
            for d in raw_dets:
                ctype = str(d.get("category_type") or "")
                order_val = d.get("order")
                if ctype in ("text_block", "title", "reference", "figure_caption", "table_caption", "equation_caption", "code_txt"):
                    txt = str(d.get("text") or "").strip()
                    if txt:
                        text_blocks.append({"category_type": ctype, "order": order_val, "text": txt})
                elif ctype == "table":
                    html_s = str(d.get("html") or d.get("text") or "").strip()
                    if html_s:
                        tables.append({"order": order_val, "html": html_s})
                elif ctype in ("equation_isolated", "equation_semantic"):
                    ltx = str(d.get("latex") or d.get("text") or "").strip()
                    if ltx:
                        equations.append({"category_type": ctype, "order": order_val, "latex": ltx})

            if not (text_blocks or tables or equations):
                continue
            subcat = str(m_row.get("subcategory") or Path(gt_rel).parts[1] if len(Path(gt_rel).parts) > 1 else "legal_gov")
            items.append(
                {
                    "id": str(m_row.get("page_id") or gt_rel),
                    "domain": cat_filter,
                    "subcategory": subcat,
                    "track": eff_track,
                    "image_rel": img_rel,
                    "image_path": str(img_path) if (img_path is not None and img_path.is_file()) else None,
                    "data_source": str(pattr.get("data_source") or "synthetic_legal"),
                    "text_blocks": text_blocks,
                    "tables": tables,
                    "equations": equations,
                }
            )

    return stratified_sha256_sample(
        items,
        dataset_key=f"puredocbench_legal_gov:{eff_track}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["subcategory"],
        total_n=max_items,
    )


def stage_tier_datasets(
    tier: str = "tier1",
    *,
    datasets: Sequence[str] | None = None,
    lock_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Download & verify all pinned files required for ``tier`` into local cache (0 GPU work)."""
    if tier not in ("tier1", "tier2"):
        raise ValueError(f"Unsupported tier={tier!r}; expected 'tier1' or 'tier2'")
    lock, lock_sha = load_lockfile(lock_path)
    eff_cache = resolve_cache_dir(cache_dir)
    target_keys = list(datasets) if datasets else list(lock["datasets"].keys())

    staged_report: dict[str, Any] = {
        "tier": tier,
        "cache_dir": str(eff_cache),
        "lockfile_sha256": lock_sha,
        "datasets": {},
    }

    for key in target_keys:
        if key not in lock["datasets"]:
            raise ValueError(f"Unknown dataset key {key!r}")
        ds = lock["datasets"][key]
        files_to_fetch: list[str] = []

        if key == "nanobeir_en":
            subs = ds["tier1_subsets"] if tier == "tier1" else ds["tier2_subsets"]
            for sub in subs:
                for kind in ("corpus", "queries", "qrels"):
                    files_to_fetch.append(f"{kind}/{sub}-00000-of-00001.parquet")
        elif key == "legalbench_rag":
            t_cfg = ds[tier]
            for sub in filter_legalbench_subsets(t_cfg["subsets"]):
                files_to_fetch.append(f"benchmarks/{sub}.json")
        elif key == "longembed":
            t_cfg = ds[tier]
            for sub in filter_longembed_subsets(t_cfg.get("subsets", [])):
                for part in ("corpus", "queries", "qrels"):
                    files_to_fetch.append(f"{sub}/{part}.jsonl")
        elif key == "mmlu_redux_2":
            for sub in ds.get("subjects", []):
                files_to_fetch.append(f"{sub}/data-00000-of-00001.arrow")
        elif key == "bfcl_v3":
            for sp in ds["splits"].values():
                files_to_fetch.append(sp["questions"])
                if sp.get("answers"):
                    files_to_fetch.append(sp["answers"])
        elif key == "liquidai_extraction":
            t_cfg = ds[tier]
            if int(t_cfg.get("test_unseen", 0)) > 0:
                files_to_fetch.append("test_unseen.parquet")
            if int(t_cfg.get("test", 0)) > 0:
                files_to_fetch.append("test.parquet")
        elif key in ("fintabnet_otsl", "doclaynet_gov"):
            t_cfg = ds.get(tier, {})
            n_val = t_cfg.get("n")
            if n_val is None or int(n_val) > 0:
                files_to_fetch.extend(t_cfg.get("files") or ds.get("files", []))
        else:
            t_cfg = ds.get(tier, {})
            n_val = t_cfg.get("n")
            if n_val is None or int(n_val) > 0:
                files_to_fetch.extend(ds.get("files", []))

        downloaded_count = 0
        total_bytes = 0
        for fn in files_to_fetch:
            p = download_pinned_file(
                key,
                fn,
                lock=lock,
                cache_dir=eff_cache,
                local_files_only=False,
                verify_checksum=verify_checksums,
            )
            downloaded_count += 1
            total_bytes += int(p.stat().st_size)

        # For LegalBench-RAG, also stage the deterministic corpus files for this tier!
        if key == "legalbench_rag" and files_to_fetch:
            lb_data = load_legalbench_rag(tier=tier, lock=lock, cache_dir=eff_cache, local_files_only=False)
            for doc_id in lb_data["corpus_texts"]:
                cp = eff_cache / "legalbench_rag" / "corpus" / doc_id
                if cp.is_file():
                    downloaded_count += 1
                    total_bytes += int(cp.stat().st_size)
        elif key == "olmocr_bench" and files_to_fetch:
            olm_pages = load_olmocr_bench(tier=tier, download_pdfs=True, lock=lock, cache_dir=eff_cache, local_files_only=False)
            for item in olm_pages:
                pp = Path(item["pdf_path"]) if item.get("pdf_path") else None
                if pp is not None and pp.is_file():
                    downloaded_count += 1
                    total_bytes += int(pp.stat().st_size)
        elif key == "omnidocbench_en" and files_to_fetch:
            od_pages = load_omnidocbench_en(tier=tier, download_images=True, lock=lock, cache_dir=eff_cache, local_files_only=False)
            for item in od_pages:
                ip = Path(item["image_path"]) if item.get("image_path") else None
                if ip is not None and ip.is_file():
                    downloaded_count += 1
                    total_bytes += int(ip.stat().st_size)
        elif key in ("vrdu_ad_buy", "vrdu_registration") and files_to_fetch:
            vrdu_items = load_vrdu(key, tier=tier, download_pdfs=True, lock=lock, cache_dir=eff_cache, local_files_only=False)
            for item in vrdu_items:
                pp = Path(item["pdf_path"]) if item.get("pdf_path") else None
                if pp is not None and pp.is_file():
                    downloaded_count += 1
                    total_bytes += int(pp.stat().st_size)

        staged_report["datasets"][key] = {
            "repo_id": ds["repo_id"],
            "revision": ds["revision"],
            "license": ds["license"],
            "files_staged": downloaded_count,
            "bytes_staged": total_bytes,
            "skipped_for_tier": downloaded_count == 0,
        }

    return staged_report


__all__ = [
    "ALLOWED_LEGALBENCH_SUBSETS",
    "ALLOWED_LONGEMBED_SUBSETS",
    "ALLOWED_REWARDBENCH2_SUBSETS",
    "CLINC150_DOMAINS",
    "CLINC150_INTENT_TO_DOMAIN",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LOCK_FILENAME",
    "DOCLAYNET_CLASS_LABELS",
    "FORBIDDEN_LEGALBENCH_SUBSETS",
    "FORBIDDEN_LONGEMBED_SUBSETS",
    "OLMOCR_BENCH_CATEGORIES",
    "build_cord_v2_schema",
    "build_vrdu_schema",
    "compute_file_sha256",
    "download_pinned_file",
    "filter_legalbench_subsets",
    "filter_longembed_subsets",
    "load_banking77",
    "load_bfcl_v3",
    "load_clinc_oos",
    "load_cord_v2",
    "load_doclaynet_gov",
    "load_extractbench",
    "load_fintabnet_otsl",
    "load_ifbench_test",
    "load_ifeval",
    "load_jsonschemabench",
    "load_legalbench_rag",
    "load_liquidai_extraction",
    "load_livebench_reasoning",
    "load_lockfile",
    "load_longembed_subset",
    "load_math_500",
    "load_mmlu_redux",
    "load_nanobeir_subset",
    "load_olmocr_bench",
    "load_omnidocbench_en",
    "load_puredocbench_legal_gov",
    "load_ragtruth",
    "load_reward_bench_2",
    "load_vrdu",
    "normalize_cord_v2_parse",
    "read_arrow_ipc_rows",
    "read_jsonl_rows",
    "read_parquet_class_names",
    "read_parquet_rows",
    "reconstruct_fintabnet_html",
    "resolve_cache_dir",
    "resolve_lockfile_path",
    "sha256_sort_key",
    "stage_tier_datasets",
    "stratified_sha256_sample",
]
