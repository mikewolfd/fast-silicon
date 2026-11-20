"""Pure-Arrow / JSONL HuggingFace dataset loader with immutable SHA pinning and SHA-256 stratified sampling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    "FORBIDDEN_LEGALBENCH_SUBSETS",
    "FORBIDDEN_LONGEMBED_SUBSETS",
    "compute_file_sha256",
    "download_pinned_file",
    "filter_legalbench_subsets",
    "filter_longembed_subsets",
    "load_banking77",
    "load_bfcl_v3",
    "load_clinc_oos",
    "load_extractbench",
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
    "load_ragtruth",
    "load_reward_bench_2",
    "read_arrow_ipc_rows",
    "read_jsonl_rows",
    "read_parquet_class_names",
    "read_parquet_rows",
    "resolve_cache_dir",
    "resolve_lockfile_path",
    "sha256_sort_key",
    "stage_tier_datasets",
    "stratified_sha256_sample",
]
