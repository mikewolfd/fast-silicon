"""Paired authoritative U.S. Government PDF <-> XML / transcription benchmark builder & loader.

Builds deterministic, SHA-256 verified page benchmarks across all three U.S. federal branches
using keyless Public-Domain sources:
- Legislative: GovInfo ``PLAW`` (USLM XML + PDF) and ``BILLS`` (XML + PDF)
- Executive: Federal Register API + GovInfo ``FR`` (3-column PDF + GPO XML / ``GPOTABLE``)
  and optional LoC ``By the People`` (``btp-dataset``) archival scans
- Judicial: GovInfo ``USCOURTS`` born-digital CM/ECF opinions (visual render vs native text)
"""

from __future__ import annotations

import hashlib
import html as _html
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


def _http_get_bytes(url: str, *, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
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
        text = _clean_ws(str(r.get("Transcription") or r.get("FullText") or ""))
        img_url = str(r.get("DownloadURL") or r.get("TiffUrl") or "").strip()
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


def stage_us_gov_ocr(
    *,
    branches: Sequence[str] = ("legislative", "executive", "judicial"),
    per_source: int = 3,
    year: int = 2025,
    cache_dir: str | Path | None = None,
    loc_btp_path: str | Path | None = None,
) -> dict[str, Any]:
    """Download and verify paired Public-Domain U.S. Gov PDF<->XML items into ``cache_dir/us_gov_ocr``."""
    eff_branches = tuple(str(b).strip().lower() for b in branches if str(b).strip())
    for b in eff_branches:
        if b not in ("legislative", "executive", "judicial"):
            raise ValueError(f"Unsupported U.S. Gov branch {b!r}")

    root = resolve_cache_dir(cache_dir) / "us_gov_ocr"
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    if "legislative" in eff_branches:
        items.extend(_stage_govinfo_sitemap_items(root, collection="PLAW", branch="legislative", year=min(year, 2024), per_source=per_source))
        items.extend(_stage_govinfo_sitemap_items(root, collection="BILLS", branch="legislative", year=year, per_source=per_source))
    if "executive" in eff_branches:
        items.extend(_stage_fr_items(root, per_source=per_source))
        if loc_btp_path is not None:
            for it in load_loc_btp_items(
                loc_btp_path,
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
        items.extend(_stage_uscourts_items(root, year=year, per_source=per_source))

    manifest = {"schema_version": 1, "year": int(year), "items": items}
    manifest_p = root / "manifest.json"
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    manifest_sha = _write_cached_bytes(manifest_p, manifest_bytes)
    sources = sorted({str(it.get("source") or "unknown") for it in items})
    return {
        "cache_dir": str(root),
        "manifest_path": str(manifest_p),
        "manifest_sha256": manifest_sha,
        "n_items": len(items),
        "by_branch": {b: sum(1 for it in items if it["branch"] == b) for b in eff_branches},
        "by_source": {s: sum(1 for it in items if it.get("source") == s) for s in sources},
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
        stratum_fn=lambda r: str(r.get("branch") or "unknown"),
        total_n=max_items,
    )


__all__ = [
    "extract_born_digital_pdf_page_blocks",
    "gpotable_to_html",
    "load_loc_btp_items",
    "load_us_gov_ocr",
    "parse_govinfo_xml_gold",
    "slice_gold_to_page_prefix",
    "stage_us_gov_ocr",
]
