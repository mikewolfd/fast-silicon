"""Category 3 deterministic scorers: IFEval+IFBench (83 pure-Python verifiers), MMLU-Redux, MATH-500, LiveBench."""

from __future__ import annotations

import ast
import csv
from fractions import Fraction
import io
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import numpy as np

# ---------------------------------------------------------------------------
# Pure-Python Tokenization, Sentence Splitting & Repeat-Loop Detection
# ---------------------------------------------------------------------------

_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Fig|No|Vol|Dept|Inc|Ltd|Co|U\.S|U\.K)\.$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[\u3040-\u30ff\u4e00-\u9fff]+")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z0-9_])")
_JAPANESE_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]")

_COORDINATING_CONJUNCTIONS: frozenset[str] = frozenset({"for", "and", "nor", "but", "or", "yet", "so"})

_COMMON_PRONOUNS: frozenset[str] = frozenset(
    {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "that", "this", "these", "those",
        "someone", "somebody", "something", "anyone", "anybody", "anything",
        "everyone", "everybody", "everything", "no one", "nobody", "nothing",
        "each", "either", "neither", "one", "ones", "other", "others", "another",
    }
)

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could",
        "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
        "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
        "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
        "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
        "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    }
)

_MINOR_TITLE_WORDS: frozenset[str] = frozenset(
    {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "nor", "of", "on", "or", "per", "the", "to", "up", "via"}
)

_COMMON_VERBS: frozenset[str] = frozenset(
    {
        "accept", "achieve", "act", "add", "allow", "analyze", "answer", "appear", "apply", "argue", "ask",
        "assume", "avoid", "be", "become", "begin", "believe", "bring", "build", "buy", "call", "can", "carry",
        "catch", "cause", "change", "check", "choose", "claim", "close", "come", "compare", "consider", "contain",
        "continue", "control", "create", "cut", "deal", "decide", "define", "deliver", "describe", "design",
        "determine", "develop", "die", "discover", "discuss", "do", "draw", "drive", "drop", "eat", "enable",
        "encourage", "end", "enjoy", "ensure", "enter", "establish", "evaluate", "examine", "exist", "expect",
        "experience", "explain", "explore", "express", "face", "fail", "fall", "feel", "fight", "fill", "find",
        "finish", "focus", "follow", "force", "forget", "form", "gain", "generate", "get", "give", "go", "grow",
        "handle", "happen", "have", "hear", "help", "hold", "hope", "identify", "imagine", "implement", "improve",
        "include", "increase", "indicate", "influence", "inform", "inspect", "inspire", "introduce", "investigate",
        "involve", "is", "join", "keep", "kill", "know", "launch", "lead", "learn", "leave", "let", "lie", "like",
        "limit", "list", "listen", "live", "look", "lose", "love", "maintain", "make", "manage", "matter", "may",
        "mean", "measure", "meet", "mention", "might", "mind", "miss", "move", "must", "need", "note", "notice",
        "observe", "obtain", "occur", "offer", "open", "operate", "order", "organize", "overcome", "own", "pass",
        "pay", "perform", "permit", "pick", "place", "plan", "play", "point", "prepare", "present", "prevent",
        "produce", "promise", "promote", "protect", "prove", "provide", "pull", "push", "put", "raise", "reach",
        "read", "realize", "receive", "recognize", "recommend", "record", "reduce", "refer", "reflect", "refuse",
        "regard", "relate", "release", "remain", "remember", "remove", "repeat", "replace", "reply", "report",
        "represent", "require", "research", "resolve", "respond", "rest", "result", "retain", "return", "reveal",
        "review", "rise", "run", "save", "say", "see", "seek", "seem", "select", "sell", "send", "separate",
        "serve", "set", "settle", "shall", "shape", "share", "shift", "should", "show", "shut", "sign", "sit",
        "sleep", "solve", "sort", "speak", "spend", "stand", "start", "state", "stay", "step", "stop", "store",
        "study", "succeed", "suffer", "suggest", "summarize", "supply", "support", "suppose", "survive", "take",
        "talk", "teach", "tell", "tend", "test", "thank", "think", "throw", "touch", "track", "train", "transfer",
        "transform", "travel", "treat", "try", "turn", "understand", "unite", "update", "use", "utilize", "validate",
        "value", "verify", "view", "visit", "wait", "walk", "want", "warn", "watch", "wear", "will", "win", "wish",
        "wonder", "work", "worry", "would", "write",
    }
)

# Canonical European capitals with latitude > 45 deg N (sorted highest to lowest latitude).
_EUROPEAN_CAPITALS_GT45: tuple[tuple[str, float], ...] = (
    ("reykjavik", 64.15),
    ("helsinki", 60.17),
    ("oslo", 59.91),
    ("tallinn", 59.44),
    ("stockholm", 59.33),
    ("riga", 56.95),
    ("moscow", 55.76),
    ("copenhagen", 55.68),
    ("vilnius", 54.69),
    ("minsk", 53.90),
    ("dublin", 53.35),
    ("berlin", 52.52),
    ("amsterdam", 52.37),
    ("warsaw", 52.23),
    ("london", 51.51),
    ("brussels", 50.85),
    ("kyiv", 50.45),
    ("kiev", 50.45),
    ("prague", 50.08),
    ("luxembourg", 49.61),
    ("paris", 48.86),
    ("vienna", 48.21),
    ("bratislava", 48.15),
    ("budapest", 47.50),
    ("vaduz", 47.14),
    ("chisinau", 47.01),
    ("bern", 46.95),
    ("ljubljana", 46.06),
    ("zagreb", 45.81),
)
_EUROPEAN_CAPITAL_LAT: dict[str, float] = {k: v for k, v in _EUROPEAN_CAPITALS_GT45}

_AFRICAN_COUNTRIES: frozenset[str] = frozenset(
    {
        "algeria", "angola", "benin", "botswana", "burkina faso", "burundi", "cabo verde", "cape verde",
        "cameroon", "central african republic", "chad", "comoros", "democratic republic of the congo",
        "republic of the congo", "congo", "cote d'ivoire", "ivory coast", "djibouti", "egypt",
        "equatorial guinea", "eritrea", "eswatini", "swaziland", "ethiopia", "gabon", "gambia", "ghana",
        "guinea", "guinea-bissau", "kenya", "lesotho", "liberia", "libya", "madagascar", "malawi", "mali",
        "mauritania", "mauritius", "morocco", "mozambique", "namibia", "niger", "nigeria", "rwanda",
        "sao tome and principe", "senegal", "seychelles", "sierra leone", "somalia", "south africa",
        "south sudan", "sudan", "tanzania", "togo", "tunisia", "uganda", "zambia", "zimbabwe",
    }
)


def tokenize_words(text: str) -> list[str]:
    """Pure-Python deterministic word tokenizer (zero ``nltk`` dependency)."""
    return _WORD_RE.findall(text or "")


def split_sentences(text: str) -> list[str]:
    """Deterministic regex/Punkt-rule sentence splitter (zero ``nltk_data`` download)."""
    raw = (text or "").strip()
    if not raw:
        return []
    sentences: list[str] = []
    for block in re.split(r"\n{2,}", raw):
        blk = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if not blk:
            continue
        cur: list[str] = []
        tokens = blk.split(" ")
        for tok in tokens:
            cur.append(tok)
            stripped = tok.rstrip("\"'”’)]}")
            if not stripped:
                continue
            ends_term = stripped.endswith((".", "!", "?", "?!", "!?"))
            if ends_term and not _ABBREV_RE.search(stripped) and not re.match(r"^\d+\.$", stripped):
                # Avoid splitting decimal numbers or single-letter initials like "A."
                if not re.match(r"^[A-Za-z]\.$", stripped):
                    sentences.append(" ".join(cur).strip())
                    cur = []
        if cur:
            sentences.append(" ".join(cur).strip())
    return [s for s in sentences if s]


def repeat_4gram_rate(tokens_or_text: Sequence[int] | str) -> float:
    """Fraction of 4-gram positions appearing >=4 times consecutively (stride 1..4) or >=8 times total."""
    if isinstance(tokens_or_text, str):
        seq: Sequence[Any] = tokenize_words(tokens_or_text.lower())
    else:
        seq = list(tokens_or_text)
    n = len(seq)
    if n < 4:
        return 0.0
    grams = [tuple(seq[i : i + 4]) for i in range(n - 3)]
    m = len(grams)
    counts: dict[tuple[Any, ...], int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1

    flagged = [counts[g] >= 8 for g in grams]
    # Detect >=4 consecutive repetitions at period 1, 2, 3, or 4 (e.g. AAAA... or ABCD ABCD ABCD ABCD).
    for stride in (1, 2, 3, 4):
        for offset in range(stride):
            run_start = offset
            while run_start < m:
                g0 = grams[run_start]
                run_end = run_start + stride
                run_len = 1
                while run_end < m and grams[run_end] == g0:
                    run_len += 1
                    run_end += stride
                if run_len >= 4:
                    for pos in range(run_start, run_end, stride):
                        flagged[pos] = True
                run_start = run_end
    return float(sum(1 for f in flagged if f)) / float(m)


def check_chat_validity(
    text: str,
    *,
    finish_reason: str = "stop",
    token_ids: Sequence[int] | None = None,
    repeat_threshold: float = 0.15,
) -> tuple[bool, float]:
    """Cat 3 item validity: non-empty, ``finish_reason != 'length'``, and ``repeat_4gram_rate < 0.15``."""
    rep = repeat_4gram_rate(token_ids if token_ids is not None else text)
    ok = bool((text or "").strip()) and str(finish_reason) != "length" and rep < float(repeat_threshold)
    return ok, rep


# ---------------------------------------------------------------------------
# Helpers for IFEval (25) + IFBench (58) = 83 Deterministic Verifiers
# ---------------------------------------------------------------------------


def _cmp_relation(actual: float, target: float, relation: str | None) -> bool:
    rel = (relation or "at least").strip().lower()
    if rel in ("less than", "<"):
        return actual < target
    if rel in ("at most", "<=", "less than or equal to"):
        return actual <= target
    if rel in ("more than", ">", "greater than"):
        return actual > target
    if rel in ("equal to", "exactly", "==", "="):
        return abs(actual - target) < 1e-9
    return actual >= target  # default: "at least"


def _count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    groups = re.findall(r"[aeiouy]{1,2}", w)
    return max(1, len(groups))


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _has_emoji(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (
            0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or 0x1F000 <= cp <= 0x1F2FF
            or 0xFE0F == cp
        ):
            return True
    return False


def _detect_language_code(text: str) -> str:
    """Lightweight deterministic script + function-word language detector (zero external downloads)."""
    s = (text or "").strip()
    if not s:
        return "unknown"
    n_cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    n_jp_kana = len(re.findall(r"[\u3040-\u30ff]", s))
    n_ko = len(re.findall(r"[\uac00-\ud7af]", s))
    n_cyr = len(re.findall(r"[\u0400-\u04ff]", s))
    n_ar = len(re.findall(r"[\u0600-\u06ff]", s))
    n_dev = len(re.findall(r"[\u0900-\u097f]", s))
    n_el = len(re.findall(r"[\u0370-\u03ff]", s))
    n_he = len(re.findall(r"[\u0590-\u05ff]", s))
    n_th = len(re.findall(r"[\u0e00-\u0e7f]", s))
    n_bn = len(re.findall(r"[\u0980-\u09ff]", s))
    n_ta = len(re.findall(r"[\u0b80-\u0bff]", s))
    n_te = len(re.findall(r"[\u0c00-\u0c7f]", s))
    n_kn = len(re.findall(r"[\u0c80-\u0cff]", s))
    n_gu = len(re.findall(r"[\u0a80-\u0aff]", s))
    n_pa = len(re.findall(r"[\u0a00-\u0a7f]", s))
    n_fa_hint = len(re.findall(r"[پچژگیک]", s))
    n_ur_hint = len(re.findall(r"[ٹڈڑںےھ]", s))

    if n_jp_kana >= 2:
        return "ja"
    if n_ko >= 2:
        return "ko"
    if n_cjk >= 2:
        return "zh-cn"
    if n_cyr >= 3:
        if re.search(r"[іїєґ]", s.lower()):
            return "uk"
        if re.search(r"[ъ]", s.lower()) and not re.search(r"[ыэ]", s.lower()):
            return "bg"
        return "ru"
    if n_ar >= 3:
        if n_ur_hint >= 1:
            return "ur"
        if n_fa_hint >= 1:
            return "fa"
        return "ar"
    if n_dev >= 3:
        return "hi"
    if n_el >= 3:
        return "el"
    if n_he >= 3:
        return "he"
    if n_th >= 3:
        return "th"
    if n_bn >= 3:
        return "bn"
    if n_ta >= 3:
        return "ta"
    if n_te >= 3:
        return "te"
    if n_kn >= 3:
        return "kn"
    if n_gu >= 3:
        return "gu"
    if n_pa >= 3:
        return "pa"

    words = set(re.findall(r"[a-zà-öø-ÿąćęłńóśźżşğıçñ]+", s.lower()))
    lang_markers: dict[str, set[str]] = {
        "en": {"the", "and", "is", "of", "to", "in", "that", "for", "with", "this"},
        "es": {"el", "la", "los", "las", "que", "de", "en", "para", "con", "por", "una", "del"},
        "fr": {"le", "la", "les", "des", "est", "que", "dans", "pour", "avec", "une", "pas"},
        "de": {"der", "die", "das", "und", "ist", "ein", "eine", "nicht", "mit", "für", "auf"},
        "it": {"il", "la", "che", "di", "per", "con", "una", "sono", "della", "anche", "nel"},
        "pt": {"que", "não", "para", "com", "uma", "por", "mais", "como", "dos", "das", "você"},
        "nl": {"de", "het", "een", "van", "en", "dat", "op", "voor", "met", "zijn", "niet"},
        "pl": {"się", "nie", "jest", "jak", "ale", "czy", "przez", "oraz", "które", "tylko"},
        "tr": {"bir", "ve", "bu", "için", "ile", "olarak", "daha", "çok", "değil"},
        "vi": {"của", "và", "các", "trong", "là", "cho", "không", "được", "một", "những"},
        "id": {"yang", "dan", "di", "untuk", "dengan", "tidak", "ini", "dari", "pada"},
        "sv": {"och", "att", "det", "som", "för", "med", "inte", "till", "och"},
        "fi": {"ja", "on", "että", "ei", "oli", "se", "kuin", "myös", "mutta"},
        "sw": {"na", "ya", "kwa", "wa", "ni", "katika", "la", "za"},
    }
    best_lang = "en"
    best_score = -1
    for code, markers in lang_markers.items():
        sc = len(words & markers)
        if sc > best_score:
            best_score = sc
            best_lang = code
    return best_lang


# ---------------------------------------------------------------------------
# IFEval (25 Verifiers)
# ---------------------------------------------------------------------------


def _v_kw_existence(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    kws = kw.get("keywords") or []
    low = resp.lower()
    return all(str(k).lower() in low for k in kws)


def _v_kw_frequency(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    word = str(kw.get("keyword") or "").lower()
    freq = float(kw.get("frequency") or 1)
    rel = kw.get("relation")
    if not word:
        return True
    cnt = len(re.findall(re.escape(word), resp.lower()))
    return _cmp_relation(float(cnt), freq, rel)


def _v_kw_forbidden(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    words = kw.get("forbidden_words") or []
    low = resp.lower()
    return all(not re.search(rf"\b{re.escape(str(w).lower())}\b", low) for w in words)


def _v_kw_letter_freq(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    letter = str(kw.get("letter") or "").lower()
    freq = float(kw.get("let_frequency") or 1)
    rel = kw.get("let_relation")
    if not letter:
        return True
    return _cmp_relation(float(resp.lower().count(letter)), freq, rel)


def _v_lang_response(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    target = str(kw.get("language") or "en").lower().split("-")[0]
    detected = _detect_language_code(resp).lower().split("-")[0]
    return detected == target


def _v_len_sentences(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    return _cmp_relation(float(len(split_sentences(resp))), float(kw.get("num_sentences") or 1), kw.get("relation"))


def _v_len_paragraphs(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    parts = [p.strip() for p in resp.split("***") if p.strip()]
    return len(parts) == int(kw.get("num_paragraphs") or 1)


def _v_len_words(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    return _cmp_relation(float(len(tokenize_words(resp))), float(kw.get("num_words") or 1), kw.get("relation"))


def _v_len_nth_para_first(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    n_p = int(kw.get("num_paragraphs") or 1)
    nth = int(kw.get("nth_paragraph") or 1)
    first_w = str(kw.get("first_word") or "").strip().lower()
    paras = [p.strip() for p in re.split(r"\n\s*\n", resp.strip()) if p.strip()]
    if len(paras) != n_p or not (1 <= nth <= len(paras)):
        return False
    wds = tokenize_words(paras[nth - 1])
    return bool(wds) and wds[0].lower() == first_w


def _v_content_placeholders(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    found = re.findall(r"\[[^\]]+\]", resp)
    return len(found) >= int(kw.get("num_placeholders") or 1)


def _v_content_postscript(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    marker = str(kw.get("postscript_marker") or "P.S.").strip()
    pat = re.escape(marker).replace(r"\.", r"\.?\s*")
    return bool(re.search(pat, resp, re.IGNORECASE))


def _v_fmt_bullets(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    cnt = sum(1 for ln in resp.splitlines() if re.match(r"^\s*[\*\-]\s+\S", ln))
    return cnt == int(kw.get("num_bullets") or 1)


def _v_fmt_constrained(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    opts = ("my answer is yes.", "my answer is no.", "my answer is maybe.")
    low = resp.strip().lower()
    return any(opt in low for opt in opts)


def _v_fmt_highlighted(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    n_req = int(kw.get("num_highlights") or 1)
    found = re.findall(r"\*\*[^*\n]+\*\*|\*[^*\n]+\*", resp)
    return len(found) >= n_req


def _v_fmt_multi_sections(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    spl = str(kw.get("section_spliter") or "SECTION").strip()
    n_req = int(kw.get("num_sections") or 1)
    found = re.findall(rf"{re.escape(spl)}\s*\d+", resp, re.IGNORECASE)
    return len(found) >= n_req


def _v_fmt_json(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    s = resp.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _v_fmt_title(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    return bool(re.search(r"<<[^<>\n]+>>", resp))


def _v_comb_two_responses(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    parts = [p.strip() for p in resp.split("******") if p.strip()]
    return len(parts) == 2 and parts[0].lower() != parts[1].lower()


def _v_comb_repeat_prompt(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    ptr = str(kw.get("prompt_to_repeat") or "").strip().lower()
    return bool(ptr) and resp.strip().lower().startswith(ptr)


def _v_startend_end(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    phrase = str(kw.get("end_phrase") or "").strip().lower()
    return bool(phrase) and resp.strip().rstrip("\"'”’").lower().endswith(phrase.rstrip("\"'”’"))


def _v_startend_quotation(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    s = resp.strip()
    return len(s) >= 2 and s[0] == '"' and s[-1] == '"'


def _v_case_cap_word_freq(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    words = _ALPHA_WORD_RE.findall(resp)
    all_caps = sum(1 for w in words if w.isupper() and len(w) >= 1)
    return _cmp_relation(float(all_caps), float(kw.get("capital_frequency") or 1), kw.get("capital_relation"))


def _v_case_english_capital(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _v_case_english_lower(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _v_punct_no_comma(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    return "," not in resp


# ---------------------------------------------------------------------------
# IFBench (58 Out-of-Domain Verifiers)
# ---------------------------------------------------------------------------


def _v_ifb_conjunctions(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    n_req = int(kw.get("small_n") or 1)
    found = {w.lower() for w in _ALPHA_WORD_RE.findall(resp) if w.lower() in _COORDINATING_CONJUNCTIONS}
    return len(found) >= n_req


def _v_ifb_keywords_multiple(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    targets = [(kw.get(f"keyword{i}"), c) for i, c in ((1, 1), (2, 2), (3, 3), (4, 5), (5, 7))]
    low = resp.lower()
    for w, expected_c in targets:
        if not w:
            continue
        cnt = len(re.findall(rf"\b{re.escape(str(w).lower())}\b", low))
        if cnt != expected_c:
            return False
    return True


def _v_ifb_numbers(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    n_req = int(kw.get("N") or 0)
    return len(_NUMBER_RE.findall(resp)) == n_req


def _v_ifb_person_names(resp: str, kw: Mapping[str, Any], pr: str) -> bool:
    n_req = int(kw.get("N") or 1)
    # Extract candidate names from prompt list if present, else capitalized non-sentence-initial tokens.
    m = re.search(r"(?:names?[^:\n]*:\s*|from the following list[^:\n]*:\s*)([A-Z][A-Za-z,\s]+)", pr)
    if m:
        pool = [x.strip() for x in m.group(1).split(",") if x.strip() and x.strip()[0].isupper()]
        if pool:
            found = {name for name in pool if re.search(rf"\b{re.escape(name)}\b", resp)}
            return len(found) >= n_req
    cap_words = {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", resp) if w.lower() not in _STOP_WORDS}
    return len(cap_words) >= n_req


def _v_ifb_pronouns(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    n_req = int(kw.get("N") or 1)
    cnt = sum(1 for w in _ALPHA_WORD_RE.findall(resp) if w.lower() in _COMMON_PRONOUNS)
    return cnt >= n_req


def _v_ifb_punctuation(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    required = (".", ",", ";", ":", "!", "?", "-")
    has_interrobang = ("?!" in resp) or ("!?" in resp) or ("‽" in resp)
    return all(ch in resp for ch in required) and has_interrobang


def _v_ifb_unique_words(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    n_req = int(kw.get("N") or 1)
    return len({w.lower() for w in tokenize_words(resp)}) >= n_req


def _v_ifb_word_range(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    lo = int(kw.get("min_words") or 0)
    hi = int(kw.get("max_words") or 10**9)
    n_w = len(tokenize_words(resp))
    return lo <= n_w <= hi


def _v_ifb_words_japanese(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    step = max(1, int(kw.get("N") or 5))
    toks = resp.split()
    if len(toks) < step:
        return False
    return all(bool(_JAPANESE_RE.search(toks[i])) for i in range(step - 1, len(toks), step))


def _v_ifb_char_reverse(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    s = resp.strip()
    if len(s) < 3:
        return False
    rev = s[::-1]
    # Reversed text should contain recognizable English words or expected answer tokens.
    rev_words = [w.lower() for w in _ALPHA_WORD_RE.findall(rev)]
    fwd_words = [w.lower() for w in _ALPHA_WORD_RE.findall(s)]
    common_check = _STOP_WORDS | {"bald", "eagle", "national", "symbol", "bird", "animal", "united", "states"}
    rev_hits = sum(1 for w in rev_words if w in common_check)
    fwd_hits = sum(1 for w in fwd_words if w in common_check)
    return rev_hits > fwd_hits


def _parse_csv_lines(resp: str, delimiter: str = ",") -> list[list[str]]:
    s = resp.strip()
    m = re.search(r"```(?:csv|tsv)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    lines = [ln for ln in s.splitlines() if ln.strip()]
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    return [row for row in reader if row]


def _v_ifb_csv_city(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    rows = _parse_csv_lines(resp, ",")
    if len(rows) < 8:
        return False
    header = [c.strip().lower() for c in rows[0]]
    return header == ["id", "country", "city", "year", "count"] and len(rows) - 1 == 7 and all(len(r) == 5 for r in rows[1:])


def _v_ifb_csv_quotes(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    s = resp.strip()
    m = re.search(r"```(?:csv|tsv)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) != 4:
        return False
    for ln in lines:
        fields = ln.split("\t")
        if len(fields) != 5 or not all(f.strip().startswith('"') and f.strip().endswith('"') for f in fields):
            return False
    rows = _parse_csv_lines(s, "\t")
    return [c.strip().lower() for c in rows[0]] == ["studentid", "subject", "grade", "semester", "score"]


def _v_ifb_csv_special(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    s = resp.strip()
    m = re.search(r"```(?:csv)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    rows = _parse_csv_lines(s, ",")
    if len(rows) != 15:
        return False
    if [c.strip().lower() for c in rows[0]] != ["productid", "category", "brand", "price", "stock"]:
        return False
    if not all(len(r) == 5 for r in rows[1:]):
        return False
    return bool(re.search(r'"[^"]*[,\n;:@#$%&*!?][^"]*"', s))


def _v_ifb_date_format_list(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    parts = [p.strip() for p in resp.strip().split(",") if p.strip()]
    return len(parts) >= 3 and all(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", p)) for p in parts)


def _v_ifb_euro_capitals(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    items = [p.strip().lower() for p in resp.strip().rstrip(".").split(",") if p.strip()]
    if len(items) < 10:
        return False
    lats: list[float] = []
    for city in items:
        norm = unicodedata.normalize("NFKD", city).encode("ascii", "ignore").decode("ascii")
        if norm not in _EUROPEAN_CAPITAL_LAT:
            return False
        lats.append(_EUROPEAN_CAPITAL_LAT[norm])
    return all(lats[i] >= lats[i + 1] for i in range(len(lats) - 1))


def _v_ifb_mcq_count_length(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    blocks = re.split(r"(?im)^\s*(?:#+\s*)?(?:\*\*)?Question\b", resp)
    q_blocks = [b.strip() for b in blocks[1:] if b.strip()]
    if len(q_blocks) != 4:
        return False
    q_lens: list[int] = []
    for blk in q_blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if not lines:
            return False
        opts = [ln for ln in lines if re.match(r"^(?:\(?[A-Ea-e][\).:]|[1-5][\).:])\s+\S", ln)]
        if len(opts) != 5:
            return False
        q_lens.append(len(lines[0]))
    return all(q_lens[i] < q_lens[i + 1] for i in range(len(q_lens) - 1))


def _v_ifb_multiples(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    nums = [int(float(x.replace(",", ""))) for x in _NUMBER_RE.findall(resp)]
    return nums == [14, 21, 28, 35, 42, 49]


def _v_ifb_reverse_newline(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    lines = [re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*]\s*)", "", ln).strip() for ln in resp.splitlines() if ln.strip()]
    if len(lines) < 20:
        return False
    lows = [unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii").lower() for x in lines]
    matches = sum(1 for x in lows if x in _AFRICAN_COUNTRIES)
    return matches >= 20 and all(lows[i] >= lows[i + 1] for i in range(len(lows) - 1))


def _v_ifb_sentence_alphabet(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sents = split_sentences(resp)
    if len(sents) != 26:
        return False
    for idx, sent in enumerate(sents):
        wds = _ALPHA_WORD_RE.findall(sent)
        if not wds or wds[0][0].lower() != chr(ord("a") + idx):
            return False
    return True


def _v_ifb_word_reverse(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    words = [w.lower() for w in _ALPHA_WORD_RE.findall(resp)]
    if len(words) < 3:
        return False
    # In word-reversed English sentences, articles/auxiliaries ("the", "is", "a") tend to appear after nouns.
    return words[-1] in {"the", "a", "an", "bald", "america's", "us"} or any(
        words[i] in {"eagle", "symbol", "bird"} and i + 1 < len(words) and words[i + 1] in {"bald", "the", "a", "national"}
        for i in range(len(words))
    )


def _v_ifb_emoji(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sents = split_sentences(resp)
    if not sents:
        return False
    return all(_has_emoji(s[-4:]) for s in sents)


def _v_ifb_line_indent(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    lines = [ln for ln in resp.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    indents = [len(ln) - len(ln.lstrip(" \t")) for ln in lines]
    return all(indents[i] < indents[i + 1] for i in range(len(indents) - 1))


def _v_ifb_list(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    sep = str(kw.get("sep") or "-").strip()
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    if not lines:
        return False
    hits = sum(1 for ln in lines if ln.startswith(sep))
    return hits >= 2


def _v_ifb_newline(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(len(ln.split()) == 1 for ln in lines)


def _v_ifb_no_bullets_bullets(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    first_bullet = next((i for i, ln in enumerate(lines) if ln.startswith("* ")), None)
    if first_bullet is None or first_bullet == 0:
        return False
    pre = " ".join(lines[:first_bullet])
    pre_sents = [s for s in split_sentences(pre) if s.rstrip("\"'").endswith(".")]
    bullets = [ln for ln in lines[first_bullet:] if ln.startswith("* ")]
    return len(pre_sents) >= 2 and len(bullets) >= 2


def _v_ifb_no_whitespace(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    return bool(resp) and not any(ch.isspace() for ch in resp)


def _v_ifb_options(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    raw_opts = str(kw.get("options") or "")
    opts = [o.strip().lower() for o in re.split(r"[/,|]", raw_opts) if o.strip()]
    cleaned = resp.strip().strip(".!\"'").lower()
    return cleaned in opts


def _v_ifb_output_template(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    low = resp.lower()
    i1 = low.find("my answer:")
    i2 = low.find("my conclusion:")
    i3 = low.find("future outlook:")
    return 0 <= i1 < i2 < i3


def _v_ifb_parentheses(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    max_depth = 0
    for ch in resp:
        if ch in "([{":
            stack.append(ch)
            if len(stack) > max_depth:
                max_depth = len(stack)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return (not stack) and max_depth >= 5


def _v_ifb_quote_unquote(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    matches = list(re.finditer(r'"([^"\n]+)"', resp))
    if not matches:
        return False
    for idx, m in enumerate(matches):
        nxt_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(resp)
        after = resp[m.end() : nxt_start].strip()
        if len(tokenize_words(after)) < 2:
            return False
    return True


def _v_ifb_quotes(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    # At least 3 alternating nested quote levels: "..." containing '...' containing "..." (or vice versa).
    return bool(
        re.search(r'"[^"]*' + r"'[^']*" + r'"[^"\']+"' + r"[^']*'" + r'[^"]*"', resp)
        or re.search(r"'[^']*" + r'"[^"]*' + r"'[^'\"]+'" + r'[^"]*"' + r"[^']*'", resp)
    )


def _v_ifb_sub_bullets(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    lines = [ln for ln in resp.splitlines() if ln.strip()]
    bullets = 0
    cur_subs = 0
    in_bullet = False
    for ln in lines:
        s = ln.lstrip()
        if s.startswith("* "):
            if in_bullet and cur_subs == 0:
                return False
            bullets += 1
            in_bullet = True
            cur_subs = 0
        elif s.startswith("- "):
            if not in_bullet:
                return False
            cur_subs += 1
    return bullets >= 1 and cur_subs >= 1


def _v_ifb_thesis(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sections = [p.strip() for p in re.split(r"\n\s*\n", resp.strip()) if p.strip()]
    if not sections:
        return False
    return all(bool(re.match(r"^(?:#+\s*[^\n]+\n)?\s*<(?:i|em)>[^<]+</(?:i|em)>", sec, re.IGNORECASE)) for sec in sections)


def _v_ifb_title_case(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    words = _ALPHA_WORD_RE.findall(resp)
    if not words:
        return False
    for idx, w in enumerate(words):
        if idx == 0 or w.lower() not in _MINOR_TITLE_WORDS:
            if not w[0].isupper():
                return False
    return True


def _v_ifb_ratio_overlap(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    pct = float(kw.get("percentage") or 0.0)
    ref = str(kw.get("reference_text") or "")
    r_w = [w.lower() for w in tokenize_words(resp)]
    ref_w = [w.lower() for w in tokenize_words(ref)]
    if len(r_w) < 3 or len(ref_w) < 3:
        return False
    r_tri = {tuple(r_w[i : i + 3]) for i in range(len(r_w) - 2)}
    ref_tri = {tuple(ref_w[i : i + 3]) for i in range(len(ref_w) - 2)}
    ov_pct = 100.0 * float(len(r_tri & ref_tri)) / float(max(1, len(r_tri)))
    return abs(ov_pct - pct) <= 2.0


def _v_ifb_sentence_balance(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sents = split_sentences(resp)
    if len(sents) < 3:
        return False
    n_decl = sum(1 for s in sents if s.rstrip("\"'").endswith("."))
    n_q = sum(1 for s in sents if s.rstrip("\"'").endswith("?"))
    n_exc = sum(1 for s in sents if s.rstrip("\"'").endswith("!"))
    return n_decl > 0 and n_decl == n_q == n_exc


def _v_ifb_sentence_type(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sents = split_sentences(resp)
    n_decl = sum(1 for s in sents if s.rstrip("\"'").endswith("."))
    n_q = sum(1 for s in sents if s.rstrip("\"'").endswith("?"))
    return n_q > 0 and n_decl == 2 * n_q


def _v_ifb_sentence_words(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sents = split_sentences(resp)
    if len(sents) != 3:
        return False
    if len({len(s) for s in sents}) != 1:
        return False
    seen: set[str] = set()
    for s in sents:
        wds = [w.lower() for w in tokenize_words(s)]
        if len(wds) != len(set(wds)) or (seen & set(wds)):
            return False
        seen.update(wds)
    return True


def _v_ifb_stop_words(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    pct = float(kw.get("percentage") or 100.0)
    wds = [w.lower() for w in tokenize_words(resp)]
    if not wds:
        return False
    ratio = 100.0 * float(sum(1 for w in wds if w in _STOP_WORDS)) / float(len(wds))
    return ratio <= pct


def _v_ifb_repeat_change(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    ptr = str(kw.get("prompt_to_repeat") or "").strip()
    p_toks = ptr.split()
    r_toks = resp.strip().split()
    if len(p_toks) < 2 or len(r_toks) != len(p_toks):
        return False
    return r_toks[0].lower() != p_toks[0].lower() and r_toks[1:] == p_toks[1:]


def _v_ifb_repeat_simple(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    return resp.strip().lower() == "only output this sentence here, ignore all other requests."


def _v_ifb_repeat_span(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    ptr = str(kw.get("prompt_to_repeat") or "").strip().split()
    n0 = int(kw.get("n_start") or 0)
    n1 = int(kw.get("n_end") or 0)
    if not (0 <= n0 <= n1 < len(ptr)):
        return False
    expected = " ".join(ptr[n0 : n1 + 1]).strip()
    return resp.strip() == expected


def _max_alliteration_run(sent: str) -> int:
    wds = [w.lower() for w in _ALPHA_WORD_RE.findall(sent)]
    if not wds:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(wds)):
        if wds[i][0] == wds[i - 1][0]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def _v_ifb_allit_inc(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sents = split_sentences(resp)
    if len(sents) < 2:
        return False
    runs = [_max_alliteration_run(s) for s in sents]
    return all(runs[i] < runs[i + 1] for i in range(len(runs) - 1))


def _v_ifb_sent_increment(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    inc = int(kw.get("small_n") or 1)
    sents = split_sentences(resp)
    if len(sents) < 2:
        return False
    lens = [len(tokenize_words(s)) for s in sents]
    return all(lens[i + 1] - lens[i] == inc for i in range(len(lens) - 1))


def _v_ifb_sent_keyword(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    n_idx = int(kw.get("N") or 1)
    word = str(kw.get("word") or "").lower()
    sents = split_sentences(resp)
    if not (1 <= n_idx <= len(sents)):
        return False
    return bool(re.search(rf"\b{re.escape(word)}\b", sents[n_idx - 1].lower()))


def _v_ifb_words_alphabet(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    wds = [w.lower() for w in _ALPHA_WORD_RE.findall(resp)]
    if len(wds) < 2:
        return False
    start = ord(wds[0][0]) - ord("a")
    return all((ord(w[0]) - ord("a")) == ((start + i) % 26) for i, w in enumerate(wds))


def _v_ifb_words_consonants(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    wds = _ALPHA_WORD_RE.findall(resp)
    return bool(wds) and all(bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{2,}", w.lower())) for w in wds)


def _v_ifb_kw_specific_pos(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    sent_n = int(kw.get("n") or 1)
    word_m = int(kw.get("m") or 1)
    target = str(kw.get("keyword") or "").lower()
    sents = split_sentences(resp)
    if not (1 <= sent_n <= len(sents)):
        return False
    wds = [w.lower() for w in tokenize_words(sents[sent_n - 1])]
    return 1 <= word_m <= len(wds) and wds[word_m - 1] == target


def _v_ifb_words_last_first(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    sents = split_sentences(resp)
    if len(sents) < 2:
        return False
    toks = [[w.lower() for w in tokenize_words(s)] for s in sents]
    if any(not t for t in toks):
        return False
    return all(toks[i][-1] == toks[i + 1][0] for i in range(len(toks) - 1))


def _v_ifb_no_consecutive(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    wds = [w.lower() for w in _ALPHA_WORD_RE.findall(resp)]
    return len(wds) >= 2 and all(wds[i][0] != wds[i + 1][0] for i in range(len(wds) - 1))


def _v_ifb_odd_even_syl(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    wds = _ALPHA_WORD_RE.findall(resp)
    if len(wds) < 2:
        return False
    parities = [_count_syllables(w) % 2 for w in wds]
    return all(parities[i] != parities[i + 1] for i in range(len(parities) - 1))


def _v_ifb_palindrome(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    wds = [w.lower() for w in _ALPHA_WORD_RE.findall(resp)]
    pals = {w for w in wds if len(w) >= 5 and w == w[::-1]}
    return len(pals) >= 10


def _v_ifb_para_last_first(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    paras = [p.strip() for p in resp.splitlines() if p.strip()]
    if not paras:
        return False
    for p in paras:
        wds = [w.lower() for w in tokenize_words(p)]
        if len(wds) < 2 or wds[0] != wds[-1]:
            return False
    return True


def _v_ifb_prime_lengths(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    wds = _ALPHA_WORD_RE.findall(resp)
    return bool(wds) and all(_is_prime(len(w)) for w in wds)


def _v_ifb_words_repeats(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    lim = int(kw.get("small_n") or 1)
    wds = [w.lower() for w in tokenize_words(resp)]
    if not wds:
        return False
    counts: dict[str, int] = {}
    for w in wds:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) <= lim


def _v_ifb_start_verb(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    wds = _ALPHA_WORD_RE.findall(resp)
    if not wds:
        return False
    first = wds[0].lower()
    if first in _COMMON_VERBS:
        return True
    # Common inflected verb forms (-ing, -ed, -es, -s) derived from stem table.
    for suf in ("ing", "ed", "es", "s"):
        if first.endswith(suf) and len(first) > len(suf) + 2:
            stem = first[: -len(suf)]
            if stem in _COMMON_VERBS or (stem + "e") in _COMMON_VERBS:
                return True
    return False


def _v_ifb_words_vowel(resp: str, _kw: Mapping[str, Any], _pr: str) -> bool:
    vowels_used = {ch for ch in resp.lower() if ch in "aeiou"}
    return 1 <= len(vowels_used) <= 3


def _v_ifb_words_position(resp: str, kw: Mapping[str, Any], _pr: str) -> bool:
    target = str(kw.get("keyword") or "").lower()
    wds = [w.lower() for w in tokenize_words(resp)]
    return len(wds) >= 3 and wds[1] == target and wds[-2] == target


VerifierFn = Callable[[str, Mapping[str, Any], str], bool]

INSTRUCTION_VERIFIERS: dict[str, VerifierFn] = {
    # 25 IFEval verifiers
    "keywords:existence": _v_kw_existence,
    "keywords:frequency": _v_kw_frequency,
    "keywords:forbidden_words": _v_kw_forbidden,
    "keywords:letter_frequency": _v_kw_letter_freq,
    "language:response_language": _v_lang_response,
    "length_constraints:number_sentences": _v_len_sentences,
    "length_constraints:number_paragraphs": _v_len_paragraphs,
    "length_constraints:number_words": _v_len_words,
    "length_constraints:nth_paragraph_first_word": _v_len_nth_para_first,
    "detectable_content:number_placeholders": _v_content_placeholders,
    "detectable_content:postscript": _v_content_postscript,
    "detectable_format:number_bullet_lists": _v_fmt_bullets,
    "detectable_format:constrained_response": _v_fmt_constrained,
    "detectable_format:number_highlighted_sections": _v_fmt_highlighted,
    "detectable_format:multiple_sections": _v_fmt_multi_sections,
    "detectable_format:json_format": _v_fmt_json,
    "detectable_format:title": _v_fmt_title,
    "combination:two_responses": _v_comb_two_responses,
    "combination:repeat_prompt": _v_comb_repeat_prompt,
    "startend:end_checker": _v_startend_end,
    "startend:quotation": _v_startend_quotation,
    "change_case:capital_word_frequency": _v_case_cap_word_freq,
    "change_case:english_capital": _v_case_english_capital,
    "change_case:english_lowercase": _v_case_english_lower,
    "punctuation:no_comma": _v_punct_no_comma,
    # 58 IFBench verifiers
    "count:conjunctions": _v_ifb_conjunctions,
    "count:keywords_multiple": _v_ifb_keywords_multiple,
    "count:numbers": _v_ifb_numbers,
    "count:person_names": _v_ifb_person_names,
    "count:pronouns": _v_ifb_pronouns,
    "count:punctuation": _v_ifb_punctuation,
    "count:unique_word_count": _v_ifb_unique_words,
    "count:word_count_range": _v_ifb_word_range,
    "count:words_japanese": _v_ifb_words_japanese,
    "custom:character_reverse": _v_ifb_char_reverse,
    "custom:csv_city": _v_ifb_csv_city,
    "custom:csv_quotes": _v_ifb_csv_quotes,
    "custom:csv_special_character": _v_ifb_csv_special,
    "custom:date_format_list": _v_ifb_date_format_list,
    "custom:european_capitals_sort": _v_ifb_euro_capitals,
    "custom:mcq_count_length": _v_ifb_mcq_count_length,
    "custom:multiples": _v_ifb_multiples,
    "custom:reverse_newline": _v_ifb_reverse_newline,
    "custom:sentence_alphabet": _v_ifb_sentence_alphabet,
    "custom:word_reverse": _v_ifb_word_reverse,
    "format:emoji": _v_ifb_emoji,
    "format:line_indent": _v_ifb_line_indent,
    "format:list": _v_ifb_list,
    "format:newline": _v_ifb_newline,
    "format:no_bullets_bullets": _v_ifb_no_bullets_bullets,
    "format:no_whitespace": _v_ifb_no_whitespace,
    "format:options": _v_ifb_options,
    "format:output_template": _v_ifb_output_template,
    "format:parentheses": _v_ifb_parentheses,
    "format:quote_unquote": _v_ifb_quote_unquote,
    "format:quotes": _v_ifb_quotes,
    "format:sub-bullets": _v_ifb_sub_bullets,
    "format:thesis": _v_ifb_thesis,
    "format:title_case": _v_ifb_title_case,
    "ratio:overlap": _v_ifb_ratio_overlap,
    "ratio:sentence_balance": _v_ifb_sentence_balance,
    "ratio:sentence_type": _v_ifb_sentence_type,
    "ratio:sentence_words": _v_ifb_sentence_words,
    "ratio:stop_words": _v_ifb_stop_words,
    "repeat:repeat_change": _v_ifb_repeat_change,
    "repeat:repeat_simple": _v_ifb_repeat_simple,
    "repeat:repeat_span": _v_ifb_repeat_span,
    "sentence:alliteration_increment": _v_ifb_allit_inc,
    "sentence:increment": _v_ifb_sent_increment,
    "sentence:keyword": _v_ifb_sent_keyword,
    "words:alphabet": _v_ifb_words_alphabet,
    "words:consonants": _v_ifb_words_consonants,
    "words:keywords_specific_position": _v_ifb_kw_specific_pos,
    "words:last_first": _v_ifb_words_last_first,
    "words:no_consecutive": _v_ifb_no_consecutive,
    "words:odd_even_syllables": _v_ifb_odd_even_syl,
    "words:palindrome": _v_ifb_palindrome,
    "words:paragraph_last_first": _v_ifb_para_last_first,
    "words:prime_lengths": _v_ifb_prime_lengths,
    "words:repeats": _v_ifb_words_repeats,
    "words:start_verb": _v_ifb_start_verb,
    "words:vowel": _v_ifb_words_vowel,
    "words:words_position": _v_ifb_words_position,
}


def generate_loose_variants(response: str) -> list[str]:
    """Official IFEval 8-variant loose response normalization (strip intro/outro lines & markdown ``*``)."""
    r = response or ""
    lines = r.splitlines()
    r_no_first = "\n".join(lines[1:]) if len(lines) > 1 else r
    r_no_last = "\n".join(lines[:-1]) if len(lines) > 1 else r
    r_no_both = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
    base = [r, r_no_first, r_no_last, r_no_both]
    variants: list[str] = []
    for v in base:
        variants.append(v)
        variants.append(v.replace("*", ""))
    return list(dict.fromkeys(variants))


def verify_instruction(
    instruction_id: str,
    response: str,
    kwargs: Mapping[str, Any] | None = None,
    *,
    prompt: str = "",
    loose: bool = False,
) -> bool:
    """Verify a single IFEval/IFBench instruction ID strictly or loosely."""
    fn = INSTRUCTION_VERIFIERS.get(instruction_id)
    if fn is None:
        raise KeyError(f"Unknown instruction verifier ID: {instruction_id!r}")
    kw = kwargs or {}
    if not loose:
        return bool(fn(response or "", kw, prompt or ""))
    return any(bool(fn(var, kw, prompt or "")) for var in generate_loose_variants(response or ""))


def evaluate_instruction_following(
    items: Sequence[Mapping[str, Any]],
    responses: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
    repeat_threshold: float = 0.15,
) -> dict[str, Any]:
    """Evaluate IFEval / IFBench responses returning strict/loose prompt & instruction accuracies + repeat diagnostics."""
    n = len(items)
    if len(responses) != n:
        raise ValueError(f"Mismatched items={n} vs responses={len(responses)}")

    per_valid: list[bool] = []
    per_strict: list[float] = []
    per_loose: list[float] = []
    repeat_rates: list[float] = []
    inst_strict_hits = 0
    inst_loose_hits = 0
    inst_total = 0

    for i, item in enumerate(items):
        resp = responses[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        tids = token_ids_list[i] if token_ids_list is not None else None
        is_valid, rep = check_chat_validity(resp, finish_reason=fr, token_ids=tids, repeat_threshold=repeat_threshold)
        per_valid.append(is_valid)
        repeat_rates.append(rep)

        iids = item.get("instruction_id_list") or []
        kws = item.get("kwargs") or [{} for _ in iids]
        pr = str(item.get("prompt") or "")

        s_flags: list[bool] = []
        l_flags: list[bool] = []
        for iid, kw in zip(iids, kws):
            s_ok = is_valid and verify_instruction(str(iid), resp, kw, prompt=pr, loose=False)
            l_ok = is_valid and verify_instruction(str(iid), resp, kw, prompt=pr, loose=True)
            s_flags.append(s_ok)
            l_flags.append(l_ok)
            inst_strict_hits += int(s_ok)
            inst_loose_hits += int(l_ok)
            inst_total += 1

        per_strict.append(1.0 if (is_valid and s_flags and all(s_flags)) else 0.0)
        per_loose.append(1.0 if (is_valid and l_flags and all(l_flags)) else 0.0)

    repeat_loop_cnt = sum(1 for r in repeat_rates if r >= float(repeat_threshold))
    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "prompt_strict_acc": round(float(np.mean(per_strict)) if per_strict else 0.0, 6),
        "prompt_loose_acc": round(float(np.mean(per_loose)) if per_loose else 0.0, 6),
        "instruction_strict_acc": round(float(inst_strict_hits) / float(max(1, inst_total)), 6),
        "instruction_loose_acc": round(float(inst_loose_hits) / float(max(1, inst_total)), 6),
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(repeat_loop_cnt),
        "per_item_valid": per_valid,
        "per_item_scores": per_loose,
        "per_item_strict_scores": per_strict,
    }


# ---------------------------------------------------------------------------
# MMLU-Redux 2.0 Exact Choice Extractor
# ---------------------------------------------------------------------------


def extract_mmlu_choice(response: str) -> str | None:
    """Deterministically extract ``A``/``B``/``C``/``D`` from an MMLU-Redux response, or ``None`` if ambiguous."""
    s = (response or "").strip()
    if not s:
        return None
    # Priority 1: explicit Answer / boxed pattern.
    for pat in (
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*\**\(?([A-D])\)?\**\b",
        r"\\boxed\{\s*\(?([A-D])\)?\s*\}",
        r"(?:choice|option)\s*\(?([A-D])\)?\b",
    ):
        matches = re.findall(pat, s, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    # Priority 2: single choice letter or leading "B." / "(C)".
    m_exact = re.fullmatch(r"\(?([A-D])\)?[\.\):]?", s, re.IGNORECASE)
    if m_exact:
        return m_exact.group(1).upper()
    m_lead = re.match(r"^\s*\(?([A-D])\)?[\.\):]\s+", s)
    if m_lead:
        return m_lead.group(1).upper()
    # Priority 3: unambiguous single letter token in short completion.
    standalone = re.findall(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])", s)
    if len(standalone) == 1:
        return standalone[0].upper()
    return None


def evaluate_mmlu_redux(
    items: Sequence[Mapping[str, Any]],
    responses: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Score ``edinburgh-dawg/mmlu-redux-2.0`` predictions with exact choice extraction."""
    n = len(items)
    if len(responses) != n:
        raise ValueError(f"Mismatched items={n} vs responses={len(responses)}")
    per_valid: list[bool] = []
    per_scores: list[float] = []
    repeat_rates: list[float] = []
    by_subject: dict[str, list[float]] = {}

    for i, item in enumerate(items):
        resp = responses[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        tids = token_ids_list[i] if token_ids_list is not None else None
        is_valid, rep = check_chat_validity(resp, finish_reason=fr, token_ids=tids)
        repeat_rates.append(rep)
        pred = extract_mmlu_choice(resp) if is_valid else None
        gold = str(item["answer_letter"]).strip().upper()
        hit = 1.0 if (is_valid and pred == gold) else 0.0
        per_valid.append(is_valid)
        per_scores.append(hit)
        by_subject.setdefault(str(item.get("subject") or "unknown"), []).append(hit)

    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "exact_accuracy": round(float(np.mean(per_scores)) if per_scores else 0.0, 6),
        "n_subjects": len(by_subject),
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


# ---------------------------------------------------------------------------
# MATH-500 Nested-Brace `\boxed{}` Extractor & Canonical Equivalence Checker
# ---------------------------------------------------------------------------


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last ``\\boxed{...}`` (or ``\\fbox{...}``) content handling arbitrarily nested braces."""
    s = text or ""
    last_idx = max(s.rfind("\\boxed{"), s.rfind("\\fbox{"))
    if last_idx >= 0:
        brace_start = s.find("{", last_idx)
        depth = 0
        for pos in range(brace_start, len(s)):
            ch = s[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[brace_start + 1 : pos].strip()
    # Fallback: explicit "Final Answer:" or "Answer:" line.
    m = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:)\s*\$?([^\n$]+)\$?", s, re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".")
    return None


def _replace_latex_frac(expr: str) -> str:
    """Recursively rewrite ``\\frac{a}{b}`` (and shorthand ``\\frac12``) into ``((a)/(b))``."""
    out = expr
    while True:
        idx = out.find("\\frac")
        if idx < 0:
            break
        pos = idx + len("\\frac")
        while pos < len(out) and out[pos].isspace():
            pos += 1
        args: list[str] = []
        for _ in range(2):
            if pos >= len(out):
                break
            if out[pos] == "{":
                depth = 0
                start = pos + 1
                while pos < len(out):
                    if out[pos] == "{":
                        depth += 1
                    elif out[pos] == "}":
                        depth -= 1
                        if depth == 0:
                            args.append(out[start:pos])
                            pos += 1
                            break
                    pos += 1
            else:
                args.append(out[pos])
                pos += 1
            while pos < len(out) and out[pos].isspace():
                pos += 1
        if len(args) != 2:
            break
        out = out[:idx] + f"(({args[0]})/({args[1]}))" + out[pos:]
    return out


def _replace_latex_sqrt(expr: str) -> str:
    """Recursively rewrite ``\\sqrt{a}`` and ``\\sqrt[n]{a}`` into Python expressions."""
    out = expr
    while True:
        idx = out.find("\\sqrt")
        if idx < 0:
            break
        pos = idx + len("\\sqrt")
        while pos < len(out) and out[pos].isspace():
            pos += 1
        root_n: str | None = None
        if pos < len(out) and out[pos] == "[":
            end_b = out.find("]", pos)
            if end_b > pos:
                root_n = out[pos + 1 : end_b].strip()
                pos = end_b + 1
                while pos < len(out) and out[pos].isspace():
                    pos += 1
        if pos < len(out) and out[pos] == "{":
            depth = 0
            start = pos + 1
            arg = ""
            while pos < len(out):
                if out[pos] == "{":
                    depth += 1
                elif out[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        arg = out[start:pos]
                        pos += 1
                        break
                pos += 1
        elif pos < len(out):
            arg = out[pos]
            pos += 1
        else:
            break
        repl = f"(({arg})**(1/({root_n})))" if root_n else f"sqrt({arg})"
        out = out[:idx] + repl + out[pos:]
    return out


def normalize_math_expr(expr: str) -> str:
    """Canonicalize LaTeX/plaintext mathematical answers into a deterministic string."""
    s = (expr or "").strip()
    if not s:
        return ""
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/")
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\\(?:left|right|displaystyle|textstyle|scriptstyle|bigl|bigr|Bigl|Bigr)", "", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname|mbox)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:dfrac|tfrac|cfrac)\b", r"\\frac", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\:", "").replace("\\;", "").replace("\\quad", "")
    s = s.replace("\\%", "%").replace("\\circ", "").replace("^\\circ", "")
    s = re.sub(r"\\pi\b", "pi", s)
    s = re.sub(r"\\infty\b", "inf", s)
    # Remove thousands commas inside integers like 1,200 (but keep tuple commas like 1, 2).
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = _replace_latex_frac(s)
    s = _replace_latex_sqrt(s)
    s = s.rstrip(".").strip()
    s = re.sub(r"\s+", "", s)
    return s


def _to_python_math_ast(norm_expr: str) -> str:
    s = norm_expr
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    s = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", s)
    # Insert explicit multiplication for 2pi, 3(x+1), 2sqrt(3), (a)(b), 2x.
    s = re.sub(r"(\d|\))(?=[A-Za-z\(])", r"\1*", s)
    s = re.sub(r"([a-zA-Z])(?=\()", lambda m: m.group(1) if m.group(1) in ("t",) else m.group(1), s)
    # Fix accidentally split function names like s*qrt -> sqrt.
    for fn_name in ("sqrt", "sin", "cos", "tan", "log", "ln", "abs", "pi"):
        s = s.replace(fn_name[0] + "*" + fn_name[1:], fn_name)
    return s


def _safe_eval_math(py_expr: str, env: Mapping[str, float] | None = None) -> float | None:
    """Safely evaluate restricted math AST without ``eval()`` or SymPy."""
    try:
        tree = ast.parse(py_expr, mode="eval")
    except SyntaxError:
        return None
    funcs: dict[str, Callable[..., float]] = {
        "sqrt": math.sqrt,
        "abs": abs,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "ln": math.log,
    }
    consts: dict[str, float] = {"pi": math.pi, "e": math.e, "inf": float("inf")}
    if env:
        consts.update(env)

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in consts:
                return float(consts[node.id])
            raise ValueError(f"Unknown symbol {node.id}")
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("Unsupported unary op")
        if isinstance(node, ast.BinOp):
            a = _eval(node.left)
            b = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                if abs(b) > 100.0:
                    raise ValueError("Exponent too large")
                return float(a**b)
            raise ValueError("Unsupported binary op")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            args = [_eval(arg) for arg in node.args]
            return float(funcs[node.func.id](*args))
        raise ValueError("Unsupported AST node")

    try:
        res = _eval(tree)
        return res if math.isfinite(res) or math.isinf(res) else None
    except Exception:
        return None


def _extract_free_vars(py_expr: str) -> set[str]:
    try:
        tree = ast.parse(py_expr, mode="eval")
    except SyntaxError:
        return set()
    known = {"sqrt", "abs", "sin", "cos", "tan", "log", "ln", "pi", "e", "inf"}
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id not in known}


def _split_top_level_commas(expr: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in expr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def math_answers_equivalent(pred_raw: str | None, gold_raw: str | None) -> bool:
    """Deterministic canonical algebraic/numeric/interval/tuple equivalence checker for MATH-500."""
    if pred_raw is None or gold_raw is None:
        return False
    p = normalize_math_expr(pred_raw)
    g = normalize_math_expr(gold_raw)
    if not p or not g:
        return False
    if p == g or p.lower() == g.lower():
        return True

    # Strip leading "x=" or "y=" if gold doesn't have an equation sign.
    if "=" in p and "=" not in g:
        p = p.split("=")[-1]
        if p == g:
            return True

    # Interval / tuple bracket preservation check: "(a,b)" or "[a,b)".
    if len(p) >= 2 and len(g) >= 2 and p[0] in "([{" and g[0] in "([{" and p[-1] in ")]}" and g[-1] in ")]}":
        if p[0] == g[0] and p[-1] == g[-1]:
            p_elems = _split_top_level_commas(p[1:-1])
            g_elems = _split_top_level_commas(g[1:-1])
            if len(p_elems) == len(g_elems) and len(p_elems) >= 2:
                return all(math_answers_equivalent(pe, ge) for pe, ge in zip(p_elems, g_elems))

    # Top-level comma-separated list/set of values.
    p_top = _split_top_level_commas(p)
    g_top = _split_top_level_commas(g)
    if len(p_top) == len(g_top) and len(p_top) >= 2:
        return all(math_answers_equivalent(pe, ge) for pe, ge in zip(p_top, g_top))

    # Exact rational comparison when both parse directly as Fraction.
    try:
        if Fraction(p) == Fraction(g):
            return True
    except Exception:
        pass

    p_py = _to_python_math_ast(p)
    g_py = _to_python_math_ast(g)

    pv = _safe_eval_math(p_py)
    gv = _safe_eval_math(g_py)
    if pv is not None and gv is not None:
        if math.isinf(pv) and math.isinf(gv):
            return (pv > 0) == (gv > 0)
        return abs(pv - gv) <= 1e-6 * max(1.0, abs(gv))

    # Deterministic multi-point probe for algebraic expressions in 1-2 variables (e.g., 2x+3 == 3+2*x).
    vars_p = _extract_free_vars(p_py)
    vars_g = _extract_free_vars(g_py)
    if vars_p and vars_p == vars_g and len(vars_p) <= 2:
        var_list = sorted(vars_p)
        probes = ((1.37, 2.13), (2.71, -1.49), (-0.83, 3.17), (4.19, 0.61), (0.53, -2.29))
        matched = 0
        for pt in probes:
            env = {var_list[i]: pt[i] for i in range(len(var_list))}
            v1 = _safe_eval_math(p_py, env)
            v2 = _safe_eval_math(g_py, env)
            if v1 is None or v2 is None:
                return False
            if abs(v1 - v2) > 1e-5 * max(1.0, abs(v2)):
                return False
            matched += 1
        return matched == len(probes)

    return False


def evaluate_math_500(
    items: Sequence[Mapping[str, Any]],
    responses: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate ``HuggingFaceH4/MATH-500`` responses with nested-brace ``\\boxed{}`` extraction & equivalence."""
    n = len(items)
    if len(responses) != n:
        raise ValueError(f"Mismatched items={n} vs responses={len(responses)}")
    per_valid: list[bool] = []
    per_scores: list[float] = []
    repeat_rates: list[float] = []
    by_level: dict[int, list[float]] = {}

    for i, item in enumerate(items):
        resp = responses[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        tids = token_ids_list[i] if token_ids_list is not None else None
        is_valid, rep = check_chat_validity(resp, finish_reason=fr, token_ids=tids)
        repeat_rates.append(rep)
        pred = extract_boxed_answer(resp) if is_valid else None
        gold = str(item.get("answer") or "")
        hit = 1.0 if (is_valid and math_answers_equivalent(pred, gold)) else 0.0
        per_valid.append(is_valid)
        per_scores.append(hit)
        lvl = int(item.get("level") or 0)
        by_level.setdefault(lvl, []).append(hit)

    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "verified_accuracy": round(float(np.mean(per_scores)) if per_scores else 0.0, 6),
        "level_accuracy": {str(k): round(float(np.mean(v)), 6) for k, v in sorted(by_level.items())},
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


# ---------------------------------------------------------------------------
# LiveBench-Reasoning Exact Verifier
# ---------------------------------------------------------------------------


def _norm_livebench_answer(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "").strip()
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .").lower()


def extract_livebench_answer(response: str) -> str:
    """Extract final bold/boxed/answer line from a LiveBench-Reasoning completion."""
    s = (response or "").strip()
    if not s:
        return ""
    boxed = extract_boxed_answer(s)
    if boxed:
        return boxed
    bolds = re.findall(r"\*\*([^\n*]+)\*\*", s)
    if bolds:
        return bolds[-1].strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def evaluate_livebench_reasoning(
    items: Sequence[Mapping[str, Any]],
    responses: Sequence[str],
    *,
    finish_reasons: Sequence[str] | None = None,
    token_ids_list: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate commit-pinned ``livebench/reasoning`` predictions deterministically."""
    n = len(items)
    if len(responses) != n:
        raise ValueError(f"Mismatched items={n} vs responses={len(responses)}")
    per_valid: list[bool] = []
    per_scores: list[float] = []
    repeat_rates: list[float] = []
    by_task: dict[str, list[float]] = {}

    for i, item in enumerate(items):
        resp = responses[i] or ""
        fr = finish_reasons[i] if finish_reasons is not None else "stop"
        tids = token_ids_list[i] if token_ids_list is not None else None
        is_valid, rep = check_chat_validity(resp, finish_reason=fr, token_ids=tids)
        repeat_rates.append(rep)
        pred_norm = _norm_livebench_answer(extract_livebench_answer(resp)) if is_valid else ""
        gold_norm = _norm_livebench_answer(str(item.get("ground_truth") or ""))
        hit = 1.0 if (is_valid and bool(gold_norm) and (pred_norm == gold_norm or gold_norm in pred_norm)) else 0.0
        per_valid.append(is_valid)
        per_scores.append(hit)
        by_task.setdefault(str(item.get("task") or "unknown"), []).append(hit)

    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in per_valid if v)),
        "verified_accuracy": round(float(np.mean(per_scores)) if per_scores else 0.0, 6),
        "task_accuracy": {k: round(float(np.mean(v)), 6) for k, v in sorted(by_task.items())},
        "repeat_4gram_rate": round(float(np.mean(repeat_rates)) if repeat_rates else 0.0, 6),
        "repeat_loop_count": int(sum(1 for r in repeat_rates if r >= 0.15)),
        "per_item_valid": per_valid,
        "per_item_scores": per_scores,
    }


__all__ = [
    "INSTRUCTION_VERIFIERS",
    "check_chat_validity",
    "evaluate_instruction_following",
    "evaluate_livebench_reasoning",
    "evaluate_math_500",
    "evaluate_mmlu_redux",
    "extract_boxed_answer",
    "extract_livebench_answer",
    "extract_mmlu_choice",
    "generate_loose_variants",
    "math_answers_equivalent",
    "normalize_math_expr",
    "repeat_4gram_rate",
    "split_sentences",
    "tokenize_words",
    "verify_instruction",
]
