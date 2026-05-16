"""
Comment cleaning stage.

Runs before any sentiment or topic processing. Takes raw comment records
(list[dict]) and returns the same records enriched with four new fields:

    cleaned_text  — text safe for NLP: URLs and @handles removed,
                    whitespace normalised, NFC unicode. Emojis kept
                    (RoBERTa and VADER both use them). Original 'text'
                    field is never modified.
    is_spam       — True if any rule below fires. Excluded from analytics
                    but kept in the store for audit purposes.
    language      — ISO 639-1 code from langdetect, or "und" when the
                    text is too short to detect reliably.
    is_duplicate  — True for every occurrence of a normalised text hash
                    after the first. Near-duplicates share the same hash
                    (case-folded, emoji-stripped, punctuation-stripped).

Why langdetect over fasttext:
    langdetect is pure-Python, pip-installable with no separate model
    download, and deterministic with DetectorFactory.seed = 0. fasttext
    is more accurate on short texts but requires fetching a 125 MB model
    file out-of-band, which breaks the current deployment story. We
    handle langdetect's short-text weakness by returning "und" below
    MIN_LANG_DETECT_LENGTH rather than making a noisy guess.
"""

import hashlib
import re
import unicodedata

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # deterministic results across runs

# ── Spam rule configuration ────────────────────────────────────────────────────
# Each entry is (regex_pattern, human_readable_reason).
# Pattern is matched case-insensitively against the ORIGINAL comment text.
# Add, remove, or adjust patterns here — the marketing team owns this list.

SPAM_PATTERNS: list[tuple[str, str]] = [
    # ── Self-promotion & external links ────────────────────────────────────
    (r"https?://\S+",                                   "contains URL (http/https)"),
    (r"www\.\S+",                                       "contains URL (www)"),
    (r"@[A-Za-z0-9_.]{3,}",                             "contains @handle"),
    (r"\bsubscribe\s+to\s+my\b",                        "subscribe-to-my plug"),
    (r"\bcheck\s+(out\s+)?my\s+(channel|page|profile)\b", "channel self-promo"),
    (r"\bfollow\s+me\b",                                "follow-me self-promo"),
    (r"\blink\s+in\s+(my\s+)?bio\b",                   "link-in-bio promo"),

    # ── Giveaway / crypto / financial scam ─────────────────────────────────
    (r"\b(free\s+)?(bitcoin|crypto|btc|eth|nft|usdt|solana)\b", "crypto mention"),
    (r"\bgiveaway\b",                                   "giveaway mention"),
    (r"\bdm\s+me\b",                                    "DM solicitation"),
    (r"\bwhatsapp\b",                                   "WhatsApp solicitation"),
    (r"\btelegram\b",                                   "Telegram solicitation"),
    (r"\bearn\s+\$",                                    "earn-money scam"),
    (r"\bmake\s+money\b",                               "make-money scam"),
    (r"\bcash\s*app\b",                                 "CashApp solicitation"),

    # ── Classic low-effort / bot patterns ───────────────────────────────────
    (r"^\s*first\s*[!.]*\s*$",                         "lone 'first' comment"),
    (r"\bwho'?s?\s+(here|watching)\s+in\s+2\d{3}\b",  "who's watching in <year>"),
    (r"\bstill\s+watching\s+in\s+2\d{3}\b",           "still watching in <year>"),
    (r"\blike\s+if\s+you'?r?e?\s+(still\s+)?watching\b", "like-if-watching bait"),
    (r"\b(early\s+)?gang\s*[!.]*$",                    "early gang comment"),
    (r"\bnotification\s+squad\b",                       "notification squad bot"),
]

# Minimum length of cleaned_text (chars) before we apply pattern matching.
# Comments shorter than this are spam regardless of pattern matches.
MIN_CLEANED_LENGTH: int = 4

# Minimum number of distinct alphabetic tokens (≥ 2 chars each).
# Catches lone-emoji comments, "lol", "nice", etc.
MIN_TOKEN_COUNT: int = 2

# Texts shorter than this (chars) get language = "und" (undetermined).
# langdetect is unreliable on very short strings.
MIN_LANG_DETECT_LENGTH: int = 20

# ── Internal compiled state ────────────────────────────────────────────────────

_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), reason)
    for pat, reason in SPAM_PATTERNS
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_.]{3,}")
_WHITESPACE_RE = re.compile(r"\s+")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_for_processing(text: str) -> str:
    """Remove URLs and @handles; NFC-normalise; collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_for_dedup(text: str) -> str:
    """
    Aggressive normalisation used only for duplicate hashing, not for NLP.
    Lowercases, removes emoji (by Unicode category), strips punctuation,
    collapses whitespace.
    """
    text = text.lower()
    # Strip 'Symbol, other' (So) and 'Symbol, modifier' (Sk) — covers most emoji
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Sk"))
    text = re.sub(r"[^\w\s]", "", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _check_spam(original: str, cleaned: str) -> bool:
    """Return True if this comment should be excluded from analytics."""
    # Ultra-low-information: too short after cleaning
    if len(cleaned) < MIN_CLEANED_LENGTH:
        return True

    # Ultra-low-information: fewer than MIN_TOKEN_COUNT alphabetic tokens
    alpha_tokens = re.findall(r"[a-zA-Z]{2,}", cleaned)
    if len(alpha_tokens) < MIN_TOKEN_COUNT:
        return True

    # Pattern rules — checked against original text so URLs etc. aren't
    # stripped before matching
    return any(pat.search(original) for pat, _ in _COMPILED_PATTERNS)


def _detect_language(text: str) -> str:
    if len(text) < MIN_LANG_DETECT_LENGTH:
        return "und"
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def _dedup_hash(normalized: str) -> str:
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()  # noqa: S324 — not security-sensitive


# ── Public API ─────────────────────────────────────────────────────────────────

def clean_comments(records: list[dict]) -> list[dict]:
    """
    Enrich raw comment records with cleaning metadata.

    Parameters
    ----------
    records:
        Raw comment dicts as returned by YouTubeClient.get_all_comments().
        Each dict must contain at least a 'text' key.

    Returns
    -------
    Same records (new dicts, originals not mutated) with four additional keys:
        cleaned_text  (str)   text ready for sentiment/topic models
        is_spam       (bool)  True → excluded from all downstream metrics
        language      (str)   ISO 639-1 or "und"
        is_duplicate  (bool)  True → all-but-first occurrence of same content
    """
    seen_hashes: set[str] = set()
    enriched: list[dict] = []

    for rec in records:
        rec = dict(rec)  # shallow copy — never mutate caller's data
        text = str(rec.get("text", ""))

        # 1. Clean for NLP downstream
        cleaned = _clean_for_processing(text)
        rec["cleaned_text"] = cleaned

        # 2. Spam gate
        rec["is_spam"] = _check_spam(text, cleaned)

        # 3. Language detection on cleaned text (URLs removed → fewer false "en")
        rec["language"] = _detect_language(cleaned)

        # 4. Near-duplicate detection (hash of aggressively normalised text)
        h = _dedup_hash(_normalize_for_dedup(cleaned))
        if h in seen_hashes:
            rec["is_duplicate"] = True
        else:
            seen_hashes.add(h)
            rec["is_duplicate"] = False

        enriched.append(rec)

    return enriched
