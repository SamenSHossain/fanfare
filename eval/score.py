#!/usr/bin/env python3
"""
Score a manually-labeled comment CSV against VADER and the transformer.

Usage
-----
    python eval/score.py eval/to_label.csv
    python eval/score.py eval/to_label.csv --no-transformer
    python eval/score.py eval/to_label.csv --batch-size 16

Expected input
--------------
The CSV produced by eval/sample.py, with 'human_label' filled in:
    human_label   Positive | Neutral | Negative
    text          original text
    cleaned_text  model-facing text (preferred if present)

Rows with blank or invalid human_label are silently skipped.

Output
------
  1. Accuracy + macro-F1 table: VADER vs transformer, all comments
  2. Certain-only sub-table (rows where transformer is not Uncertain)
  3. Sarcasm / inverted-slang slice (editable patterns below)
  4. Threshold calibration: suggested UNCERTAIN_THRESHOLD and score cutoffs
     — values are PRINTED ONLY, never auto-applied

Rerunning
---------
This script is stateless: run it any time the labeled set grows or after
editing SARCASM_PATTERNS.  Transformer probs are computed fresh each run;
the grid search takes only ms once inference is done.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.scoring import UNCERTAIN_THRESHOLD as _DEFAULT_UNC

# ══════════════════════════════════════════════════════════════════════════════
# EDITABLE: sarcasm / inverted-slang patterns
#
# Each entry is an OR-ed regex.  A comment matching ANY pattern is included
# in the sarcasm slice.  Edit freely — add fan-specific slang as you spot it.
# ══════════════════════════════════════════════════════════════════════════════
SARCASM_PATTERNS: list[str] = [
    # Explicit sarcasm markers
    r"\byeah right\b",
    r"\bsure jan\b",
    r"\boh\s+sure\b",
    r"\bsuuure\b",
    r"\bobviously\b",
    r"\btotally\b",                           # "totally earned that" — often ironic
    r"\bwow\s+wow\s+wow\b",

    # Inverted praise / mock hype
    r"\bsooo+\s+(good|great|amazing|elite)\b",          # exaggerated positive = possible sarcasm
    r"\bwaow\b",                                          # phonetic mock
    r"\boverrated+d*\b",
    r"\b(definitely|clearly)\s+(not)\b",

    # Negation before positive adjective (basic pattern)
    r"\bnot\s+\w+\s+(good|great|amazing|best|elite|special|incredible)\b",

    # Scare / air quotes
    r'(?<!\w)"[^"]{3,40}"',                  # "amazing" effort
    r"'[^']{3,30}'",                          # 'great' performance

    # Emoji markers common in ironic fan comments
    r"💀",                                    # "he's 💀" = dead / unbelievably bad or good
    r"🙄|😒|🤡|😬",

    # Slang signals
    r"\b(lmao|lol|bruh|smh|💀|kek|kappa)\b",
    r"\bmy\s+sides\b",                        # "my sides are gone" = can't believe it
    r"\bsure\s+(buddy|bro|man|pal|champ)\b",

    # Fan-specific inverted frustration
    r"\b(he('?s)?\s+(so|such\s+a))\s+(bad|terrible|awful|garbage|trash)\b",
    r"\bwhat\s+a\s+(joke|clown|fraud)\b",
    r"\bsure\s+he\s+(can|will|does)\b",
]
# ══════════════════════════════════════════════════════════════════════════════

_VALID_LABELS = ("Positive", "Neutral", "Negative")
_DIVIDER = "═" * 72


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compile_sarcasm() -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in SARCASM_PATTERNS)
    return re.compile(combined, re.IGNORECASE)


def _apply_prob_threshold(neg: float, neu: float, pos: float, thr: float) -> str:
    if max(neg, neu, pos) < thr:
        return "Uncertain"
    if pos >= neg and pos >= neu:
        return "Positive"
    if neg >= pos and neg >= neu:
        return "Negative"
    return "Neutral"


def _apply_score_cutoff(score: float, pos_cut: float, neg_cut: float) -> str:
    """3-class labeling from a scalar sentiment score (no uncertainty gate)."""
    if score >= pos_cut:
        return "Positive"
    if score <= neg_cut:
        return "Negative"
    return "Neutral"


def _metrics(y_true: list, y_pred: list, labels=_VALID_LABELS) -> dict:
    present = [l for l in labels if l in set(y_true) or l in set(y_pred)]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(present), zero_division=0)
    per = f1_score(y_true, y_pred, average=None, labels=list(present), zero_division=0)
    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "per_class": dict(zip(present, per)),
    }


def _fmt_row(name: str, n: int, m: dict) -> str:
    per = "  ".join(f"{k[:3]}: {v:.3f}" for k, v in m["per_class"].items())
    return (
        f"  {name:<30} n={n:<5}  "
        f"acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}   [{per}]"
    )


# ── Threshold calibration (no re-inference) ───────────────────────────────────

def _calibrate(
    human: list[str],
    neg_arr: np.ndarray,
    neu_arr: np.ndarray,
    pos_arr: np.ndarray,
    score_arr: np.ndarray,
) -> dict:
    """Grid-search thresholds using already-computed transformer probs."""

    # ── 1. UNCERTAIN_THRESHOLD sweep ─────────────────────────────────────────
    unc_sweep: list[dict] = []
    best_unc_thr  = _DEFAULT_UNC
    best_unc_f1   = -1.0
    best_coverage = 1.0

    for thr in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        preds = [
            _apply_prob_threshold(neg_arr[i], neu_arr[i], pos_arr[i], thr)
            for i in range(len(human))
        ]
        certain_mask = [p != "Uncertain" for p in preds]
        y_true_c = [h for h, m in zip(human, certain_mask) if m]
        y_pred_c = [p for p, m in zip(preds, certain_mask) if m]
        coverage = sum(certain_mask) / max(1, len(human))
        if not y_true_c:
            continue
        f1 = f1_score(y_true_c, y_pred_c, average="macro", labels=list(_VALID_LABELS), zero_division=0)
        unc_sweep.append({
            "threshold": thr,
            "macro_f1":  round(f1, 4),
            "coverage":  round(coverage, 4),
            "n_certain": sum(certain_mask),
        })
        if f1 > best_unc_f1:
            best_unc_f1   = f1
            best_unc_thr  = thr
            best_coverage = coverage

    # ── 2. Score-cutoff sweep (symmetric and asymmetric) ─────────────────────
    # Models a single continuous threshold T: score > T → Positive,
    # score < -T → Negative, else Neutral.  Also tries asymmetric pairs.
    cut_candidates = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    best_pos_cut = 0.10
    best_neg_cut = -0.10
    best_cut_f1  = -1.0

    for pos_cut in cut_candidates:
        for neg_cut in [-c for c in cut_candidates]:
            preds = [_apply_score_cutoff(float(s), pos_cut, neg_cut) for s in score_arr]
            f1 = f1_score(human, preds, average="macro", labels=list(_VALID_LABELS), zero_division=0)
            if f1 > best_cut_f1:
                best_cut_f1  = f1
                best_pos_cut = pos_cut
                best_neg_cut = neg_cut

    return {
        "unc_sweep":        unc_sweep,
        "best_unc_thr":     best_unc_thr,
        "best_unc_f1":      best_unc_f1,
        "best_coverage":    best_coverage,
        "best_pos_cut":     best_pos_cut,
        "best_neg_cut":     best_neg_cut,
        "best_cut_f1":      best_cut_f1,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate VADER vs transformer against human labels")
    ap.add_argument("csv",            help="Labeled CSV (human_label column filled)")
    ap.add_argument("--no-transformer", action="store_true", help="Skip transformer (VADER + calibration only)")
    ap.add_argument("--batch-size",   type=int, default=32)
    args = ap.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    if "human_label" not in df.columns:
        print("ERROR: 'human_label' column not found in CSV.", file=sys.stderr)
        sys.exit(1)

    df = df[df["human_label"].isin(_VALID_LABELS)].copy().reset_index(drop=True)
    if df.empty:
        print(
            "ERROR: No valid human labels found.\n"
            "       Fill the 'human_label' column with: Positive | Neutral | Negative",
            file=sys.stderr,
        )
        sys.exit(1)

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    texts = df[text_col].fillna("").astype(str).tolist()
    human = df["human_label"].tolist()

    print(f"\nLoaded {len(df)} labeled rows  (text column: '{text_col}')")
    dist = df["human_label"].value_counts().to_dict()
    print(f"Label distribution: {dist}")

    # ── Sarcasm slice ─────────────────────────────────────────────────────────
    sarc_re = _compile_sarcasm()
    sarc_mask_bool = (
        df[text_col].fillna("").astype(str)
        .apply(lambda t: bool(sarc_re.search(t)))
        .tolist()
    )
    n_sarc = sum(sarc_mask_bool)

    # ── VADER ─────────────────────────────────────────────────────────────────
    print("\nScoring with VADER…")
    from src.scoring import score_comments
    vader_out = score_comments(
        [{"text": t, "cleaned_text": t} for t in texts],
        text_col="cleaned_text",
        use_vader=True,
        run_emotion=False,
    )
    # VADER rarely produces Uncertain but can; map it to Neutral for comparison
    vader_labels = [
        r["sentiment_label"] if r["sentiment_label"] != "Uncertain" else "Neutral"
        for r in vader_out
    ]

    # ── Transformer ───────────────────────────────────────────────────────────
    trans_labels: list[str] = []
    neg_arr = neu_arr = pos_arr = score_arr = np.array([])

    if not args.no_transformer:
        print("Scoring with transformer (may take ~60 s on CPU)…")
        records = [{"text": t, "cleaned_text": t} for t in texts]
        trans_out = score_comments(
            records,
            text_col="cleaned_text",
            batch_size=args.batch_size,
            run_emotion=False,
        )
        trans_df = pd.DataFrame(trans_out)
        trans_labels = trans_df["sentiment_label"].tolist()
        neg_arr   = trans_df["neg_prob"].to_numpy()
        neu_arr   = trans_df["neu_prob"].to_numpy()
        pos_arr   = trans_df["pos_prob"].to_numpy()
        score_arr = trans_df["sentiment_score"].to_numpy()

        n_uncertain = sum(1 for l in trans_labels if l == "Uncertain")
        # Remap Uncertain → Neutral for direct comparison
        trans_mapped = [l if l != "Uncertain" else "Neutral" for l in trans_labels]

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{_DIVIDER}")
    print("  VALIDATION RESULTS")
    print(_DIVIDER)

    # All comments (Uncertain → Neutral)
    print(f"\n  ALL COMMENTS  (n={len(df)}, transformer-Uncertain mapped to Neutral)")
    print(_fmt_row("VADER", len(df), _metrics(human, vader_labels)))
    if not args.no_transformer:
        print(_fmt_row("Transformer", len(df), _metrics(human, trans_mapped)))
        print(
            f"    (transformer flagged {n_uncertain}/{len(df)} = "
            f"{n_uncertain/len(df)*100:.1f}% as Uncertain)"
        )

    # Certain-only sub-table
    if not args.no_transformer and n_uncertain < len(df):
        certain_idx = [i for i, l in enumerate(trans_labels) if l != "Uncertain"]
        h_cert = [human[i]        for i in certain_idx]
        v_cert = [vader_labels[i] for i in certain_idx]
        t_cert = [trans_labels[i] for i in certain_idx]
        print(f"\n  CERTAIN-ONLY SUBSET  (n={len(certain_idx)}  —  rows where transformer is not Uncertain)")
        print("  Note: this subset is selected by the model, so metrics are optimistically biased")
        print(_fmt_row("VADER  (on certain subset)", len(h_cert), _metrics(h_cert, v_cert)))
        print(_fmt_row("Transformer (certain only)", len(h_cert), _metrics(h_cert, t_cert)))

    # Sarcasm slice
    print(f"\n  SARCASM / INVERTED-SLANG SLICE")
    if n_sarc < 5:
        print(
            f"  Only {n_sarc} matching comments — too few to report reliably.\n"
            f"  Tip: widen SARCASM_PATTERNS at the top of this file."
        )
    else:
        h_sarc = [h for h, m in zip(human, sarc_mask_bool) if m]
        v_sarc = [l for l, m in zip(vader_labels, sarc_mask_bool) if m]
        print(_fmt_row("VADER  (sarcasm)", n_sarc, _metrics(h_sarc, v_sarc)))
        if not args.no_transformer:
            t_sarc = [l for l, m in zip(trans_mapped, sarc_mask_bool) if m]
            print(_fmt_row("Transformer (sarcasm)", n_sarc, _metrics(h_sarc, t_sarc)))
            delta_f1 = _metrics(h_sarc, t_sarc)["macro_f1"] - _metrics(h_sarc, v_sarc)["macro_f1"]
            sign = "+" if delta_f1 >= 0 else ""
            print(f"    Transformer advantage on sarcasm: {sign}{delta_f1:.3f} macro-F1")

    # Threshold calibration
    if not args.no_transformer and len(neg_arr) > 0:
        print(f"\n  THRESHOLD CALIBRATION  (grid search, no re-inference needed)")
        cal = _calibrate(human, neg_arr, neu_arr, pos_arr, score_arr)

        print(f"\n  Uncertainty-threshold sweep  (maximize macro-F1 on certain-only subset):")
        print(f"  {'threshold':>12}  {'macro-F1':>10}  {'coverage':>10}  {'n_certain':>10}")
        for row in cal["unc_sweep"]:
            marker = "  ← best" if abs(row["threshold"] - cal["best_unc_thr"]) < 0.001 else ""
            print(
                f"  {row['threshold']:>12.2f}  {row['macro_f1']:>10.4f}  "
                f"{row['coverage']*100:>9.1f}%  {row['n_certain']:>10}{marker}"
            )

        print(f"\n  ┌─ SUGGESTED VALUES  (review before applying — do NOT auto-apply) ─┐")
        print(f"  │")
        unc_changed = abs(cal["best_unc_thr"] - _DEFAULT_UNC) > 0.001
        prefix = "CHANGE" if unc_changed else "keep  "
        print(
            f"  │  UNCERTAIN_THRESHOLD  = {cal['best_unc_thr']:.2f}   [{prefix}; current = {_DEFAULT_UNC:.2f}]"
        )
        print(f"  │    macro-F1 on certain subset: {cal['best_unc_f1']:.4f}")
        print(f"  │    coverage (% certain):        {cal['best_coverage']*100:.1f}%")
        print(f"  │")

        pos_changed = abs(cal["best_pos_cut"] - 0.10) > 0.001
        neg_changed = abs(cal["best_neg_cut"] - (-0.10)) > 0.001
        pp = "CHANGE" if pos_changed else "keep  "
        np_ = "CHANGE" if neg_changed else "keep  "
        print(
            f"  │  Positive score cutoff  = +{cal['best_pos_cut']:.2f}   [{pp}; current = +0.10]"
        )
        print(
            f"  │  Negative score cutoff  = {cal['best_neg_cut']:.2f}    [{np_}; current = -0.10]"
        )
        print(f"  │    macro-F1 with score-cutoff labels: {cal['best_cut_f1']:.4f}")
        print(f"  │")
        print(f"  │  To apply: edit scoring.py (UNCERTAIN_THRESHOLD)")
        print(f"  │            and analytics.py / app.py (_action_signal cutoffs)")
        print(f"  └────────────────────────────────────────────────────────────────┘")

    print(f"\n{_DIVIDER}\n")


if __name__ == "__main__":
    main()
