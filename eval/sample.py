#!/usr/bin/env python3
"""
Draw a stratified random sample of real fan comments for manual labeling.

Usage
-----
    python eval/sample.py
    python eval/sample.py --n 250 --videos 25 --per-video 120 --seed 42
    python eval/sample.py --out eval/round2.csv --seed 99   # different sample

Output CSV columns
------------------
    comment_id    YouTube comment ID (for traceability)
    video_id      Source video (for stratification context)
    like_count    Raw likes (gives labeler engagement context)
    language      Detected language code
    text          Original comment text (label THIS)
    cleaned_text  Cleaned version the models will actually score
    human_label   BLANK — fill with Positive | Neutral | Negative

Stratification
--------------
Slots are allocated proportionally to each video's comment volume, with a
minimum of 1 slot per video so no video is entirely excluded.  The final
sample is shuffled so video blocks aren't grouped.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

YOUTUBE_API_KEY = "AIzaSyBHCfCa25OzyRfLXSqWZ1IPjRgVAD6DgLg"
CHANNEL_HANDLE = "jaredmccain024"


# ── Stratified sampler ────────────────────────────────────────────────────────

def _stratified_sample(df: pd.DataFrame, n: int, group_col: str, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    groups = [(gid, grp.copy()) for gid, grp in df.groupby(group_col)]
    total = len(df)

    # Proportional allocation with floor of 1
    slots: dict = {}
    for gid, grp in groups:
        slots[gid] = max(1, round(n * len(grp) / total))

    # Trim or pad to exactly n
    allocated = sum(slots.values())
    delta = n - allocated
    # Sort largest groups first so adjustments land there
    order = [gid for gid, _ in sorted(groups, key=lambda x: -len(x[1]))]
    for i in range(abs(delta)):
        gid = order[i % len(order)]
        if delta > 0:
            slots[gid] += 1
        elif slots[gid] > 0:
            slots[gid] -= 1

    pieces: list[pd.DataFrame] = []
    for gid, grp in groups:
        k = min(slots[gid], len(grp))
        if k == 0:
            continue
        idx = rng.sample(list(grp.index), k)
        pieces.append(grp.loc[idx])

    return pd.concat(pieces, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Sample comments for manual labeling")
    ap.add_argument("--n",         type=int, default=250, help="Target sample size (default 250)")
    ap.add_argument("--videos",    type=int, default=25,  help="Videos to draw from (default 25)")
    ap.add_argument("--per-video", type=int, default=120, help="Max comments per video (default 120)")
    ap.add_argument("--seed",      type=int, default=42,  help="Random seed (default 42)")
    ap.add_argument("--out",       default="eval/to_label.csv", help="Output CSV path")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from src.youtube_client import YouTubeClient
    from src.cleaning import clean_comments

    print(f"Fetching channel info for @{CHANNEL_HANDLE}…")
    client = YouTubeClient(YOUTUBE_API_KEY)
    channel = client.get_channel_info(handle=CHANNEL_HANDLE)
    if not channel:
        print("ERROR: could not fetch channel. Check API key.", file=sys.stderr)
        sys.exit(1)
    print(f"  Channel: {channel['title']}")

    print(f"Fetching up to {args.videos} video IDs…")
    video_ids = client.get_video_ids(channel["uploads_playlist_id"], args.videos)
    print(f"  Got {len(video_ids)} videos")

    print(f"Fetching up to {args.per_video} comments per video…")
    raw_df = client.get_all_comments(video_ids, max_per_video=args.per_video)
    print(f"  Raw: {len(raw_df):,} comments")

    print("Cleaning (spam / near-duplicate filter)…")
    records = clean_comments(raw_df.to_dict("records"))
    comments_df = pd.DataFrame(records)
    clean_mask = ~comments_df["is_spam"] & ~comments_df["is_duplicate"]
    pool = comments_df[clean_mask].copy().reset_index(drop=True)
    n_videos_in_pool = pool["video_id"].nunique()
    print(f"  Clean pool: {len(pool):,} comments across {n_videos_in_pool} videos")

    # Sample
    n_target = min(args.n, len(pool))
    if n_target < args.n:
        print(f"  Warning: only {len(pool)} clean comments — sampling all of them")
    sample = _stratified_sample(pool, n_target, "video_id", args.seed)
    # Shuffle so video blocks aren't grouped in the CSV
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Select labeler-facing columns
    keep = ["comment_id", "video_id", "like_count", "language", "text", "cleaned_text"]
    keep = [c for c in keep if c in sample.columns]
    out = sample[keep].copy()
    out["human_label"] = ""

    out.to_csv(out_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    per_vid = sample.groupby("video_id").size().sort_values(ascending=False)
    lang_counts = sample["language"].value_counts().to_dict() if "language" in sample.columns else {}

    print(f"\n{'─'*60}")
    print(f"  Sampled {len(sample):,} comments from {sample['video_id'].nunique()} videos")
    print(f"  Output:  {out_path}")
    print(f"  Seed:    {args.seed}")
    print(f"\n  Per-video breakdown (top 10):")
    for vid, cnt in per_vid.head(10).items():
        print(f"    {vid}  →  {cnt} comments")
    if len(per_vid) > 10:
        print(f"    … and {len(per_vid) - 10} more videos")
    if lang_counts:
        print(f"\n  Language mix: {lang_counts}")
    print(f"{'─'*60}")
    print(f"\n  Next steps:")
    print(f"  1. Open {out_path} in any spreadsheet editor")
    print(f"  2. Fill 'human_label' column: Positive | Neutral | Negative")
    print(f"     (leave blank to skip a row; comment the text column, not cleaned_text)")
    print(f"  3. Run: python eval/score.py {out_path}")
    print()


if __name__ == "__main__":
    main()
