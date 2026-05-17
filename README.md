# Fanfare

YouTube fan intelligence dashboard for **Jared McCain** — built for marketing and social media leads who need to understand and act on fan activity without wading through raw data.

---

## Roadmap

The three highest-value additions with more time, in priority order:

- **Cross-channel benchmarking.** The sentiment score and engagement rate only matter relative to something. Pull two or three comparable athletes (Scoot Henderson, Paolo Banchero) and overlay their numbers so the lead can answer "are we at 72% positive because we're doing well, or because sports YouTube skews positive in general?" Without a baseline, a single number tells you nothing.

- **Scheduled weekly digest.** The lead shouldn't have to open a dashboard to stay informed. A Monday-morning Slack or email summarising what changed since last week — which alert fired, which video outperformed, whether sentiment trended up or down — closes the action loop entirely. The data pipeline is already built; it just needs a trigger and a formatter.

- **Reply draft suggestions.** The hardest part of community management isn't knowing which comments to respond to — it's writing the response. Pipe the top positive and critical comments through an LLM to draft reply copy. The fan segmentation already identifies which commenters are worth prioritising (Advocates vs. At-Risk); the LLM step turns that signal into something the social team can act on in 30 seconds.

- **Shorts vs. long-form split.** Shorts and long-form videos attract different audiences and drive fundamentally different comment behaviour. A 60-second highlight gets reactive emoji comments; a 20-minute breakdown gets substantive fan discussion. Mixing them in a single sentiment average obscures both signals.

---

## Quick start (no setup required)

The API key and channel are pre-configured. Clone the repo, install dependencies, and run:

```bash
git clone https://github.com/SamenSHossain/fanfare.git
cd fanfare
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**, click **Refresh Data** in the sidebar, and wait ~60–90 seconds for the full pipeline to run.

### Credentials for testing

| What | Value |
|------|-------|
| YouTube API key | `AIzaSyBHCfCa25OzyRfLXSqWZ1IPjRgVAD6DgLg` |
| Channel | `@jaredmccain024` (Jared McCain, NBA guard) |
| Daily API quota | 10,000 units free — a full run costs ~100 units |

Both are hardcoded in `app.py` — no `.env` file or secrets needed to run the demo.

---

## Walkthrough

### Sidebar

- **Date range** — filters all charts and metrics to videos published in the last 7, 28, or 90 days, or shows all time. Start with **Last 28 days** for a representative view.
- **Refresh Data** — fetches from the YouTube API and runs the full analysis pipeline (sentiment, topics, fan segments, alerts). Takes ~60–90 seconds. Results are cached for 1 hour.
- **Advanced** — exposes the raw API fetch parameters (videos to fetch, comments per video). Leave at defaults unless you want to reduce quota usage or get a faster preview.
- **Last updated** — shows the timestamp of the most recent fetch.

### Tab: Overview

The landing tab. Shows:

1. **Active alerts** at the top — any statistically significant signals that cleared multiple-comparisons correction. Each shows magnitude and a direct YouTube link. See the Alerts tab for full detail.
2. **Key insights** — auto-generated action cards: top sentiment signal, most viral video, highest-engagement video, super fan to engage, top keyword sentiment. All links are clickable.

### Tab: Engagement

Video performance charts filtered to the selected date range:

- Views per video over time (color = engagement rate)
- Top 10 videos by views
- Views vs. Likes scatter (bubble size = comment volume)
- Summary row: avg views, likes, comments, engagement rate per video

### Tab: Sentiment

Fan comment sentiment analysis. Read the **yellow info box at the top** before interpreting numbers — it explains what the model can and cannot detect (target attribution, sarcasm).

- **5 metrics row** — Positive / Neutral / Negative / Uncertain / Likely Sarcastic counts with percentages
- **Distribution pie** + **% summary** — headline Positive% with delta vs. Negative%; Like-Weighted and Flat Mean scores; Endorsed Gap signal
- **Sentiment trend** — like-weighted sentiment per video over time, with flat mean overlay
- **Emotion breakdown** — 7-class emotion distribution (joy, anger, sadness, etc.)
- **Video Deep Dive** — pick any video from the dropdown to see its own sentiment and emotion breakdown, plus sample comments by emotion
- **Sample Comments** — most positive, most critical, and uncertain examples

**Interpreting the numbers:**
- Positive% is of *confident* comments only (Uncertain excluded — see `UNCERTAIN_THRESHOLD = 0.55` in `src/scoring.py`)
- Like-Weighted score weights each comment's sentiment by `log(1 + likes)` so highly-endorsed comments carry more weight
- The Gap signal fires when liked comments systematically diverge from the average

### Tab: Top Fans

k-means fan segmentation (k selected by silhouette score) across 6 features: comment count, videos commented, average sentiment, likes earned, recency, comment consistency.

- **Segment cards** — each cluster shows its label, size, recommended action, and centroid profile (z-scores vs. average fan)
- **Fan Map** — scatter of all fans colored by segment (activity × sentiment, bubble = likes)
- **Fan Lookup** — full sortable table with YouTube channel links

### Tab: Trending Topics

BERTopic semantic clustering of all clean comments (all-MiniLM-L6-v2 embeddings, TruncatedSVD, KeyBERTInspired labels).

- **Prominence bar chart** — topics ranked by Σ log(1 + likes), colored by sentiment signal
- **Sentiment scatter** — prominence vs. like-weighted sentiment for topics with ≥ 20 confident comments
- **Topic cards** — per-topic: action signal, comment count, sample comments

### Tab: All Videos

Full video table (all fetched videos, regardless of date range filter) with ▶ Watch links and CSV export. Below it: **Comment Explorer** with sentiment/emotion filters and a "Show technical columns" toggle for probability scores.

### Tab: Alerts

Statistically corrected signals across three families:

| Family | Method | What it tests |
|--------|--------|---------------|
| Sentiment spike | One-sample t-test per video vs. channel mean | Bonferroni/BH corrected |
| Velocity anomaly | Z-score of comment count vs. channel distribution | Corrected sigma cutoff |
| Keyword shift | One-sample t-test per top keyword vs. channel mean | Bonferroni/BH corrected |

Each alert shows: recommended action, YouTube link, sample comments that drove the signal. Statistical details (p-values, adjusted threshold) are collapsed behind **Statistical details** — the marketing lead doesn't need them.

The methodology expander at the bottom explains Bonferroni vs. BH, the sample guard, and the effect-size gate.

---

## Pipeline architecture

```
YouTube API
    └── YouTubeClient (src/youtube_client.py)
            ├── get_channel_info      1 quota unit
            ├── get_video_ids         1 unit/page (50 videos/page)
            ├── get_video_details     1 unit/batch of 50
            └── get_all_comments      1 unit/page (100 comments/page)

clean_comments (src/cleaning.py)
    └── spam filter, dedup, language detection, text normalization

score_comments (src/scoring.py)
    ├── cardiffnlp/twitter-roberta-base-sentiment-latest  (EN)
    ├── cardiffnlp/twitter-xlm-roberta-base-sentiment     (non-EN)
    └── j-hartmann/emotion-english-distilroberta-base     (all)

run_topic_model (src/topics.py)
    └── BERTopic + all-MiniLM-L6-v2 + TruncatedSVD + KeyBERTInspired

run_fan_segmentation (src/fans.py)
    └── k-means, k selected by silhouette score

run_alerts (src/alerts.py)
    └── three families, Bonferroni correction by default
```

All five stages run inside `@st.cache_data(ttl=3600)` — results are in-memory for one hour and recomputed on Refresh.

---

## Quota usage

A full run at defaults (95 videos, 100 comments/video):

| Call | Units |
|------|-------|
| `channels.list` | 1 |
| `playlistItems.list` (~2 pages) | 2 |
| `videos.list` (~2 batches of 50) | 2 |
| `commentThreads.list` (~95 pages) | 95 |
| **Total** | **~100** |

`search.list` (100 units/call) is intentionally never used.

---

## Stack

- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — sentiment
- [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) — emotion
- [BERTopic](https://maartengr.github.io/BERTopic/) — topic modeling
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — embeddings
- [Streamlit](https://streamlit.io) — dashboard framework
- [Plotly](https://plotly.com/python/) — charts
- [scikit-learn](https://scikit-learn.org) — k-means, TruncatedSVD, silhouette
- [scipy](https://scipy.org) — t-tests, normal distribution for alert correction
