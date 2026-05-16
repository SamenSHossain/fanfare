# Fanfare

YouTube fan intelligence dashboard for **Jared McCain** — built for marketing and social media leads who need to understand and act on fan activity without wading through raw data.

## Demo

> Deploy to [Streamlit Community Cloud](https://share.streamlit.io) and paste the link here.

## Features

| Tab | What you get |
|-----|-------------|
| **Key Insights** | Auto-generated action items: top sentiment signal, most viral video, super fan to engage, hottest keyword |
| **📊 Engagement** | Views/likes/comments per video over time, top-10 by views, engagement rate scatter |
| **💬 Sentiment** | Positive/Neutral/Negative breakdown, sentiment gauge, trend line across videos, sample best and worst comments |
| **🏆 Top Fans** | Leaderboard ranked by comment volume + sentiment; Super Fan callouts for ambassador/giveaway targeting |
| **🔥 Trending Topics** | Top 30 keywords, keyword × sentiment scatter, action table flagging what to amplify vs. monitor |
| **📋 All Videos** | Sortable full dataset with CSV export for videos and comments |

## Setup

**Prerequisites:** Python 3.10+, a [YouTube Data API v3 key](https://console.cloud.google.com) (free, no credit card)

```bash
git clone https://github.com/SamenSHossain/fanfare.git
cd fanfare
pip install -r requirements.txt
cp .env.example .env        # paste your API key into .env
streamlit run app.py
```

The app reads your key from `.env` automatically. You can also enter it directly in the sidebar at runtime.

## Testing Instructions

1. Run the app with `streamlit run app.py`
2. Enter a YouTube API v3 key in the sidebar (or set `YOUTUBE_API_KEY` in `.env`)
3. Leave the channel set to `@JaredMcCain` (default) and click **Fetch & Analyze**
4. Adjust the **Videos** and **Comments per video** sliders to control how much data is pulled
5. Explore each tab — the Key Insights panel at the top gives the fastest summary

**Quota note:** The app uses only `channels.list` (1 unit), `playlistItems.list` (1 unit/page), `videos.list` (1 unit/50 videos), and `commentThreads.list` (1 unit/page). `search.list` (100 units/call) is intentionally disabled. A typical run of 20 videos × 100 comments costs roughly 60–80 units out of the 10,000 free daily quota.

## Deploying to Streamlit Community Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select `SamenSHossain/fanfare`, set entrypoint to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   YOUTUBE_API_KEY = "your_key_here"
   ```
5. Click **Deploy** — you'll get a public `*.streamlit.app` URL

## Roadmap

With more time, the highest-value additions would be:

- **Cross-channel benchmarking** — pull two or three comparable athletes and overlay their engagement rates and sentiment scores so the marketing lead has context ("McCain's 5.2% engagement rate vs. league average of 3.1%")
- **Comment reply suggestions** — pipe the top positive/critical comments through an LLM to draft reply copy, saving the social team time on the hardest part of community management
- **Scheduled digests** — a weekly email or Slack message with the Key Insights delta (what changed since last week), so the lead doesn't need to open the dashboard to stay informed
- **Shorts vs. long-form split** — separate the engagement analytics by video type, since Shorts and long-form drive very different comment behaviors and audience segments
- **Keyword alerts** — let the user pin specific keywords (e.g. a sponsor name, a controversy term) and get notified when mention volume spikes

## Stack

- [YouTube Data API v3](https://developers.google.com/youtube/v3) — channel, video, and comment data
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — social-media-tuned sentiment analysis, no API key required
- [Streamlit](https://streamlit.io) — dashboard framework
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org) — data processing
