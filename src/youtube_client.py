import re
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def _parse_duration(duration_str: str) -> int:
    """Convert ISO 8601 duration (PT4M13S) to total seconds."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str or "")
    if not match:
        return 0
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = int(match.group(3) or 0)
    return h * 3600 + m * 60 + s


class YouTubeClient:
    def __init__(self, api_key: str):
        self.service = build("youtube", "v3", developerKey=api_key)

    def get_channel_info(self, handle: str = "", channel_id: str = "") -> dict | None:
        """
        Resolve channel by handle (forHandle, 1 quota unit) or direct channel ID.
        Never uses search.list — that costs 100 units per call.
        """
        if channel_id:
            resp = self._channels_by_id(channel_id)
        else:
            handle = handle.lstrip("@")
            try:
                resp = self.service.channels().list(
                    part="snippet,statistics,contentDetails",
                    forHandle=handle,
                ).execute()
            except HttpError:
                return None

        if not resp.get("items"):
            return None

        item = resp["items"][0]
        snippet = item["snippet"]

        stats = item.get("statistics", {})
        uploads = item["contentDetails"]["relatedPlaylists"].get("uploads", "")

        return {
            "channel_id": item["id"],
            "title": snippet["title"],
            "description": snippet.get("description", ""),
            "subscriber_count": int(stats.get("subscriberCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
            "video_count": int(stats.get("videoCount", 0)),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "uploads_playlist_id": uploads,
            "custom_url": snippet.get("customUrl", ""),
            "published_at": snippet.get("publishedAt", ""),
        }

    def _channels_by_id(self, channel_id: str) -> dict:
        try:
            return self.service.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id,
            ).execute()
        except HttpError:
            return {}

    def get_video_ids(self, uploads_playlist_id: str, max_videos: int = 100) -> list[str]:
        video_ids: list[str] = []
        next_page_token = None

        while len(video_ids) < max_videos:
            batch_size = min(50, max_videos - len(video_ids))
            try:
                resp = self.service.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=batch_size,
                    pageToken=next_page_token,
                ).execute()
            except HttpError:
                break

            for item in resp.get("items", []):
                vid_id = item["snippet"]["resourceId"].get("videoId")
                if vid_id:
                    video_ids.append(vid_id)

            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids

    def get_video_details(self, video_ids: list[str]) -> pd.DataFrame:
        rows = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            try:
                resp = self.service.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(batch),
                ).execute()
            except HttpError:
                continue

            for item in resp.get("items", []):
                snippet = item["snippet"]
                stats = item.get("statistics", {})
                duration_secs = _parse_duration(
                    item.get("contentDetails", {}).get("duration", "")
                )
                rows.append(
                    {
                        "video_id": item["id"],
                        "title": snippet["title"],
                        "published_at": snippet["publishedAt"],
                        "thumbnail": snippet.get("thumbnails", {})
                        .get("medium", {})
                        .get("url", ""),
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                        "duration_secs": duration_secs,
                        "tags": snippet.get("tags", []),
                    }
                )

        return pd.DataFrame(rows)

    def get_comments(self, video_id: str, max_comments: int = 100) -> list[dict]:
        comments: list[dict] = []
        next_page_token = None

        while len(comments) < max_comments:
            batch_size = min(100, max_comments - len(comments))
            try:
                resp = self.service.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=batch_size,
                    textFormat="plainText",
                    order="relevance",
                    pageToken=next_page_token,
                ).execute()
            except HttpError:
                # Comments disabled or quota error — skip this video
                break

            for item in resp.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(
                    {
                        "video_id": video_id,
                        "author": top.get("authorDisplayName", "Unknown"),
                        "author_channel_id": top.get("authorChannelId", {}).get(
                            "value", ""
                        ),
                        "text": top.get("textOriginal", ""),
                        "like_count": top.get("likeCount", 0),
                        "published_at": top.get("publishedAt", ""),
                        "reply_count": item["snippet"].get("totalReplyCount", 0),
                    }
                )

            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        return comments

    def get_all_comments(
        self, video_ids: list[str], max_per_video: int = 100
    ) -> pd.DataFrame:
        all_comments: list[dict] = []
        for vid_id in video_ids:
            all_comments.extend(self.get_comments(vid_id, max_per_video))
        return pd.DataFrame(all_comments) if all_comments else pd.DataFrame()
