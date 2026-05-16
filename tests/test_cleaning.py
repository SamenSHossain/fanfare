"""
Unit tests for src/cleaning.py.

All test comments are representative of real YouTube fan comment patterns.
Run with: pytest tests/test_cleaning.py -v
"""

import pytest
from src.cleaning import clean_comments


# ── Helpers ────────────────────────────────────────────────────────────────────

def _record(text: str, **kwargs) -> dict:
    """Minimal comment record with required fields."""
    return {
        "text": text,
        "video_id": "v_test",
        "author": "test_user",
        "author_channel_id": "UC_test",
        "like_count": 0,
        "published_at": "2024-01-01T00:00:00Z",
        "reply_count": 0,
        **kwargs,
    }


def _one(text: str, **kwargs) -> dict:
    """Run clean_comments on a single record and return the enriched dict."""
    return clean_comments([_record(text, **kwargs)])[0]


# ── Spam detection ─────────────────────────────────────────────────────────────

class TestSpamURLs:
    def test_http_url(self):
        r = _one("Check out my video http://youtube.com/c/fakeaccount")
        assert r["is_spam"] is True

    def test_https_url(self):
        r = _one("Best highlights https://streamlink.tv/live amazing")
        assert r["is_spam"] is True

    def test_www_url(self):
        r = _one("Go to www.mysite.com for more basketball content")
        assert r["is_spam"] is True


class TestSpamHandles:
    def test_at_handle(self):
        r = _one("Follow @bestbasketball on Instagram for updates")
        assert r["is_spam"] is True

    def test_at_handle_mid_sentence(self):
        r = _one("Shoutout to @jaredmccain024 official page")
        assert r["is_spam"] is True


class TestSpamSelfPromo:
    def test_subscribe_to_my(self):
        r = _one("Please subscribe to my channel for more NBA content!")
        assert r["is_spam"] is True

    def test_check_out_my_channel(self):
        r = _one("Check out my channel for daily basketball highlights")
        assert r["is_spam"] is True

    def test_follow_me(self):
        r = _one("Follow me for more sports content every day")
        assert r["is_spam"] is True

    def test_link_in_bio(self):
        r = _one("link in my bio for the full breakdown")
        assert r["is_spam"] is True


class TestSpamScams:
    def test_crypto_bitcoin(self):
        r = _one("DM me for free Bitcoin 💰 limited time offer")
        assert r["is_spam"] is True

    def test_crypto_nft(self):
        r = _one("I made $5000 with NFT last month message me")
        assert r["is_spam"] is True

    def test_giveaway(self):
        r = _one("Big giveaway happening on my channel right now enter!")
        assert r["is_spam"] is True

    def test_dm_me(self):
        r = _one("dm me for the secret method to earn money fast")
        assert r["is_spam"] is True

    def test_earn_money(self):
        r = _one("I can show you how to earn $500 a day easily")
        assert r["is_spam"] is True

    def test_cashapp(self):
        r = _one("Send me your cashapp I'll send you free money")
        assert r["is_spam"] is True


class TestSpamBotPatterns:
    def test_lone_first(self):
        r = _one("First!")
        assert r["is_spam"] is True

    def test_first_no_punctuation(self):
        r = _one("first")
        assert r["is_spam"] is True

    def test_whos_watching_2025(self):
        r = _one("Who's watching in 2025? 🙌")
        assert r["is_spam"] is True

    def test_whos_here_in_year(self):
        r = _one("who's here in 2024 still bumping this")
        assert r["is_spam"] is True

    def test_still_watching(self):
        r = _one("Still watching in 2024 🔥🔥🔥")
        assert r["is_spam"] is True

    def test_like_if_watching(self):
        r = _one("Like if you're still watching this in 2025")
        assert r["is_spam"] is True

    def test_notification_squad(self):
        r = _one("Notification squad where you at")
        assert r["is_spam"] is True


class TestSpamLowInformation:
    def test_lone_emoji(self):
        r = _one("🔥")
        assert r["is_spam"] is True

    def test_multiple_emoji_only(self):
        r = _one("🏀🔥💯")
        assert r["is_spam"] is True

    def test_single_word_too_short(self):
        r = _one("ok")
        assert r["is_spam"] is True

    def test_single_meaningful_word(self):
        # One alphabetic token — still below MIN_TOKEN_COUNT
        r = _one("amazing")
        assert r["is_spam"] is True

    def test_empty_string(self):
        r = _one("")
        assert r["is_spam"] is True


class TestValidComments:
    def test_positive_fan_comment(self):
        r = _one("Jared McCain is such an incredible player, love watching him dominate!")
        assert r["is_spam"] is False

    def test_critical_comment(self):
        r = _one("He really struggled in the fourth quarter, defense needs serious work")
        assert r["is_spam"] is False

    def test_mixed_emoji_valid(self):
        r = _one("Rookie of the year no question 🏀 the numbers speak for themselves")
        assert r["is_spam"] is False

    def test_short_but_valid(self):
        r = _one("Great game today")
        assert r["is_spam"] is False

    def test_question_comment(self):
        r = _one("Does anyone know when his next game is streamed?")
        assert r["is_spam"] is False


# ── cleaned_text field ─────────────────────────────────────────────────────────

class TestCleanedText:
    def test_url_removed_from_cleaned(self):
        r = _one("Cool video https://spam.com check it out")
        assert "https://spam.com" not in r["cleaned_text"]

    def test_handle_removed_from_cleaned(self):
        r = _one("Love what @fakechannel is doing here")
        assert "@fakechannel" not in r["cleaned_text"]

    def test_original_text_never_mutated(self):
        original = "Go to https://spam.com and follow @spammer"
        r = _one(original)
        assert r["text"] == original

    def test_emoji_kept_in_cleaned(self):
        # Emoji should remain in cleaned_text for RoBERTa/VADER
        r = _one("Jared McCain is amazing 🏀🔥 best rookie ever")
        assert "🏀" in r["cleaned_text"]

    def test_whitespace_collapsed(self):
        r = _one("Jared   McCain    is    great")
        assert "  " not in r["cleaned_text"]


# ── Duplicate detection ────────────────────────────────────────────────────────

class TestDuplicates:
    def test_exact_duplicate(self):
        records = [
            _record("Jared McCain is the GOAT no cap"),
            _record("Jared McCain is the GOAT no cap"),
        ]
        results = clean_comments(records)
        assert results[0]["is_duplicate"] is False
        assert results[1]["is_duplicate"] is True

    def test_case_insensitive_duplicate(self):
        records = [
            _record("jared mccain is great"),
            _record("JARED MCCAIN IS GREAT"),
        ]
        results = clean_comments(records)
        assert results[1]["is_duplicate"] is True

    def test_emoji_stripped_for_dedup(self):
        records = [
            _record("Lets go Jared 🏀🔥"),
            _record("Lets go Jared"),
        ]
        results = clean_comments(records)
        assert results[1]["is_duplicate"] is True

    def test_punctuation_stripped_for_dedup(self):
        records = [
            _record("He's incredible!!!"),
            _record("Hes incredible"),
        ]
        results = clean_comments(records)
        assert results[1]["is_duplicate"] is True

    def test_first_occurrence_not_duplicate(self):
        r = _one("Unique and original fan comment right here")
        assert r["is_duplicate"] is False

    def test_distinct_comments_not_flagged(self):
        records = [
            _record("Amazing player with incredible court vision"),
            _record("His three point shooting is elite level"),
        ]
        results = clean_comments(records)
        assert results[0]["is_duplicate"] is False
        assert results[1]["is_duplicate"] is False

    def test_third_duplicate_also_flagged(self):
        records = [_record("great game")] * 3
        results = clean_comments(records)
        assert results[0]["is_duplicate"] is False
        assert results[1]["is_duplicate"] is True
        assert results[2]["is_duplicate"] is True


# ── Language detection ─────────────────────────────────────────────────────────

class TestLanguage:
    def test_english_detected(self):
        r = _one("Jared McCain is such an incredible young player for the Philadelphia 76ers")
        assert r["language"] == "en"

    def test_short_text_undetermined(self):
        # Below MIN_LANG_DETECT_LENGTH — we don't guess
        r = _one("amazing")
        assert r["language"] == "und"

    def test_spanish_detected(self):
        r = _one("Jared McCain es increíble, el mejor rookie de la temporada sin duda")
        assert r["language"] == "es"


# ── Output schema ──────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_all_four_fields_present(self):
        r = _one("Jared McCain is a fantastic player to watch")
        assert "cleaned_text" in r
        assert "is_spam" in r
        assert "language" in r
        assert "is_duplicate" in r

    def test_original_fields_preserved(self):
        rec = _record("Great game today", author="superfan", like_count=42)
        result = clean_comments([rec])[0]
        assert result["author"] == "superfan"
        assert result["like_count"] == 42

    def test_empty_input_returns_empty(self):
        assert clean_comments([]) == []

    def test_is_spam_is_bool(self):
        r = _one("Jared McCain is great player in the NBA")
        assert isinstance(r["is_spam"], bool)

    def test_is_duplicate_is_bool(self):
        r = _one("Jared McCain is great player in the NBA")
        assert isinstance(r["is_duplicate"], bool)
