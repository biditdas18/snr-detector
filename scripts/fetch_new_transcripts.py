#!/usr/bin/env python3
"""
Fetch 30 real YouTube transcripts per domain using YouTube Data API v3.
Deduplicates against the existing 60-video test set.

Usage:
    export YOUTUBE_API_KEY="your_key"
    python scripts/fetch_new_transcripts.py --domain career
    python scripts/fetch_new_transcripts.py --domain tech_ai
    python scripts/fetch_new_transcripts.py --domain general_education

Output:
    data/raw/new_transcripts_{domain}.jsonl   (one JSON per line, append-safe)
    data/raw/new_transcripts_{domain}_failed.txt
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# Only these tightly-scoped strings indicate a real IP block / rate limit.
# Do NOT add broad words like "block" — transcript-disabled messages also say "blocked".
_RATE_LIMIT_SIGNALS = ("429", "too many requests", "rate limit", "ip banned")

def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RATE_LIMIT_SIGNALS)

# ── config ────────────────────────────────────────────────────────────────────

TARGET_PER_DOMAIN = 30
SEARCH_BUFFER     = 80   # fetch up to this many unique candidates (many will lack EN captions)
MIN_WORDS         = 300
MAX_WORDS         = 8000
DELAY_MIN         = 8
DELAY_MAX         = 15
BACKOFF_STEPS     = [60, 120, 180]

DOMAIN_QUERIES = {
    "career": [
        "career development advice productivity",
        "how to get a job software engineer",
        "personal finance skills salary negotiation",
        "time management focus deep work",
        "how to learn faster study techniques",
    ],
    "tech_ai": [
        "how does machine learning work explained",
        "system design interview software engineering",
        "large language models how they work",
        "data structures algorithms tutorial",
        "software architecture explained",
    ],
    "general_education": [
        "how does the brain work neuroscience",
        "physics explained simply",
        "history of science discovery",
        "how economics works explained",
        "biology evolution explained simply",
    ],
}

# Channels known to produce HIGH or MID signal (allow-list bias)
PREFERRED_CHANNEL_IDS = {
    "tech_ai": [
        "UCVhQ2NnY5Rskt6UjCUkJ_DA",  # Fireship
        "UC9-y-6csu5WGm29I7JiwpnA",  # Computerphile
        "UCWX3yGbODI3BKMAxVFMlq5Q",  # ByteByteGo
        "UC0RhatS1pyxInC00YKjjBqQ",  # Fireship (backup)
        "UCJQMAI7645Kv3EtMBuVFMHw",  # 3Blue1Brown
    ],
    "career": [
        "UCoOae5nYA7VqaXzerajD0lg",  # Ali Abdaal
        "UC2eYFnH61tmytImy1mTYvhA",  # Thomas Frank
    ],
    "general_education": [
        "UCHnyfMqiRRG1u-2MsSQLbXA",  # Veritasium
        "UCZYTClx2T1of7BRZ86-8fow",  # SciShow
        "UCsooa4yRKGN_zEE8iknghZA",  # TED-Ed
        "UCsXVk37bltHxD1rDPwtNM8Q",  # Kurzgesagt
    ],
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_existing_ids() -> set:
    path = Path("data/labels/video_selection_60.csv")
    df = pd.read_csv(path)
    ids = set(df["video_id"].dropna().astype(str).tolist())
    # also sweep transcripts.csv
    t_path = Path("data/transcripts/transcripts.csv")
    if t_path.exists():
        df2 = pd.read_csv(t_path)
        ids.update(df2["video_id"].dropna().astype(str).tolist())
    return ids


def load_already_fetched(out_path: Path) -> set:
    ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["video_id"])
                except Exception:
                    pass
    return ids


def search_videos(youtube, query: str, existing_ids: set, max_results: int = 10) -> list[dict]:
    resp = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        videoDuration="medium",       # 4–20 min — good signal density
        videoCaption="closedCaption", # must have captions
        relevanceLanguage="en",
        maxResults=max_results,
    ).execute()

    candidates = []
    for item in resp.get("items", []):
        vid_id = item["id"].get("videoId")
        if not vid_id or vid_id in existing_ids:
            continue
        candidates.append({
            "video_id": vid_id,
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "channel_id": item["snippet"]["channelId"],
        })
    return candidates


COOKIE_FILE = Path("data/raw/yt_cookies.txt")

def _make_session() -> "requests.Session | None":
    """Load a requests.Session with YouTube cookies if the cookie file exists."""
    import requests, http.cookiejar
    if not COOKIE_FILE.exists():
        return None
    session = requests.Session()
    jar = http.cookiejar.MozillaCookieJar(str(COOKIE_FILE))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies = jar  # type: ignore[assignment]
        return session
    except Exception as e:
        print(f"  [warn] could not load cookie file: {e}")
        return None

def fetch_transcript(video_id: str) -> tuple[str | None, bool]:
    """Returns (transcript_text, is_rate_limited).
    - (text, False)  → success
    - (None, False)  → video simply has no usable EN captions — skip, no backoff
    - (None, True)   → suspected IP block / rate limit — trigger backoff
    """
    session = _make_session()
    api = YouTubeTranscriptApi(http_client=session) if session else YouTubeTranscriptApi()
    try:
        tl = api.list(video_id)
        t = None
        for attempt in [
            lambda: tl.find_manually_created_transcript(["en"]),
            lambda: tl.find_generated_transcript(["en"]),
            lambda: tl.find_transcript(["en"]),
            lambda: next(iter(tl)),
        ]:
            try:
                t = attempt()
                break
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
                continue
            except Exception:
                continue
        if t is None:
            return None, False
        snippets = t.fetch()
        text = " ".join(s.text for s in snippets).replace("\n", " ").strip()
        words = len(text.split())
        if words < MIN_WORDS or words > MAX_WORDS:
            return None, False
        return text, False
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None, False
    except Exception as e:
        rate_lim = _is_rate_limited(e)
        print(f"    [debug] exception ({type(e).__name__}): {str(e)[:120]}  → rate_limited={rate_lim}")
        return None, rate_lim


def sleep_with_backoff(attempt: int):
    if attempt == 0:
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    else:
        idx = min(attempt - 1, len(BACKOFF_STEPS) - 1)
        secs = BACKOFF_STEPS[idx]
        print(f"  Backoff: sleeping {secs}s (attempt {attempt})")
        time.sleep(secs)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_QUERIES.keys()))
    args = parser.parse_args()
    domain = args.domain

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY not set in environment")

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path     = out_dir / f"new_transcripts_{domain}.jsonl"
    failed_path  = out_dir / f"new_transcripts_{domain}_failed.txt"

    existing_ids     = load_existing_ids()
    already_fetched  = load_already_fetched(out_path)
    seen             = existing_ids | already_fetched

    print(f"Domain: {domain}")
    print(f"Existing IDs to skip: {len(existing_ids)}")
    print(f"Already fetched this run: {len(already_fetched)}")
    print(f"Target: {TARGET_PER_DOMAIN}  |  Buffer cap: {SEARCH_BUFFER}\n")

    youtube = build("youtube", "v3", developerKey=api_key)

    # Collect candidates across queries until we have enough
    # Each query returns up to 10; we run multiple pages per query to fill the buffer
    candidates: list[dict] = []
    seen_cand_ids: set[str] = set()

    for query in DOMAIN_QUERIES[domain]:
        if len(candidates) >= SEARCH_BUFFER:
            break
        try:
            results = search_videos(youtube, query, seen, max_results=10)
            added = 0
            for r in results:
                if r["video_id"] not in seen_cand_ids:
                    candidates.append(r)
                    seen_cand_ids.add(r["video_id"])
                    added += 1
            print(f"Query '{query}': +{added} candidates  (total {len(candidates)})")
        except Exception as e:
            print(f"Search error for '{query}': {e}")
        time.sleep(1)

    # If still short, repeat queries with pageToken or slightly varied terms
    if len(candidates) < SEARCH_BUFFER:
        extra_queries = [q + " tips" for q in DOMAIN_QUERIES[domain]] + \
                        [q + " tutorial" for q in DOMAIN_QUERIES[domain]]
        for query in extra_queries:
            if len(candidates) >= SEARCH_BUFFER:
                break
            try:
                results = search_videos(youtube, query, seen, max_results=10)
                added = 0
                for r in results:
                    if r["video_id"] not in seen_cand_ids:
                        candidates.append(r)
                        seen_cand_ids.add(r["video_id"])
                        added += 1
                if added:
                    print(f"Extra query '{query}': +{added} candidates  (total {len(candidates)})")
            except Exception as e:
                print(f"Search error for '{query}': {e}")
            time.sleep(1)

    # Prioritize preferred channels
    preferred = PREFERRED_CHANNEL_IDS.get(domain, [])
    candidates.sort(key=lambda c: (0 if c["channel_id"] in preferred else 1))
    candidates = candidates[:SEARCH_BUFFER]

    print(f"\nFetching transcripts for {len(candidates)} candidates...\n")

    fetched = list(already_fetched)  # track count
    failed  = []
    backoff_attempt = 0

    for i, cand in enumerate(candidates):
        if len(fetched) >= TARGET_PER_DOMAIN:
            print(f"Reached target of {TARGET_PER_DOMAIN}. Done.")
            break

        vid_id = cand["video_id"]
        if vid_id in seen:
            continue

        print(f"[{len(fetched):02d}/{TARGET_PER_DOMAIN}] {vid_id}  {cand['title'][:60]}")

        transcript, rate_limited = fetch_transcript(vid_id)

        if transcript is None:
            if rate_limited:
                backoff_attempt += 1
                print(f"  ✗ rate limited — backoff {BACKOFF_STEPS[min(backoff_attempt-1, len(BACKOFF_STEPS)-1)]}s")
                sleep_with_backoff(backoff_attempt)
            else:
                # Video simply has no EN captions — skip immediately, no penalty
                print(f"  ✗ no EN captions (skipping)")
                failed.append(vid_id)
            continue

        backoff_attempt = 0
        record = {
            "video_id":   vid_id,
            "url":        f"https://www.youtube.com/watch?v={vid_id}",
            "title":      cand["title"],
            "channel":    cand["channel"],
            "domain":     domain,
            "word_count": len(transcript.split()),
            "transcript": transcript,
        }

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        seen.add(vid_id)
        fetched.append(vid_id)
        print(f"  ✓ {len(transcript.split())} words  → saved")
        sleep_with_backoff(0)

    with open(failed_path, "w") as f:
        f.write("\n".join(failed))

    print(f"\nDone. {len(fetched)} transcripts saved → {out_path}")
    print(f"Failed: {len(failed)} → {failed_path}")
    if len(fetched) < TARGET_PER_DOMAIN:
        print(f"WARNING: only got {len(fetched)}/{TARGET_PER_DOMAIN}. Re-run to top up.")


if __name__ == "__main__":
    main()
