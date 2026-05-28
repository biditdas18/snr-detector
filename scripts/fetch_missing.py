#!/usr/bin/env python3
"""
Run this from YOUR OWN TERMINAL (not Claude Code) to fetch the 12 remaining
transcripts that YouTube blocked from the Claude Code IP.

Usage:
    pip install youtube-transcript-api
    python scripts/fetch_missing.py

Output:
    data/labels/transcripts_missing.json
    (tell Claude Code when done — it will merge and continue)
"""
import json, time
from youtube_transcript_api import YouTubeTranscriptApi

MISSING = [
    "p4VHMsIuPmk",  # TED-Ed: How do airplanes actually fly?        [general_education HIGH]
    "tJevBNQsKtU",  # PBS Space Time: The Edge of an Infinite Universe [general_education HIGH]
    "F3QpgXBtDeo",  # Kurzgesagt: How The Stock Exchange Works       [general_education MID]
    "72hlr-E7KA0",  # Wendover: How Airlines Price Flights           [general_education MID]
    "o1Y4Z0oh1GE",  # BBC Ideas: The quiet power of introverts       [general_education MID]
    "UEl3rUdsWXQ",  # Big Think: How the brain makes memories        [general_education MID]
    "RZ5dj-Ozwm0",  # Big Think: Michio Kaku Explains String Theory  [general_education MID]
    "47YHEVRZuXU",  # Economy Strategist: Yield Curve doom           [general_education NOISE]
    "rmVWId1Wg3k",  # CNBC: America's Debt Spiral                    [general_education NOISE]
    "DBe4yHPbUiw",  # William Spaniel: Dominoes to WW3               [general_education NOISE]
    "_9ZVEDXzVjE",  # Professor G: Stock Market Crash WARNING        [general_education NOISE]
    "cWX0Jt558l0",  # SciShow Space: What If Earth Stopped Spinning  [general_education NOISE]
]

OUTPUT = "data/labels/transcripts_missing.json"

api = YouTubeTranscriptApi()
results = {}
failed  = []

print(f"Fetching {len(MISSING)} transcripts from your IP...\n")

for i, vid in enumerate(MISSING, 1):
    try:
        tl = api.list(vid)
        t  = None
        for attempt in [
            lambda: tl.find_manually_created_transcript(['en']),
            lambda: tl.find_generated_transcript(['en']),
            lambda: tl.find_transcript(['en']),
            lambda: next(iter(tl)),
        ]:
            try:
                t = attempt(); break
            except Exception:
                continue
        if t is None:
            raise RuntimeError("no transcript found")
        snippets = t.fetch()
        text = " ".join(s.text for s in snippets).replace("\n", " ").strip()
        if len(text.split()) < 100:
            raise RuntimeError(f"transcript too short ({len(text.split())} words)")
        results[vid] = text
        print(f"[{i:02d}/{len(MISSING)}] ✓ {vid}  {len(text.split())} words")
    except Exception as e:
        failed.append(vid)
        print(f"[{i:02d}/{len(MISSING)}] ✗ {vid}  {e}")
    time.sleep(2)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(results)}/{len(MISSING)} transcripts → {OUTPUT}")
if failed:
    print(f"Failed: {failed}")
print("\nTell Claude Code you're done — it will merge and continue.")
