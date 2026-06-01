#!/usr/bin/env python3
"""
Labels all 150 real YouTube transcripts using 3-model LLM ensemble:
  - GPT-4o mini (OpenAI)
  - Gemini 2.5 Flash (Google)
  - Llama 3.3 70B (Groq)

Majority vote determines final label.
Saves progress after every transcript — safe to interrupt.

Input:
  data/labels/review_queue.csv          (60 original transcripts)
  data/raw/new_transcripts_*.jsonl      (90 new transcripts)

Output:
  data/labels/all_150_silver_labels.csv
"""

import csv
import json
import os
import time
from collections import Counter

import openai
from google import genai as google_genai
from groq import Groq

# ── PATHS ────────────────────────────────────────────────────
BASE        = "/Users/biditdas/Desktop/snr-submission/snr-detector"
ORIG_CSV    = os.path.join(BASE, "data/labels/review_queue.csv")
NEW_DIR     = os.path.join(BASE, "data/raw")
OUTPUT      = os.path.join(BASE, "data/labels/all_150_silver_labels.csv")
RUBRIC_PATH = "/Users/biditdas/Desktop/snr-submission/rubric-calibration-agent/reports/rubric_calibration/calibrated_rubrics.json"

# ── MODELS ───────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE  = 0.0
MAX_TOKENS   = 300

# Normalize domain names from raw files to rubric keys
DOMAIN_NORM = {
    "career":             "career_selfimprovement",
    "career_selfimprovement": "career_selfimprovement",
    "tech_ai":            "tech_ai",
    "general_education":  "general_education",
}
# ─────────────────────────────────────────────────────────────


def load_rubrics():
    with open(RUBRIC_PATH) as f:
        return json.load(f)


def build_prompt(transcript: str, domain: str, rubric: dict) -> str:
    high = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["HIGH"]))
    low  = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(rubric["LOW"]))
    return f"""You are an educational content quality evaluator for the {domain} domain.

DOMAIN CONTEXT: {rubric['domain_context']}

HIGH SIGNAL criteria (content must satisfy at least 2):
{high}

LOW SIGNAL criteria (any one dominant = LOW):
{low}

CONFLICT RULE: {rubric['conflict_rule']}

TRANSCRIPT:
---
{transcript[:3000]}
---

Respond in EXACTLY this format, nothing else:
LABEL: HIGH or LOW
CONFIDENCE: HIGH or MEDIUM or LOW
REASON: one sentence citing the specific criterion"""


def call_openai(prompt: str, client) -> str:
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        text = r.choices[0].message.content.strip()
        return "HIGH" if "LABEL: HIGH" in text else "LOW"
    except Exception as e:
        print(f"    OpenAI error: {str(e)[:80]}")
        return None


def call_gemini(prompt: str, client) -> str:
    try:
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = r.text.strip()
        return "HIGH" if "LABEL: HIGH" in text else "LOW"
    except Exception as e:
        print(f"    Gemini error: {str(e)[:80]}")
        return None


def call_groq(prompt: str, client) -> str:
    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        text = r.choices[0].message.content.strip()
        return "HIGH" if "LABEL: HIGH" in text else "LOW"
    except Exception as e:
        print(f"    Groq error: {str(e)[:80]}")
        return None


def majority_vote(votes: list) -> tuple:
    valid = [v for v in votes if v in ["HIGH", "LOW"]]
    if not valid:
        return "LOW", 0.0
    count = Counter(valid)
    label = count.most_common(1)[0][0]
    high  = count.get("HIGH", 0)
    low   = count.get("LOW",  0)
    total = len(valid)
    weighted = 1.0 if (high == total or low == total) else 0.667
    return label, weighted


def load_all_transcripts() -> list:
    rows = []

    # Original 60
    with open(ORIG_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "source":     "original_60",
                "domain":     DOMAIN_NORM.get(r["domain"], r["domain"]),
                "transcript": r["transcript"],
                "video_id":   "",
                "title":      ""
            })

    # New 90 — one file per domain
    for fname, raw_domain in [
        ("new_transcripts_career.jsonl",            "career"),
        ("new_transcripts_tech_ai.jsonl",           "tech_ai"),
        ("new_transcripts_general_education.jsonl", "general_education"),
    ]:
        path = os.path.join(NEW_DIR, fname)
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append({
                        "source":     "new_90",
                        "domain":     DOMAIN_NORM.get(r.get("domain", raw_domain), raw_domain),
                        "transcript": r.get("transcript", ""),
                        "video_id":   r.get("video_id", ""),
                        "title":      r.get("title", "")
                    })

    print(f"Loaded {len(rows)} transcripts total")
    return rows


def load_existing_output() -> set:
    if not os.path.exists(OUTPUT):
        return set()
    with open(OUTPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    done = {(r["source"], r["video_id"], r["title"][:30]) for r in rows}
    print(f"Resuming — {len(done)} already labeled")
    return done


def main():
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gemini_client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    groq_client   = Groq(api_key=os.environ["GROQ_API_KEY"])

    rubrics     = load_rubrics()
    transcripts = load_all_transcripts()
    done        = load_existing_output()

    fieldnames = [
        "source", "domain", "video_id", "title",
        "transcript", "signal_level",
        "openai_label", "gemini_label", "groq_label",
        "weighted_agree", "all_agree"
    ]

    file_exists = os.path.exists(OUTPUT)
    out_file = open(OUTPUT, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    labeled = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"LABELING {len(transcripts)} TRANSCRIPTS")
    print(f"Models: {OPENAI_MODEL} | {GEMINI_MODEL} | {GROQ_MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"{'='*60}\n")

    for i, t in enumerate(transcripts):
        key = (t["source"], t["video_id"], t["title"][:30])
        if key in done:
            skipped += 1
            continue

        domain = t["domain"]
        rubric = rubrics.get(domain)
        if not rubric:
            print(f"  No rubric for '{domain}' — skipping")
            continue

        label_str = t["title"][:45] if t["title"] else "original transcript"
        print(f"[{i+1}/{len(transcripts)}] {domain} | {label_str}")

        prompt = build_prompt(t["transcript"], domain, rubric)

        openai_label = call_openai(prompt, openai_client)
        time.sleep(0.5)
        gemini_label = call_gemini(prompt, gemini_client)
        time.sleep(0.5)
        groq_label   = call_groq(prompt, groq_client)

        votes = [openai_label, gemini_label, groq_label]
        valid = [v for v in votes if v]

        if len(valid) < 2:
            print(f"  WARNING: only {len(valid)} valid votes — skipping")
            continue

        label, weighted = majority_vote(valid)
        all_agree = len(set(valid)) == 1

        row = {
            "source":         t["source"],
            "domain":         domain,
            "video_id":       t["video_id"],
            "title":          t["title"],
            "transcript":     t["transcript"],
            "signal_level":   label,
            "openai_label":   openai_label or "",
            "gemini_label":   gemini_label or "",
            "groq_label":     groq_label or "",
            "weighted_agree": round(weighted, 3),
            "all_agree":      all_agree
        }

        writer.writerow(row)
        out_file.flush()

        votes_str = f"O={openai_label} G={gemini_label} L={groq_label}"
        agree_str = "✓" if all_agree else "~"
        print(f"  {agree_str} {label} | {votes_str}")

        labeled += 1

    out_file.close()

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"Labeled: {labeled} | Skipped (resume): {skipped}")
    print(f"Saved: {OUTPUT}")

    with open(OUTPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    from collections import defaultdict
    dist = Counter(r["signal_level"] for r in rows)
    print(f"\nFinal distribution: HIGH={dist['HIGH']} LOW={dist['LOW']}")
    by_domain = defaultdict(Counter)
    for r in rows:
        by_domain[r["domain"]][r["signal_level"]] += 1
    for domain, counts in sorted(by_domain.items()):
        print(f"  {domain}: {dict(counts)}")
    all_agree_count = sum(1 for r in rows if r["all_agree"] == "True")
    print(f"All-agree: {all_agree_count}/{len(rows)} ({all_agree_count/len(rows)*100:.0f}%)")


if __name__ == "__main__":
    main()
