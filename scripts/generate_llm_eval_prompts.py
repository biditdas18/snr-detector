#!/usr/bin/env python3
"""
Generates 3 evaluation prompt files — one per domain.
Each file is ready to paste directly into ChatGPT, Gemini, or Meta AI.
All 3 LLMs receive identical prompts.

Output files:
  prompts/eval_career_selfimprovement.txt
  prompts/eval_tech_ai.txt
  prompts/eval_general_education.txt
"""

import csv
import os

BASE = "/Users/biditdas/Desktop/snr-submission/snr-detector"
INPUT = os.path.join(BASE, "data/labels/review_queue.csv")
OUTPUT_DIR = os.path.join(BASE, "prompts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WORDS_PER_TRANSCRIPT = 500  # truncate to keep prompt manageable

# ── CALIBRATED RUBRICS ────────────────────────────────────────

RUBRICS = {
    "career_selfimprovement": {
        "context": "Career advice, job market, productivity, self-improvement, immigration career advice",
        "HIGH": [
            "Contains at least 2 specific procedural steps with named methods, tools, timelines, or defined techniques — conceptual definitions alone do not qualify but named practices or described processes do",
            "Provides concrete outcomes or metrics — numbers, percentages, named frameworks, specific results",
            "If addressing a problem or risk, provides a specific action plan or named solution, not just awareness",
            "Ideas build sequentially — each point adds new information rather than restating the previous one"
        ],
        "LOW": [
            "Motivational language dominates the transcript with no actionable method attached — phrases like 'believe in yourself' or 'stay consistent' appear repeatedly and procedural steps are absent or trivial",
            "Repeats the same warning, risk, or point more than twice without adding new information",
            "Primary hook is fear or urgency — job loss, visa denial, market collapse — with no concrete mitigation steps offered",
            "More than 30% of content promotes a course, consultation service, community, or product",
            "Claims are authoritative but cite no evidence, data, named source, or personal experiment — named books, frameworks, cited statistics, or described personal tests do NOT trigger this criterion"
        ],
        "conflict": "If HIGH-signal content coexists with dominant LOW patterns (>30% of transcript), assign LOW. Isolated motivational framing around otherwise procedural content does NOT constitute a dominant LOW pattern."
    },
    "tech_ai": {
        "context": "Software engineering, AI/ML, cloud computing, system design, coding tutorials",
        "HIGH": [
            "Names specific technologies, algorithms, data structures, or architectural patterns with accurate technical detail (e.g. stack-based O(n) solution, transformer attention mechanism, M5 neural engine)",
            "Explains mechanism or tradeoff — not just what but why, with technical reasoning",
            "Code concepts, benchmarks, or system design decisions are concrete and implementable",
            "Covers a technical problem with a specific solution path, not just awareness of the problem"
        ],
        "LOW": [
            "Uses hype language without technical substance — 'AI will change everything', 'this is revolutionary', 'you must learn this now' with no technical explanation",
            "Claims about AI capabilities or job market are sweeping and unsubstantiated by data or named research",
            "Primary message is fear — job loss from AI, being left behind — with no actionable technical skill offered",
            "Recommends learning a technology or skill without any explanation of what it does, how it works, or why it matters technically",
            "More than 30% of content promotes a course, bootcamp, newsletter, or paid resource; a brief sponsor mention alone does NOT qualify"
        ],
        "conflict": "If technical content is accurate but framing is primarily fear-based or promotional, assign LOW."
    },
    "general_education": {
        "context": "Science, history, economics, mathematics, philosophy, geopolitics — explanatory content",
        "HIGH": [
            "Explains a mechanism, phenomenon, or concept from first principles with logical steps",
            "Uses a specific analogy, thought experiment, or concrete example that illuminates the concept accurately",
            "Arrives at a non-obvious insight or counterintuitive conclusion supported by the explanation",
            "Accurately represents complexity — does not oversimplify to the point of inaccuracy",
            "If covering a problem or risk, provides historical context, causal analysis, or named expert perspectives"
        ],
        "LOW": [
            "Uses alarming or conspiratorial framing as the primary hook without evidence — 'what they don't want you to know', 'the system is broken'",
            "Makes confident causal claims without evidence, named sources, or logical reasoning",
            "Presents complex multi-causal phenomena as simple good-vs-evil narratives",
            "Emotional manipulation — outrage, fear, tribalism — dominates over explanation"
        ],
        "conflict": "IMPORTANT: Absence of actionable steps does NOT indicate LOW signal for general education. Judge on conceptual accuracy and explanatory depth only."
    }
}


def truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [transcript truncated]"


def build_prompt(domain: str, rubric: dict, transcripts: list) -> str:
    high_list = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(rubric["HIGH"])
    )
    low_list = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(rubric["LOW"])
    )

    transcript_block = ""
    for i, t in enumerate(transcripts):
        transcript_block += f"""
---
TRANSCRIPT {i+1}:
{truncate(t, WORDS_PER_TRANSCRIPT)}
"""

    return f"""You are an expert educational content quality evaluator.
Your task: evaluate {len(transcripts)} YouTube video transcripts from the
{domain.replace('_', ' ').upper()} domain and assign each a quality label.

================================================================
DOMAIN: {rubric['context']}
================================================================

HIGH SIGNAL criteria — content must satisfy AT LEAST 2:
{high_list}

LOW SIGNAL criteria — ANY ONE dominant pattern = LOW:
{low_list}

CONFLICT RULE: {rubric['conflict']}

================================================================
INSTRUCTIONS
================================================================

1. Read each transcript carefully
2. Apply the rubric above — do not use your own judgment about
   quality, only apply these specific criteria
3. For each transcript output EXACTLY this format:

TRANSCRIPT [N]: [HIGH or LOW]
REASON: [one sentence citing the specific criterion that determined your label]
CRITERIA MET: [comma-separated list of HIGH criteria numbers met, or LOW criterion number triggered]

4. After all transcripts, output a summary table:

SUMMARY:
| # | Label | Key Criterion |
|---|-------|---------------|
| 1 | HIGH  | H2: concrete metrics provided |
| 2 | LOW   | L1: motivational language dominates |
...

DO NOT skip any transcript. DO NOT add commentary outside the format.
Label every transcript as either HIGH or LOW — no other labels.

================================================================
TRANSCRIPTS TO EVALUATE ({len(transcripts)} total)
================================================================
{transcript_block}
================================================================
BEGIN EVALUATION NOW
================================================================"""


def main():
    # Load transcripts by domain
    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    domain_transcripts = {}
    for row in rows:
        d = row["domain"]
        if d not in domain_transcripts:
            domain_transcripts[d] = []
        domain_transcripts[d].append(row["transcript"])

    print(f"Loaded {len(rows)} transcripts across {len(domain_transcripts)} domains")

    # Generate one prompt file per domain
    for domain, transcripts in sorted(domain_transcripts.items()):
        rubric = RUBRICS.get(domain)
        if not rubric:
            print(f"  No rubric for {domain} — skipping")
            continue

        prompt = build_prompt(domain, rubric, transcripts)

        output_path = os.path.join(OUTPUT_DIR, f"eval_{domain}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        word_count = len(prompt.split())
        print(f"  {domain}: {len(transcripts)} transcripts | "
              f"{word_count:,} words | saved to prompts/eval_{domain}.txt")

    print(f"\n{'='*60}")
    print("PROMPT FILES GENERATED")
    print(f"{'='*60}")
    print(f"\nThree files in: {OUTPUT_DIR}/")
    print(f"  eval_career_selfimprovement.txt")
    print(f"  eval_tech_ai.txt")
    print(f"  eval_general_education.txt")
    print(f"\nFor each LLM (ChatGPT, Gemini, Meta AI):")
    print(f"  1. Open the LLM")
    print(f"  2. Paste eval_career_selfimprovement.txt -> get response -> save")
    print(f"  3. Paste eval_tech_ai.txt -> get response -> save")
    print(f"  4. Paste eval_general_education.txt -> get response -> save")
    print(f"  Total: 9 paste operations (3 LLMs x 3 domains)")
    print(f"\nSave each response as:")
    print(f"  chatgpt_career.txt, chatgpt_tech.txt, chatgpt_education.txt")
    print(f"  gemini_career.txt, gemini_tech.txt, gemini_education.txt")
    print(f"  metaai_career.txt, metaai_tech.txt, metaai_education.txt")
    print(f"\nUpload all 9 files to Claude to compute pairwise agreement.")


if __name__ == "__main__":
    main()
