"""
Generate 300 synthetic transcripts for SNR binary classifier training.
Labels assigned by prompt construction: HIGH prompt → HIGH label, LOW prompt → LOW label.
"""

import anthropic
import csv
import os
import time
import random

OUTPUT_PATH = "data/synthetic/synthetic_transcripts.csv"
FIELDNAMES = ["domain", "transcript", "signal_level", "topic", "batch_id"]

CAREER_HIGH_TOPICS = [
    "career transitions", "salary negotiation", "networking strategies",
    "productivity systems", "job interview preparation", "LinkedIn optimization",
    "skill development roadmaps", "remote work effectiveness",
    "performance reviews", "building professional reputation",
]

CAREER_LOW_TOPICS = [
    "career anxiety", "job market fears", "hustle culture motivation",
    "AI replacing jobs", "competition in the workplace", "success mindset",
    "why most people fail", "the importance of hard work",
    "fear of failure", "staying motivated when behind",
]

TECH_HIGH_TOPICS = [
    "how transformers work", "vector databases explained", "RAG architecture",
    "fine-tuning vs prompting tradeoffs", "AWS service deep dives",
    "system design for ML pipelines", "LLM evaluation methods",
    "attention mechanisms", "embedding models comparison",
    "building production ML systems",
]

TECH_LOW_TOPICS = [
    "AI replacing software engineers", "the end of coding jobs",
    "ChatGPT taking over industries", "why you must learn AI now",
    "tech layoffs and AI automation", "the AI bubble", "future of work anxiety",
    "automation destroying careers", "why programmers are obsolete",
    "surviving the AI revolution",
]

GENERAL_HIGH_TOPICS = [
    "how black holes form", "the prisoner's dilemma explained",
    "why inflation happens", "Godel's incompleteness theorems",
    "how vaccines train the immune system", "the tragedy of the commons",
    "Bayesian reasoning in everyday life", "how compilers work",
    "the history of the internet protocol", "quantum entanglement basics",
]

GENERAL_LOW_TOPICS = [
    "economic collapse warnings", "government conspiracy framing",
    "why the financial system is broken", "hidden truths about history",
    "what they don't want you to know about science",
    "society is collapsing and here is why", "mainstream media lies",
    "the hidden agenda behind global events", "why experts can't be trusted",
    "the truth they suppress",
]


def build_prompt(domain: str, signal: str, topic: str) -> str:
    if domain == "career_selfimprovement" and signal == "HIGH":
        return f"""Write a 400-500 word educational YouTube video transcript about "{topic}" that demonstrates HIGH signal quality.

Requirements:
- Present at least 3 concrete, specific actionable steps
- Include at least one specific example, statistic, or named framework (e.g. a real study, a named technique, a specific tool)
- Ideas must build on each other logically — not random tips
- Conversational YouTube tone but substantive throughout
- No course promotion, no channel plugs, no fear framing
- End with a clear takeaway the viewer can apply today

Write only the transcript text. No titles, no labels, no meta-commentary."""

    if domain == "career_selfimprovement" and signal == "LOW":
        return f"""Write a 400-500 word educational YouTube video transcript about "{topic}" that demonstrates LOW signal quality.

Requirements:
- Use vague motivational language with no specifics (e.g. 'believe in yourself', 'work harder than everyone else')
- Repeat the same core message 3-4 times in different words
- Include at least one fear-based hook about job loss, competition, or falling behind — with no concrete solution offered
- Optional: include a soft course or community promotion
- Conversational YouTube tone
- Viewer should feel warned but have nothing actionable to do

Write only the transcript text. No titles, no labels, no meta-commentary."""

    if domain == "tech_ai" and signal == "HIGH":
        return f"""Write a 400-500 word educational YouTube video transcript about "{topic}" that demonstrates HIGH signal quality.

Requirements:
- Explain a specific technical concept or system clearly
- Use at least one concrete example, code concept, benchmark, or real-world application
- Structure must follow: problem → explanation → application
- Mention specific tools, libraries, papers, or companies by name
- No hype language, no 'AI will change everything' vagueness
- Viewer should understand something specific they didn't before

Write only the transcript text. No titles, no labels, no meta-commentary."""

    if domain == "tech_ai" and signal == "LOW":
        return f"""Write a 400-500 word educational YouTube video transcript about "{topic}" that demonstrates LOW signal quality.

Requirements:
- Lead with fear or hype: AI is taking over, jobs are disappearing, you will be left behind if you don't act now
- Make broad sweeping claims without citing specifics or evidence
- Offer no concrete technical explanation or actionable skill
- Include vague advice like 'learn AI' or 'adapt or die'
- Optional: funnel toward a course, bootcamp, or newsletter
- Viewer feels anxious but has no clearer understanding of the tech

Write only the transcript text. No titles, no labels, no meta-commentary."""

    if domain == "general_education" and signal == "HIGH":
        return f"""Write a 400-500 word educational YouTube video transcript about "{topic}" that demonstrates HIGH signal quality.

Requirements:
- Explain a concept from first principles with clear logical steps
- Use at least one analogy, thought experiment, or concrete example
- Build toward a non-obvious insight or counterintuitive conclusion
- Reference real historical events, scientific findings, or named theories where appropriate
- Viewer should understand the concept well enough to explain it
- No fear framing, no political agenda, no manufactured urgency

Write only the transcript text. No titles, no labels, no meta-commentary."""

    if domain == "general_education" and signal == "LOW":
        return f"""Write a 400-500 word educational YouTube video transcript about "{topic}" that demonstrates LOW signal quality.

Requirements:
- Use alarming or conspiratorial framing as the primary hook
- Make confident claims without evidence or cited sources
- Present complex issues as simple good vs evil narratives
- Provide no actionable insight or genuine conceptual explanation
- Leave the viewer more anxious or confused than informed
- Optional: appeal to authority without naming the authority

Write only the transcript text. No titles, no labels, no meta-commentary."""

    raise ValueError(f"Unknown domain/signal combo: {domain}/{signal}")


def load_existing(path: str) -> set:
    """Return set of (domain, signal_level, topic) already generated."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {(r["domain"], r["signal_level"], r["topic"]) for r in reader}


def build_work_queue() -> list[dict]:
    """Build full 300-item list, alternating domain+signal within each batch."""
    domains_config = [
        ("career_selfimprovement", CAREER_HIGH_TOPICS, CAREER_LOW_TOPICS),
        ("tech_ai", TECH_HIGH_TOPICS, TECH_LOW_TOPICS),
        ("general_education", GENERAL_HIGH_TOPICS, GENERAL_LOW_TOPICS),
    ]

    # Each domain needs 50 HIGH and 50 LOW.
    # Topics list has 10 entries → cycle 5 times each, with slight variation via index suffix.
    queue = []
    batch_id = 1

    for domain, high_topics, low_topics in domains_config:
        entries = []
        for i in range(50):
            topic = high_topics[i % len(high_topics)]
            # Add a numeric suffix so duplicate topics vary slightly in the prompt
            if i >= len(high_topics):
                topic = f"{topic} (part {i // len(high_topics) + 1})"
            entries.append({"domain": domain, "signal_level": "HIGH", "topic": topic})
        for i in range(50):
            topic = low_topics[i % len(low_topics)]
            if i >= len(low_topics):
                topic = f"{topic} (part {i // len(low_topics) + 1})"
            entries.append({"domain": domain, "signal_level": "LOW", "topic": topic})

        # Interleave HIGH and LOW, then chunk into batches of ~50
        high_e = [e for e in entries if e["signal_level"] == "HIGH"]
        low_e = [e for e in entries if e["signal_level"] == "LOW"]
        interleaved = [x for pair in zip(high_e, low_e) for x in pair]

        for i, item in enumerate(interleaved):
            item["batch_id"] = batch_id + (i // 50)
        queue.extend(interleaved)
        batch_id += (len(interleaved) // 50)

    return queue


def main():
    client = anthropic.Anthropic()

    os.makedirs("data/synthetic", exist_ok=True)
    existing = load_existing(OUTPUT_PATH)
    print(f"Already generated: {len(existing)} transcripts")

    queue = build_work_queue()
    todo = [
        item for item in queue
        if (item["domain"], item["signal_level"], item["topic"]) not in existing
    ]
    print(f"Remaining to generate: {len(todo)}")

    file_exists = os.path.exists(OUTPUT_PATH)
    skipped = 0
    generated = 0

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for idx, item in enumerate(todo, 1):
            prompt = build_prompt(item["domain"], item["signal_level"], item["topic"])
            print(f"[{idx}/{len(todo)}] {item['domain']} | {item['signal_level']} | {item['topic'][:50]}", end=" ... ", flush=True)

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                transcript = response.content[0].text.strip()
                writer.writerow({
                    "domain": item["domain"],
                    "transcript": transcript,
                    "signal_level": item["signal_level"],
                    "topic": item["topic"],
                    "batch_id": item["batch_id"],
                })
                f.flush()
                generated += 1
                print("OK")
            except Exception as e:
                skipped += 1
                print(f"FAILED: {e}")

            # Small sleep to avoid rate limits
            time.sleep(0.3)

    print(f"\nDone. Generated: {generated}, Skipped: {skipped}")
    print(f"Output: {OUTPUT_PATH}")

    # Final count check
    with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
        total = sum(1 for _ in csv.DictReader(f))
    print(f"Total rows in CSV: {total}")


if __name__ == "__main__":
    main()
