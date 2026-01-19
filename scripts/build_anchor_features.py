#!/usr/bin/env python3
"""
Build anchor_features.csv from your locked gold dataset.

Input:  gold_labels_llm_snrC.csv (must include: transcript + labels)
Output: anchor_features.csv (features + labels)

Usage:
  python scripts/build_anchor_features.py \
    --input /path/to/gold_labels_llm_snrC.csv \
    --output /path/to/anchor_features.csv
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

# deterministic text similarity (recycled-signal penalty)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Lightweight text utilities
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def load_generic_advice_bank(path: str) -> list[str]:
    """Loads generic advice lines (one per line). Empty lines are ignored."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def build_recycled_similarity_fn(generic_bank_path: str):
    """Returns a deterministic function: transcript -> max cosine similarity vs generic advice bank."""
    bank = load_generic_advice_bank(generic_bank_path)
    if not bank:
        raise ValueError(f"Generic advice bank is empty: {generic_bank_path}")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1
    )
    bank_mat = tfidf.fit_transform(bank)

    def recycled_similarity(transcript_text: str) -> float:
        t = normalize_text(transcript_text)
        if not t:
            return 0.0
        vec = tfidf.transform([t])
        sims = cosine_similarity(vec, bank_mat)
        return float(sims.max())

    return recycled_similarity


def normalize_text(t: str) -> str:
    t = "" if t is None else str(t)
    t = t.replace("\u00a0", " ")
    t = _WS_RE.sub(" ", t).strip()
    return t

def split_sentences(t: str) -> List[str]:
    t = normalize_text(t)
    if not t:
        return []
    # keep it simple + deterministic
    sents = _SENT_SPLIT_RE.split(t)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def tokenize(t: str) -> List[str]:
    t = normalize_text(t).lower()
    return _WORD_RE.findall(t)

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def shannon_entropy(tokens: List[str]) -> float:
    """Higher = more diverse; lower = repetitive."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    n = len(tokens)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log(p + 1e-12, 2)
    return ent

def top_ngram_repetition(tokens: List[str], n: int = 3) -> float:
    """Returns repetition ratio for top n-gram: max_count / total_ngrams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    top = counts.most_common(1)[0][1]
    return safe_div(top, total)

def count_phrases(text_lc: str, phrases: List[str]) -> int:
    """Counts phrase occurrences (substring matches) deterministically."""
    return sum(text_lc.count(p) for p in phrases)

def count_regex(text_lc: str, pattern: str) -> int:
    return len(re.findall(pattern, text_lc))


# ----------------------------
# Lexicons (small + editable)
# ----------------------------
# These are intentionally short. We will evolve them later as part of drift control,
# without scraping new data.
FEAR_PHRASES = [
    "destroy", "devastat", "panic", "crisis", "catastroph", "nightmare",
    "you are doomed", "ruin", "collapse", "mass layoff", "you will be replaced",
    "no hope", "game over", "urgent", "warning", "worst", "terrifying"
]

PROMO_PHRASES = [
    "sponsor", "sponsored", "link in the description", "use my code", "discount",
    "free trial", "sign up", "buy", "limited time", "offer", "join my course",
    "newsletter", "patreon", "membership", "affiliate"
]

HYPE_PHRASES = [
    "10x", "game changer", "revolutionary", "break the internet", "insane",
    "nobody is talking about", "secret", "exposed", "shocking", "you won't believe",
    "this changes everything"
]

GENERIC_ADVICE_PHRASES = [
    "just be consistent", "work hard", "believe in yourself", "never give up",
    "stay motivated", "you got this", "trust the process", "grind"
]

EVIDENCE_MARKERS = [
    "because", "therefore", "however", "for example", "evidence", "data",
    "study", "research", "result", "numbers", "benchmark", "experiment"
]

STRUCTURE_MARKERS = [
    "first", "second", "third", "step", "steps", "in summary", "to summarize",
    "takeaway", "key point", "here's why", "let's break down"
]


@dataclass
class Features:
    # core stats
    word_count: int
    sent_count: int
    avg_sent_len: float
    unique_ratio: float
    entropy: float
    # repetition
    top_trigram_rep: float
    # recycled-signal penalty
    recycled_signal_similarity: float  # max TF-IDF cosine similarity vs generic advice bank
    recycled_penalty: float  # gated penalty: similarity * (1 - explainer_strength)
    # lexical cues
    fear_hits: int
    promo_hits: int
    hype_hits: int
    generic_advice_hits: int
    evidence_hits: int
    structure_hits: int
    # rhetorical / persuasion proxies
    qmark_count: int
    exclam_count: int
    you_count: int
    i_count: int
    modal_count: int  # should, must, need, have to
    imperative_count: int  # "do X", "stop", "avoid" (very rough)
    # density-ish
    content_density_proxy: float  # evidence + structure per 100 words


def extract_features(transcript: str, recycled_sim_fn=None) -> Features:
    text = normalize_text(transcript)
    text_lc = text.lower()
    tokens = tokenize(text)
    sents = split_sentences(text)

    wc = len(tokens)
    sc = len(sents)
    avg_sl = safe_div(wc, sc)

    uniq_ratio = safe_div(len(set(tokens)), wc)
    ent = shannon_entropy(tokens)
    tri_rep = top_ngram_repetition(tokens, n=3)

    recycled_sim = 0.0
    if recycled_sim_fn is not None:
        recycled_sim = float(recycled_sim_fn(text))


    fear = count_phrases(text_lc, FEAR_PHRASES)
    promo = count_phrases(text_lc, PROMO_PHRASES)
    hype = count_phrases(text_lc, HYPE_PHRASES)
    gen = count_phrases(text_lc, GENERIC_ADVICE_PHRASES)
    evid = count_phrases(text_lc, EVIDENCE_MARKERS)
    struct = count_phrases(text_lc, STRUCTURE_MARKERS)
    
    # ---- gated recycled-signal penalty ----
    explainer_strength = (evid + struct) / 30.0
    explainer_strength = max(0.0, min(1.0, explainer_strength))
    recycled_penalty = recycled_sim * (1.0 - explainer_strength)

    qmarks = text.count("?")
    excls = text.count("!")

    # simple token counts for pronouns
    you_c = sum(1 for t in tokens if t in ("you", "your", "you're", "youre"))
    i_c = sum(1 for t in tokens if t in ("i", "me", "my", "i'm", "im", "mine"))

    modals = sum(1 for t in tokens if t in ("should", "must", "need", "needs", "have", "to"))
    # very rough imperative proxy: count lines starting with verbs is hard without NLP
    # Instead: count occurrences of "do ", "stop ", "avoid ", "try ", "remember "
    imperative = count_regex(text_lc, r"\b(do|stop|avoid|try|remember|start)\b")

    density = safe_div((evid + struct), max(wc, 1)) * 100.0  # per 100 words

    return Features(
        word_count=wc,
        sent_count=sc,
        avg_sent_len=avg_sl,
        unique_ratio=uniq_ratio,
        entropy=ent,
        top_trigram_rep=tri_rep,
        recycled_signal_similarity=recycled_sim,
        recycled_penalty=recycled_penalty,
        fear_hits=fear,
        promo_hits=promo,
        hype_hits=hype,
        generic_advice_hits=gen,
        evidence_hits=evid,
        structure_hits=struct,
        qmark_count=qmarks,
        exclam_count=excls,
        you_count=you_c,
        i_count=i_c,
        modal_count=modals,
        imperative_count=imperative,
        content_density_proxy=density,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to gold_labels_llm_snrC.csv")
    ap.add_argument("--output", required=True, help="Path to write anchor_features.csv")
    ap.add_argument("--generic-advice-bank", default="data/generic_advice_bank.txt",required=True ,help="Path to generic advice bank (one line per item) for recycled-signal similarity")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Build recycled-signal similarity function (deterministic TF-IDF cosine similarity)
    recycled_sim_fn = None
    try:
        recycled_sim_fn = build_recycled_similarity_fn(args.generic_advice_bank)
        print(f"Loaded generic advice bank: {args.generic_advice_bank}")
    except FileNotFoundError:
        print(f"WARN: generic advice bank not found at {args.generic_advice_bank}. Setting recycled_signal_similarity=0 for all rows.")
    except Exception as e:
        print(f"WARN: could not initialize recycled similarity feature ({e}). Setting recycled_signal_similarity=0 for all rows.")


    required = [
        "video_id", "title", "transcript",
        "signal_level", "noise_superclass", "noise_subtype", "primary_topic",
        "takeaway_clarity_1_5", "insight_depth_1_5", "snr_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        feats = extract_features(row["transcript"], recycled_sim_fn=recycled_sim_fn)
        out = {
            "video_id": row["video_id"],
            "title": row["title"],
            "word_count": feats.word_count,
            "sent_count": feats.sent_count,
            "avg_sent_len": feats.avg_sent_len,
            "unique_ratio": feats.unique_ratio,
            "entropy": feats.entropy,
            "top_trigram_rep": feats.top_trigram_rep,
            "recycled_signal_similarity": feats.recycled_signal_similarity,
            "recycled_penalty": feats.recycled_penalty,
            "fear_hits": feats.fear_hits,
            "promo_hits": feats.promo_hits,
            "hype_hits": feats.hype_hits,
            "generic_advice_hits": feats.generic_advice_hits,
            "evidence_hits": feats.evidence_hits,
            "structure_hits": feats.structure_hits,
            "qmark_count": feats.qmark_count,
            "exclam_count": feats.exclam_count,
            "you_count": feats.you_count,
            "i_count": feats.i_count,
            "modal_count": feats.modal_count,
            "imperative_count": feats.imperative_count,
            "content_density_proxy": feats.content_density_proxy,
            # labels
            "signal_level": row["signal_level"],
            "noise_superclass": row["noise_superclass"],
            "noise_subtype": row["noise_subtype"],
            "primary_topic": row["primary_topic"],
            "takeaway_clarity_1_5": row["takeaway_clarity_1_5"],
            "insight_depth_1_5": row["insight_depth_1_5"],
            "snr_score": row["snr_score"],
            "short_reasoning": row.get("short_reasoning", ""),
            "notes": row.get("notes", ""),
        }
        feature_rows.append(out)

    out_df = pd.DataFrame(feature_rows)

    # deterministic column ordering (nice for diffing / paper reproducibility)
    col_order = [
        "video_id", "title",
        "word_count", "sent_count", "avg_sent_len", "unique_ratio", "entropy",
        "top_trigram_rep",
        "recycled_signal_similarity",
        "recycled_penalty",
        "fear_hits", "promo_hits", "hype_hits", "generic_advice_hits",
        "evidence_hits", "structure_hits",
        "qmark_count", "exclam_count", "you_count", "i_count", "modal_count",
        "imperative_count", "content_density_proxy",
        "signal_level", "noise_superclass", "noise_subtype", "primary_topic",
        "takeaway_clarity_1_5", "insight_depth_1_5", "snr_score",
        "short_reasoning", "notes",
    ]
    out_df = out_df[col_order]

    out_df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}  (rows={len(out_df)}, cols={len(out_df.columns)})")


if __name__ == "__main__":
    main()