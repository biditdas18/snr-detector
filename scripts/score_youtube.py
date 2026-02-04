#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def run(cmd: list[str]) -> str:
    """Run command and return stdout; raise on failure with captured output."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nOutput:\n{p.stdout}")
    return p.stdout


def ensure_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing dependency: {name}\nInstall: {install_hint}")


def clean_vtt_to_text(vtt_path: Path) -> str:
    text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("WEBVTT") or "-->" in ln:
            continue
        # drop tags like "<v Speaker>" or other VTT markup
        ln = re.sub(r"<[^>]+>", "", ln).strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def classify_from_engine(clarity_z: float, depth_z: float, penalty_z: float) -> str:
    """
    Interpretable labeling tuned for educational content:
    HIGH = very clear + low noise (depth helps, but isn't mandatory)
    """
    # If it's noisy, it's not high signal.
    if penalty_z >= 0.8:
        return "LOW"

    # High-signal education: strong clarity and low noise.
    if clarity_z >= 0.9 and penalty_z <= 0.25:
        return "HIGH"

    # If clarity is weak and depth is weak, it's low-ish.
    if clarity_z <= -0.5 and depth_z <= -0.6:
        return "LOW"

    return "MID"


def reason_from_components(row: pd.Series) -> str:
    """Deterministic explanation using clarity/depth/penalty z-components."""
    parts = []

    penalty_z = float(row.get("penalty_z", 0.0))
    clarity_z = float(row.get("clarity_z", 0.0))
    depth_z = float(row.get("depth_z", 0.0))

    # Penalty first (noise)
    if penalty_z > 0.75:
        parts.append("Strong noise signals (fear/promo/recycled) detected")
    elif penalty_z > 0.25:
        parts.append("Some noise signals present (fear/promo/recycled)")
    elif penalty_z < -0.25:
        parts.append("Low noise signals (little fear/promo/recycled)")

    # Clarity next (structure/evidence)
    if clarity_z > 0.5:
        parts.append("Clear structure/evidence markers")
    elif clarity_z < -0.5:
        parts.append("Weak structure/evidence markers")

    # Depth next (richness proxy)
    if depth_z > 0.5:
        parts.append("Higher lexical/semantic richness (depth proxy)")
    elif depth_z < -0.5:
        parts.append("Lower lexical/semantic richness (depth proxy)")

    if not parts:
        return "Mixed signals; no strong clarity/depth/penalty dominance."
    return "; ".join(parts) + "."


def find_transcript_column(cols: list[str]) -> str | None:
    # Prefer common names
    preferred = ("transcript", "raw_transcript", "text", "content", "captions")
    lowered = {c.lower(): c for c in cols}
    for p in preferred:
        if p in lowered:
            return lowered[p]
    # Fallback: first column containing 'transcript'
    for c in cols:
        if "transcript" in c.lower():
            return c
    return None


def ensure_reference_anchors(
    ref_anchors_path: Path,
    gold_csv: str,
    generic_advice_bank: str,
    rebuild: bool,
) -> None:
    """
    Ensure ref anchors exist. If rebuild=True or missing, rebuild from gold set.
    """
    if (not rebuild) and ref_anchors_path.exists():
        return

    ref_anchors_path.parent.mkdir(parents=True, exist_ok=True)

    run([
        "python", "scripts/build_anchor_features.py",
        "--input", gold_csv,
        "--output", str(ref_anchors_path),
        "--generic-advice-bank", generic_advice_bank,
    ])


def main():
    ap = argparse.ArgumentParser(
        description="Score a YouTube video with SNR engine + calibration (default)."
    )
    ap.add_argument("url", help="YouTube video URL")
    ap.add_argument("--max-minutes", type=int, default=30,
                    help="Reject videos longer than this (default: 30)")
    ap.add_argument("--lang", default="en", help="Subtitle language (default: en)")
    ap.add_argument("--outdir", default="reports/youtube_demo",
                    help="Where to write intermediate outputs")
    ap.add_argument("--generic-advice-bank", default="data/generic_advice_bank.txt")
    ap.add_argument("--gold", default="data/labels/gold_labels_llm_snrC.csv")

    # Kept for backward-compatibility; calibration now uses monotonic isotonic regression.
    ap.add_argument("--cal-alpha", type=float, default=10.0,
                    help="(Deprecated) Kept for backward compatibility; no longer used.")

    ap.add_argument("--engine-only", action="store_true",
                    help="Print raw engine score only (NOT on 1–5 rubric scale).")

    # Reference anchor distribution for z-scoring single videos
    ap.add_argument("--ref-anchors", default="reports/anchor_features_v2.csv",
                    help="Reference anchor_features CSV used for z-scoring single videos (default: reports/anchor_features_v2.csv)")
    ap.add_argument("--no-cache-ref", action="store_true",
                    help="Rebuild reference anchors from --gold each run (slower). Default uses cached --ref-anchors if it exists.")

    args = ap.parse_args()

    ensure_tool("yt-dlp", "pip install yt-dlp  (or: brew install yt-dlp)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Get metadata (duration, id, title)
    info_out = run(["yt-dlp", "--dump-json", "--no-warnings", "-q", args.url])

    # Sometimes logs still leak; recover by taking the last JSON object in output
    lines = [ln.strip() for ln in info_out.splitlines() if ln.strip()]
    json_line = None
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            json_line = ln
            break

    if json_line is None:
        raise RuntimeError(f"yt-dlp did not return JSON. Output was:\n{info_out[:2000]}")

    info = json.loads(json_line)

    vid = info.get("id", "unknown_id")
    title = info.get("title", "unknown_title")
    duration = int(info.get("duration") or 0)  # seconds

    if duration and duration > args.max_minutes * 60:
        raise SystemExit(
            f"Video too long: {duration/60:.1f} minutes. "
            f"Limit is {args.max_minutes} minutes (prototype reliability constraint)."
        )

    # 2) Download auto-subs as VTT
    vtt_tmpl = str(outdir / f"{vid}.%(ext)s")
    run([
        "yt-dlp",
        "-q", "--no-warnings",
        "--skip-download",
        "--write-auto-sub",
        "--sub-lang", args.lang,
        "--sub-format", "vtt",
        "-o", vtt_tmpl,
        args.url
    ])

    vtts = sorted(outdir.glob(f"{vid}*.vtt"))
    if not vtts:
        raise SystemExit("No .vtt subtitles found. Video may not have captions/transcript available.")
    vtt_path = vtts[-1]

    # 3) Clean + save transcript text
    transcript_text = clean_vtt_to_text(vtt_path)

    txt_path = outdir / f"{vid}.txt"
    txt_path.write_text(transcript_text, encoding="utf-8")

    wc = len(transcript_text.split())
    print(f"Transcript saved: {wc} words → {txt_path}")
    if wc < 250:
        raise SystemExit(
            "Transcript too short for reliable scoring. "
            "Try another video with better captions."
        )

    # 4) Build a one-row CSV matching gold schema
    gold = pd.read_csv(args.gold)
    cols = list(gold.columns)

    transcript_col = find_transcript_column(cols)
    if transcript_col is None:
        raise SystemExit(
            "Could not find a transcript column in the gold label schema.\n"
            "Expected a column named like: transcript / raw_transcript / ...\n"
            "Fix: rename/add a transcript column in data/labels/gold_labels_llm_snrC.csv."
        )

    row = {c: "" for c in cols}
    if "video_id" in row:
        row["video_id"] = vid
    if "title" in row:
        row["title"] = title
    row[transcript_col] = transcript_text

    one_csv = outdir / "one_video.csv"
    pd.DataFrame([row]).to_csv(one_csv, index=False)

    # 5) Build anchor features for this one video
    anchors_csv = outdir / "anchor_features_one_video.csv"
    run([
        "python", "scripts/build_anchor_features.py",
        "--input", str(one_csv),
        "--output", str(anchors_csv),
        "--generic-advice-bank", args.generic_advice_bank,
    ])

    # 6) Ensure reference anchors exist (for single-video z-scoring)
    ref_anchors = Path(args.ref_anchors)
    ensure_reference_anchors(
        ref_anchors_path=ref_anchors,
        gold_csv=args.gold,
        generic_advice_bank=args.generic_advice_bank,
        rebuild=args.no_cache_ref,
    )

    # 7) Run engine on the anchors (IMPORTANT: use --ref_csv)
    engine_csv = outdir / "snr_engine_one_video.csv"
    run([
        "python", "scripts/compute_snr_engine.py",
        "--input", str(anchors_csv),
        "--output", str(engine_csv),
        "--ref_csv", str(ref_anchors),
    ])

    eng = pd.read_csv(engine_csv)
    if eng.empty:
        raise SystemExit("Engine output is empty; something went wrong in feature extraction.")
    r = eng.iloc[0]

    raw1 = float(r.get("snr_engine_score", 0.0))

    # Always show interpretable components
    print("\n=== SNR-Detector Demo ===")
    print(f"Video : {title}")
    print(f"ID    : {vid}")
    if duration:
        print(f"Length: {duration/60:.1f} minutes")
    print("")
    print(
        f"Engine components: "
        f"clarity_z={float(r.get('clarity_z', 0.0)):.3f}, "
        f"depth_z={float(r.get('depth_z', 0.0)):.3f}, "
        f"penalty_z={float(r.get('penalty_z', 0.0)):.3f}"
    )
    print(f"Raw engine score (internal): {raw1:.3f}")

    if args.engine_only:
        print("\nSNR (ENGINE-ONLY): internal score (not 1–5 rubric scale)")
        print("Reason:", reason_from_components(r))
        print(f"\nArtifacts written under: {outdir}")
        return

    # 8) Calibrate to 1–5 rubric scale using gold seed set (default demo behavior)
    #    Calibration is now monotonic: snr_engine_score -> snr_score (1–5)
    gold_anchor_out = outdir / "anchor_features_gold.csv"
    gold_engine_out = outdir / "snr_engine_gold.csv"

    run([
        "python", "scripts/build_anchor_features.py",
        "--input", args.gold,
        "--output", str(gold_anchor_out),
        "--generic-advice-bank", args.generic_advice_bank,
    ])
    run([
        "python", "scripts/compute_snr_engine.py",
        "--input", str(gold_anchor_out),
        "--output", str(gold_engine_out),
        "--ref_csv", str(ref_anchors),
    ])

    g = pd.read_csv(gold_engine_out)
    if len(g) < 60:
        print(
            f"\nNOTE: Calibration set is small (n={len(g)}). "
            "The 1–5 score may shrink toward MID; trust the engine-rule label more."
        )

    needed = ["snr_engine_score", "snr_score"]
    missing = [c for c in needed if c not in g.columns]
    if missing:
        raise SystemExit(f"Gold engine CSV missing required columns: {missing}")

    # Percentile of this video vs gold seed raw engine scores (intuitive, no extra labeling needed)
    raw_seed = g["snr_engine_score"].astype(float).to_numpy()
    pct = float((raw_seed <= raw1).mean() * 100.0)
    
    # Avoid awkward 100.0th percentile display
    pct_print = min(99.9, pct) if pct >= 100.0 else pct
    
    # Human-friendly display score derived from percentile (monotonic, transparent)
    display_1_5 = 1.0 + 4.0 * (pct_print / 100.0)
    display_1_5 = max(1.0, min(5.0, display_1_5))


    # Monotonic calibration: raw engine score -> 1–5 rubric score
    from sklearn.isotonic import IsotonicRegression

    xg = g["snr_engine_score"].astype(float).to_numpy()
    yg = g["snr_score"].astype(float).to_numpy()

    if len(np.unique(xg)) < 3:
        # Fallback: rank-based monotonic mapping (rare, but safe)
        order = np.argsort(xg)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(1.0, 5.0, num=len(xg))
        idx = int(np.argmin(np.abs(xg - raw1)))
        pred = float(ranks[idx])
    else:
        iso = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds="clip")
        iso.fit(xg, yg)
        pred = float(iso.predict([raw1])[0])

    # Explicit shrink toward mid for small calibration sets (reduces misleading “precision”)
    n_cal = len(g)
    shrink = min(1.0, n_cal / 80.0)  # at n=23 → ~0.29
    pred = 3.0 + shrink * (pred - 3.0)

    # Safety clamp
    pred = max(1.0, min(5.0, pred))

    cz = float(r.get("clarity_z", 0.0))
    dz = float(r.get("depth_z", 0.0))
    pz = float(r.get("penalty_z", 0.0))

    label = classify_from_engine(cz, dz, pz)
    reason = reason_from_components(r) + f" (Percentile relative to seed set; n={len(g)}.)"

    print(f"\nSNR label (engine rule): {label}")
    print(f"Score: {display_1_5:.1f}/5  |  Percentile vs seed set: {pct_print:.1f}th")
    print(f"Reason: {reason}")


    print(f"\nArtifacts written under: {outdir}")


if __name__ == "__main__":
    main()