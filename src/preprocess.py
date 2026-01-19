import subprocess, tempfile
from pathlib import Path
import pandas as pd

def featurize_transcript_row(
    video_id: str,
    title: str,
    transcript: str,
    generic_advice_bank_path: Path,
    build_script_path: Path,
) -> pd.DataFrame:
    """
    Builds anchor features for ONE transcript by invoking scripts/build_anchor_features.py
    on a temporary CSV (so we reuse your current pipeline exactly).
    Returns a single-row DataFrame with the generated feature columns.
    """
    if not generic_advice_bank_path.exists():
        raise FileNotFoundError(f"Missing: {generic_advice_bank_path}")
    if not build_script_path.exists():
        raise FileNotFoundError(f"Missing: {build_script_path}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / "eval_1.csv"
        out = td / "eval_1_features.csv"

        # match your pipeline input schema
        pd.DataFrame([{
            "video_id": video_id,
            "title": title,
            "transcript": transcript,

            # dummy placeholders required by build_anchor_features.py
            "signal_level": None,
            "noise_superclass": None,
            "noise_subtype": None,
            "primary_topic": None,
            "takeaway_clarity_1_5": None,
            "insight_depth_1_5": None,
            "snr_score": None,
        }]).to_csv(inp, index=False)

        cmd = [
            "python", str(build_script_path),
            "--input", str(inp),
            "--output", str(out),
            "--generic-advice-bank", str(generic_advice_bank_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                "Feature build failed.\n"
                f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
            )

        df = pd.read_csv(out)
        if len(df) != 1:
            raise RuntimeError(f"Expected 1 row of features, got {len(df)}")
        # Drop any label columns if present
        for c in ["snr_score"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        return df