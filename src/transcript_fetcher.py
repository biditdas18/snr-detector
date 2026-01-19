import csv
import logging
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import yt_dlp


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[1]
LINKS_FILE = BASE_DIR / "data" / "raw" / "video_links.txt"
OUTPUT_CSV = BASE_DIR / "data" / "transcripts" / "transcripts.csv"
ERROR_LOG = BASE_DIR / "logs" / "transcript_errors.log"

# ---------- Logging ----------
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------- Global API instance ----------
api = YouTubeTranscriptApi()


def normalize_youtube_url(url: str) -> str:
    url = (url or "").strip().strip('"').strip("'")
    url = unquote(url)  # decodes %5C etc if present
    # remove accidental escaping from shell/copy-paste
    url = url.replace("\\?", "?").replace("\\=", "=").replace("\\&", "&").replace("\\/", "/")
    # also remove any remaining backslashes
    url = url.replace("\\", "")
    return url

def extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from common URL formats.
    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
    Returns None if it cannot extract.
    """
    url = normalize_youtube_url(url)

    # 1) Standard watch?v=
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]
    except Exception:
        pass

    # 2) youtu.be/<id>, shorts/<id>, embed/<id>
    m = re.search(r"(?:youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)

    # 3) fallback: any 11-char token after v=
    m = re.search(r"v=([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def fetch_title_with_ytdlp(url: str) -> str | None:
    """
    Use yt-dlp to fetch video title without downloading the video.
    Returns None if it fails.
    """
    url = normalize_youtube_url(url)
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "nocheckcertificate": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title")
    except Exception as e:
        logging.info(f"Failed to fetch title for {url}: {e}")
        return None


def fetch_transcript(video_id: str) -> str | None:
    """
    Fetch transcript for a given video ID using the instance-based API.

    Strategy:
      1) Try English transcript: api.fetch(video_id, languages=["en"])
      2) If that fails, try any available transcript: api.fetch(video_id)
      3) Return concatenated text or None if unavailable.
    """
    # First attempt: English transcript
    try:
        transcript = api.fetch(video_id, languages=["en"])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        logging.info(f"No EN transcript for {video_id}: {e}")
        # Fallback: any language
        try:
            transcript = api.fetch(video_id)
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e2:
            logging.info(f"No transcript at all for {video_id}: {e2}")
            return None
        except Exception as e2:
            logging.info(f"Unexpected fallback error for {video_id}: {e2}")
            return None
    except Exception as e:
        logging.info(f"Unexpected error fetching transcript for {video_id}: {e}")
        return None

    # transcript is a list of FetchedTranscriptSnippet objects
    try:
        text = " ".join(getattr(seg, "text", "") for seg in transcript)
        text = text.replace("\n", " ").strip()
        return text or None
    except Exception as e:
        logging.info(f"Error concatenating transcript for {video_id}: {e}")
        return None


def ensure_parent_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_links(path: Path) -> list[str]:
    """
    Load URLs from the links file (one per line).
    Ignores empty lines and comment lines starting with '#'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Links file not found: {path}")

    urls: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def write_header_if_needed(csv_path: Path) -> None:
    """
    Create CSV file with header if it does not exist.
    """
    if not csv_path.exists():
        ensure_parent_dirs(csv_path)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "url", "title", "transcript"])


def append_row(csv_path: Path, row: list[str]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def process_all_links(
    links_file: Path = LINKS_FILE,
    output_csv: Path = OUTPUT_CSV,
) -> None:
    """
    Main entrypoint:
    - Load all URLs
    - Extract video IDs
    - Fetch title + transcript
    - Save to CSV
    """
    ensure_parent_dirs(ERROR_LOG)
    write_header_if_needed(output_csv)

    urls = load_links(links_file)
    logging.info(f"Loaded {len(urls)} URLs from {links_file}")

    processed = 0
    skipped = 0

    for idx, url in enumerate(urls, start=1):
        print(f"[{idx}/{len(urls)}] Processing: {url}")

        vid = extract_video_id(url)
        if not vid:
            logging.info(f"Could not extract video id from URL: {url}")
            skipped += 1
            continue

        transcript = fetch_transcript(vid)
        if not transcript:
            logging.info(f"Skipping {vid} (no transcript)")
            skipped += 1
            continue

        title = fetch_title_with_ytdlp(url) or ""

        append_row(output_csv, [vid, url, title, transcript])
        processed += 1

    logging.info(
        f"Finished. Processed={processed}, Skipped={skipped}, "
        f"Output={output_csv}"
    )
    print(f"Done. Processed={processed}, Skipped={skipped}. See {output_csv}")


if __name__ == "__main__":
    process_all_links()