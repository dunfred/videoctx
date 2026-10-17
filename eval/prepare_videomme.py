"""Prepare a Video-MME subset: download videos with yt-dlp and emit the
questions.jsonl that eval/run.py consumes.

Requires: pip install datasets   and   yt-dlp on PATH.
HF dataset: lmms-lab/Video-MME (test split). Videos are YouTube-hosted; some
will be unavailable (removed/region-locked) — those are skipped and reported.

Usage:
    python eval/prepare_videomme.py --duration short --videos 10 --out eval/data
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import click

LETTER_PREFIX = re.compile(r"^[A-D][\.\)]\s*")


@click.command()
@click.option("--duration", type=click.Choice(["short", "medium", "long"]), default="short",
              show_default=True, help="Video-MME duration category")
@click.option("--videos", "n_videos", default=10, show_default=True, help="Number of videos")
@click.option("--out", type=click.Path(path_type=Path), default=Path("eval/data"), show_default=True)
@click.option("--max-height", default=480, show_default=True, help="Download resolution cap")
def main(duration: str, n_videos: int, out: Path, max_height: int):
    from datasets import load_dataset

    out.mkdir(parents=True, exist_ok=True)
    videos_dir = out / "videos"
    videos_dir.mkdir(exist_ok=True)

    ds = load_dataset("lmms-lab/Video-MME", split="test")
    by_video: dict[str, list] = defaultdict(list)
    for row in ds:
        if row["duration"] == duration:
            by_video[row["videoID"]].append(row)

    click.echo(f"{len(by_video)} {duration} videos available; taking up to {n_videos}")

    questions = []
    downloaded = 0
    for yt_id, rows in by_video.items():
        if downloaded >= n_videos:
            break
        video_path = videos_dir / f"{yt_id}.mp4"
        if not video_path.exists():
            click.echo(f"downloading {yt_id} ({rows[0].get('domain', '?')})")
            result = subprocess.run(
                ["yt-dlp", "-q",
                 "-f", f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b",
                 "--merge-output-format", "mp4",
                 "-o", str(video_path),
                 f"https://www.youtube.com/watch?v={yt_id}"],
            )
            if result.returncode != 0 or not video_path.exists():
                click.echo(f"  SKIP {yt_id}: download failed (removed/blocked)")
                continue
        downloaded += 1
        for row in rows:
            options = [LETTER_PREFIX.sub("", o).strip() for o in row["options"]]
            questions.append({
                "id": row.get("question_id", f"{yt_id}-{len(questions)}"),
                "video": str(video_path),
                "question": row["question"],
                "options": options,
                "answer": row["answer"].strip().upper(),
                "duration": duration,
                "domain": row.get("domain"),
            })

    qfile = out / f"videomme_{duration}.jsonl"
    qfile.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    click.echo(f"\n{downloaded} videos, {len(questions)} questions -> {qfile}")
    click.echo(f"run: python eval/run.py {qfile} --methods videoctx,naive")


if __name__ == "__main__":
    main()
