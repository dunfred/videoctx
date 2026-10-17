"""Phase 0 eval runner.

Questions come from a JSONL file with one object per line:
    {"video": "path/to/video.mp4", "question": "...", "options": ["...", ...],
     "answer": "B", "id": "optional-id"}

For Video-MME, use prepare_videomme.py to download videos (yt-dlp) and emit
this JSONL. For a quick pilot, hand-write a few questions about any local
videos.

Usage:
    python eval/run.py questions.jsonl --methods videoctx,naive --out eval/results/pilot.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.methods import METHODS  # noqa: E402


@click.command()
@click.argument("questions_file", type=click.Path(exists=True, path_type=Path))
@click.option("--methods", "methods_csv", default="videoctx,naive", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("eval/results/run.json"))
@click.option("--limit", type=int, default=None, help="Only run the first N questions")
@click.option("--skip-existing", is_flag=True,
              help="Resume: skip (question, method) pairs already in the out file")
def main(questions_file: Path, methods_csv: str, out: Path, limit: int | None,
         skip_existing: bool):
    questions = [json.loads(line) for line in questions_file.read_text().splitlines() if line.strip()]
    if limit:
        questions = questions[:limit]
    method_names = [m.strip() for m in methods_csv.split(",")]

    results = []
    done: set[tuple] = set()
    if skip_existing and out.exists():
        results = json.loads(out.read_text())
        # Errored rows are retried; completed rows are kept.
        results = [r for r in results if "error" not in r]
        done = {(r["qid"], r["method"]) for r in results}
        click.echo(f"resuming: {len(done)} completed rows kept")
    quota_exhausted = False
    for qi, q in enumerate(questions):
        if quota_exhausted:
            break
        video = Path(q["video"])
        for name in method_names:
            if (q.get("id", qi), name) in done:
                continue
            fn = METHODS[name]
            t0 = time.monotonic()
            try:
                r = fn(video, q["question"], q["options"])
                row = {
                    "qid": q.get("id", qi), "method": name,
                    "predicted": r.predicted_letter, "gold": q["answer"],
                    "correct": r.predicted_letter == q["answer"],
                    "usd": round(r.usd, 6),
                    "wall_s": round(time.monotonic() - t0, 1),
                    "detail": r.detail, "answer_text": r.answer_text,
                }
            except Exception as e:  # record failures, keep going
                row = {
                    "qid": q.get("id", qi), "method": name, "error": repr(e),
                    "correct": False, "usd": 0.0,
                    "wall_s": round(time.monotonic() - t0, 1),
                }
            click.echo(f"[{qi+1}/{len(questions)}] {name}: "
                       f"{'ERR ' + row.get('error', '') if 'error' in row else ('OK' if row['correct'] else 'WRONG')} "
                       f"(${row['usd']:.4f}, {row['wall_s']}s)")
            results.append(row)
            if "RESOURCE_EXHAUSTED" in row.get("error", ""):
                click.echo("daily quota exhausted; stopping this sweep")
                quota_exhausted = True
                break
            # Write incrementally so a killed run keeps its completed rows.
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2))

    click.echo("\n== accuracy per dollar ==")
    for name in method_names:
        rows = [r for r in results if r["method"] == name]
        answered = [r for r in rows if "error" not in r]
        acc = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
        cost = sum(r["usd"] for r in rows)
        click.echo(
            f"{name:>10}: acc={acc:.1%} ({sum(r['correct'] for r in rows)}/{len(rows)}, "
            f"{len(rows) - len(answered)} errors)  total=${cost:.4f}  "
            f"acc/$ = {acc / cost:.1f}" if cost else
            f"{name:>10}: acc={acc:.1%}  total=$0"
        )
    click.echo(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
