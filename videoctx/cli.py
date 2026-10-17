from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def main():
    """videoctx: index videos once, query them cheaply with any LLM."""


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Index output dir")
@click.option("--max-edge", default=768, show_default=True, help="Keyframe long-edge pixels")
@click.option("--whisper-model", default="small", show_default=True)
@click.option("--no-transcript", is_flag=True, help="Skip transcription")
def index(video: Path, out: Path | None, max_edge: int, whisper_model: str, no_transcript: bool):
    """Index VIDEO into a .vctx directory."""
    from .indexer import build_index

    build_index(
        video, out_dir=out, max_edge=max_edge,
        whisper_model=whisper_model, skip_transcript=no_transcript,
        progress=click.echo,
    )


@main.command()
@click.argument("index_dir", type=click.Path(exists=True, path_type=Path))
def caption(index_dir: Path):
    """Add visual captions to an index's shots (one cheap VLM call per shot)."""
    from .captions import caption_shots
    from .indexer import load_index

    ctx = load_index(index_dir)
    meter = caption_shots(index_dir, ctx, progress=click.echo)
    click.echo(f"captioned; cost: {json.dumps(meter.summary())}")


@main.command()
@click.argument("index_dir", type=click.Path(exists=True, path_type=Path))
def show(index_dir: Path):
    """Print the text view of an index (what the model reads)."""
    from .indexer import load_index
    from .textview import text_view

    click.echo(text_view(load_index(index_dir)))


@main.command()
@click.argument("index_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("question")
@click.option("--model", default="claude-opus-4-8", show_default=True)
def ask(index_dir: Path, question: str, model: str):
    """Ask a question about an indexed video."""
    from .indexer import load_index
    from .query import ask as _ask

    ctx = load_index(index_dir)
    answer, meter, frames = _ask(index_dir, ctx, question, model=model)
    click.echo(answer)
    click.echo("---")
    click.echo(f"cost: {json.dumps(meter.summary())} frames_fetched={frames}")


if __name__ == "__main__":
    main()
