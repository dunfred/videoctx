from __future__ import annotations

from pathlib import Path

from .schema import VideoContext

MODEL_NAME = "clip-ViT-B-32"
EMB_FILE = "embeddings.npz"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_shots(index_dir: Path, ctx: VideoContext, progress=lambda *_: None) -> int:
    """Embed each shot's middle keyframe with a local CLIP model and store the
    vectors beside the index. Free, offline, one-time. Idempotent: skips if
    the embedding file already covers all shots."""
    import numpy as np
    from PIL import Image

    out = index_dir / EMB_FILE
    shots = [s for s in ctx.shots if s.keyframes]
    if out.exists():
        existing = np.load(out)
        if len(existing["shot_ids"]) >= len(shots):
            return 0

    model = _get_model()
    images, ids, times = [], [], []
    for shot in shots:
        kf = shot.keyframes[len(shot.keyframes) // 2]
        images.append(Image.open(index_dir / kf.file))
        ids.append(shot.id)
        times.append(kf.t_s)
    vecs = model.encode(images, batch_size=16, show_progress_bar=False)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    np.savez(out, vectors=vecs.astype(np.float32),
             shot_ids=np.array(ids), t_s=np.array(times),
             model=np.array([MODEL_NAME]))
    progress(f"  embedded {len(ids)} shots -> {out.name}")
    return len(ids)


def find_moments(index_dir: Path, query: str, top_k: int = 5) -> list[dict]:
    """Return the top_k shots whose keyframes best match a text description.
    Each hit: {shot_id, t_s, score}."""
    import numpy as np

    data = np.load(index_dir / EMB_FILE)
    q = _get_model().encode([query])[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = data["vectors"] @ q
    order = np.argsort(-sims)[:top_k]
    return [
        {"shot_id": int(data["shot_ids"][i]), "t_s": float(data["t_s"][i]),
         "score": float(sims[i])}
        for i in order
    ]
