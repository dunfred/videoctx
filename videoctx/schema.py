from __future__ import annotations

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool


class Keyframe(BaseModel):
    t_s: float
    file: str
    width: int
    height: int


class Shot(BaseModel):
    id: int
    start_s: float
    end_s: float
    keyframes: list[Keyframe] = Field(default_factory=list)
    caption: str | None = None
    caption_model: str | None = None
    # On-screen text from local OCR. None = not run; "" = ran, nothing found.
    ocr: str | None = None


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str


class Transcript(BaseModel):
    language: str | None = None
    model: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


class VideoContext(BaseModel):
    video: VideoInfo
    shots: list[Shot]
    transcript: Transcript
