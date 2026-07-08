"""Ingest + modality routing: turn raw input (text, bytes, or a file) into an InputContext.

M0 sniffs modality by extension then by magic bytes. Only TEXT has a working signal today, so
non-text inputs correctly route to zero applicable signals → abstain (which is the honest M0
answer and proves the router).
"""

from __future__ import annotations

from pathlib import Path

from .types import InputContext, Modality

_IMAGE_EXT = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".avif", ".heif", ".heic", ".bmp",
}
_AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html"}

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_JPEG_SIG = b"\xff\xd8\xff"


def sniff_modality(*, path: str | None, data: bytes | None) -> Modality:
    """Best-effort modality detection: extension first, then magic bytes, defaulting to TEXT."""
    if path is not None:
        ext = Path(path).suffix.lower()
        if ext in _IMAGE_EXT:
            return Modality.IMAGE
        if ext in _AUDIO_EXT:
            return Modality.AUDIO
        if ext in _VIDEO_EXT:
            return Modality.VIDEO
        if ext in _TEXT_EXT:
            return Modality.TEXT
    if data is not None:
        if data.startswith(_PNG_SIG) or data.startswith(_JPEG_SIG):
            return Modality.IMAGE
        if data[:3] == b"ID3" or data.startswith(b"RIFF"):
            return Modality.AUDIO
        if data[4:8] == b"ftyp":
            return Modality.VIDEO
    return Modality.TEXT


def context_from_text(text: str) -> InputContext:
    return InputContext(modality=Modality.TEXT, text=text, mime="text/plain")


def context_from_bytes(
    data: bytes, *, path: str | None = None, mime: str | None = None
) -> InputContext:
    modality = sniff_modality(path=path, data=data)
    text = data.decode("utf-8", errors="replace") if modality is Modality.TEXT else None
    return InputContext(modality=modality, text=text, data=data, path=path, mime=mime)


def context_from_path(path: str | Path) -> InputContext:
    p = Path(path)
    return context_from_bytes(p.read_bytes(), path=str(p))
