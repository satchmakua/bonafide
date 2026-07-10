"""Ingest + modality routing: turn raw input (text, bytes, or a file) into an InputContext.

Modality is sniffed by extension first, then by magic bytes. Container formats need their
*brand* checked, not just the leading magic: RIFF is both WebP and WAV, and `ftyp` is both
HEIC/AVIF and MP4 — get this wrong and a signed image silently routes to a modality with no
signals, which reads as a bland "abstain" rather than the bug it is.
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
# ISO-BMFF brands (bytes 8..12, after the `ftyp` box type) that denote a still image.
_IMAGE_FTYP_BRANDS = {b"heic", b"heix", b"heif", b"mif1", b"msf1", b"avif", b"avis"}


def sniff_mime(data: bytes) -> str | None:
    """The MIME type of an image byte string, or None if it isn't a format we recognize."""
    if data.startswith(_PNG_SIG):
        return "image/png"
    if data.startswith(_JPEG_SIG):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"avif", b"avis"}:
            return "image/avif"
        if brand in _IMAGE_FTYP_BRANDS:
            return "image/heic"
    return None


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
        if data[:3] == b"ID3":
            return Modality.AUDIO
        if data.startswith(b"RIFF"):  # WEBP vs WAVE/AVI
            return Modality.IMAGE if data[8:12] == b"WEBP" else Modality.AUDIO
        if data[4:8] == b"ftyp":  # HEIC/AVIF vs MP4/MOV
            return Modality.IMAGE if data[8:12] in _IMAGE_FTYP_BRANDS else Modality.VIDEO
    return Modality.TEXT


def context_from_text(text: str) -> InputContext:
    return InputContext(modality=Modality.TEXT, text=text, mime="text/plain")


def context_from_bytes(
    data: bytes, *, path: str | None = None, mime: str | None = None
) -> InputContext:
    modality = sniff_modality(path=path, data=data)
    text = data.decode("utf-8", errors="replace") if modality is Modality.TEXT else None
    if mime is None:
        mime = sniff_mime(data)
    return InputContext(modality=modality, text=text, data=data, path=path, mime=mime)


def context_from_path(path: str | Path) -> InputContext:
    p = Path(path)
    return context_from_bytes(p.read_bytes(), path=str(p))
