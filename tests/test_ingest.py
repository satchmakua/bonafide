"""Modality/MIME sniffing. Container brands matter: RIFF is WebP *and* WAV; `ftyp` is
HEIC/AVIF *and* MP4. Misroute an image and its provenance signal silently never runs."""

from __future__ import annotations

from bonafide.ingest import context_from_bytes, sniff_mime, sniff_modality
from bonafide.types import Modality

WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16
WAV = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 16
HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 16
AVIF = b"\x00\x00\x00\x18ftypavif" + b"\x00" * 16
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def test_extensionless_image_containers_route_to_image() -> None:
    # A browser Blob upload arrives as filename "blob" — extension tells us nothing.
    for data in (WEBP, HEIC, AVIF, PNG, JPEG):
        assert sniff_modality(path="blob", data=data) is Modality.IMAGE


def test_riff_and_ftyp_non_image_brands_still_route_correctly() -> None:
    assert sniff_modality(path="blob", data=WAV) is Modality.AUDIO
    assert sniff_modality(path="blob", data=MP4) is Modality.VIDEO


def test_sniff_mime_reads_container_brands() -> None:
    assert sniff_mime(WEBP) == "image/webp"
    assert sniff_mime(HEIC) == "image/heic"
    assert sniff_mime(AVIF) == "image/avif"
    assert sniff_mime(PNG) == "image/png"
    assert sniff_mime(JPEG) == "image/jpeg"


def test_sniff_mime_does_not_mislabel_audio_as_webp() -> None:
    assert sniff_mime(WAV) is None
    assert context_from_bytes(WAV, path="clip.wav").mime is None


def test_extension_wins_over_magic_bytes() -> None:
    assert sniff_modality(path="clip.wav", data=WAV) is Modality.AUDIO
    assert sniff_modality(path="photo.webp", data=WEBP) is Modality.IMAGE
