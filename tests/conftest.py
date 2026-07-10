"""Shared fixtures: a tiny in-code PNG and offline C2PA signing helpers.

No media files are committed (see .gitignore); test assets are generated here at run time
and signed with the vendored FOR-TESTING-ONLY certs (tests/fixtures/certs/, MIT/Apache-2.0,
from contentauth/c2pa-python v0.36.0).
"""

from __future__ import annotations

import datetime
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def make_tiny_png() -> bytes:
    """A valid 1x1 red-pixel RGB PNG, built from scratch (~70 bytes; no Pillow needed)."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")  # filter byte + one RGB pixel
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


@pytest.fixture
def tiny_png() -> bytes:
    return make_tiny_png()


def cert_chain_pem() -> bytes:
    return (FIXTURES / "certs" / "es256_certs.pem").read_bytes()


def ca_anchor_pem() -> str:
    """The CA cert(s) from the test chain (everything after the leaf) — usable as a trust
    anchor to exercise the validation_state == "Trusted" path against our own signer."""
    pem = cert_chain_pem().decode("utf-8")
    end_of_leaf = pem.index("-----END CERTIFICATE-----") + len("-----END CERTIFICATE-----")
    return pem[end_of_leaf:].lstrip()


def sign_png(tmp_path: Path, manifest: dict[str, Any], *, name: str = "signed.png") -> Path:
    """Sign a fresh tiny PNG with the test certs and the given manifest; return the path."""
    from c2pa import Builder, C2paSignerInfo, Signer

    info = C2paSignerInfo(
        alg=b"es256",
        sign_cert=cert_chain_pem(),
        private_key=(FIXTURES / "certs" / "es256_private.key").read_bytes(),
        ta_url=b"",  # constructor rejects None ...
    )
    info.ta_url = None  # ... but sign-time rejects b"" — NULL means "no timestamp authority"
    signer = Signer.from_info(info)

    source = tmp_path / f"src-{name}"
    source.write_bytes(make_tiny_png())
    destination = tmp_path / name
    Builder(manifest).sign_file(source, destination, signer)  # Builder is single-sign-use
    return destination


def actions_manifest(
    digital_source_type: str | None, *, action: str = "c2pa.created"
) -> dict[str, Any]:
    """A minimal manifest with one action, optionally carrying a digitalSourceType URI."""
    act: dict[str, Any] = {"action": action}
    if digital_source_type is not None:
        act["digitalSourceType"] = digital_source_type
    return {
        "claim_generator_info": [{"name": "bedrock-tests", "version": "0"}],
        "title": "bedrock test asset",
        "assertions": [{"label": "c2pa.actions", "data": {"actions": [act]}}],
    }


# --- throwaway cert chains, so tests can pit a trusted signer against an untrusted one -------


@dataclass(frozen=True)
class CertChain:
    """A leaf+CA chain meeting the C2PA cert profile, plus the CA PEM to use as a trust anchor."""

    chain_pem: bytes
    key_pem: bytes
    ca_pem: str


def make_cert_chain(org: str) -> CertChain:
    """Generate an ES256 leaf+CA chain c2pa will accept for signing.

    The profile is fussy: EKU emailProtection must be *critical*, and both SKI and AKI must be
    present — omit either and c2pa rejects the cert at sign time.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.UTC)
    span = (now - datetime.timedelta(days=1), now + datetime.timedelta(days=3650))
    email_protection = x509.ObjectIdentifier("1.3.6.1.5.5.7.3.4")

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"{org} CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
    ])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(span[0])
        .not_valid_after(span[1])
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, f"{org} Signer"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
            ])
        )
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(span[0])
        .not_valid_after(span[1])
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([email_protection]), critical=True)
        .add_extension(
            x509.KeyUsage(True, True, False, False, False, False, False, False, False),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    pem = serialization.Encoding.PEM
    return CertChain(
        chain_pem=leaf.public_bytes(pem) + ca.public_bytes(pem),
        key_pem=leaf_key.private_bytes(
            pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ),
        ca_pem=ca.public_bytes(pem).decode("ascii"),
    )


def signer_from(chain: CertChain) -> Any:
    from c2pa import C2paSignerInfo, Signer

    info = C2paSignerInfo(
        alg=b"es256", sign_cert=chain.chain_pem, private_key=chain.key_pem, ta_url=b""
    )
    info.ta_url = None  # NULL = no timestamp authority (b"" fails at sign time)
    return Signer.from_info(info)
