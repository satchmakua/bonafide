# Test fixtures

`certs/es256_certs.pem` + `certs/es256_private.key` are the standard C2PA **test-only**
signing credentials (subject `O=C2PA Test Signing Cert, OU=FOR TESTING_ONLY`), vendored from
[contentauth/c2pa-python](https://github.com/contentauth/c2pa-python) `tests/fixtures/` at tag
`v0.36.0` (dual-licensed MIT / Apache-2.0). The private key is public and FOR TESTING ONLY —
it signs the throwaway assets our test suite generates at run time; it must never sign
anything real. Note: the leaf cert expires 2030-08-26 (c2pa checks validity at sign time), so
these fixtures need refreshing before then.

No media files live here (see `.gitignore`): test images are generated in code (a tiny PNG
built in `conftest.py`) and signed on the fly.
