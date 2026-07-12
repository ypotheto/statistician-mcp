from __future__ import annotations

import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from statistician_mcp.oauth import OAuthVerifier
from statistician_mcp.workspace import resolve_workspace_id

_ISSUER = "https://test-tenant.kinde.com"
_AUDIENCE = "https://statistician-mcp.example/mcp"
_PERMISSION = "access:statistician-mcp"


class _StaticSigningKey:
    def __init__(self, key: Any) -> None:
        self.key = key


class _StaticJWKClient:
    """Stands in for `jwt.PyJWKClient` in tests -- PyJWKClient's own JWKS
    fetch-and-cache mechanics are PyJWT's well-tested concern, not ours; what's
    worth testing here is our own claim validation, using real RS256
    signing/verification rather than a mock that would just assume it works."""

    def __init__(self, key: Any) -> None:
        self._signing_key = _StaticSigningKey(key)

    def get_signing_key_from_jwt(self, token: str) -> Any:
        return self._signing_key


@pytest.fixture(scope="module")
def keypair() -> tuple[RSAPrivateKey, Any]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture
def verifier(keypair: tuple[RSAPrivateKey, Any]) -> OAuthVerifier:
    _, public_key = keypair
    return OAuthVerifier(
        issuer=_ISSUER,
        audience=_AUDIENCE,
        required_permission=_PERMISSION,
        jwk_client=_StaticJWKClient(public_key),
    )


def _make_token(
    private_key: RSAPrivateKey,
    *,
    issuer: str = _ISSUER,
    audience: str | None = _AUDIENCE,
    permissions: list[str] | None = None,
    sub: str | None = "kp_abc123",
    expired: bool = False,
) -> str:
    now = time.time()
    claims: dict[str, Any] = {
        "iss": issuer,
        "iat": now - 10,
        "exp": now - 5 if expired else now + 300,
        "permissions": [_PERMISSION] if permissions is None else permissions,
    }
    if audience is not None:
        claims["aud"] = audience
    if sub is not None:
        claims["sub"] = sub
    return jwt.encode(claims, private_key, algorithm="RS256")


def test_valid_token_resolves_to_workspace_id(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    token = _make_token(private_key)
    assert verifier.verify(token) == resolve_workspace_id("kp_abc123")


def test_different_users_resolve_to_different_workspaces(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    token_a = _make_token(private_key, sub="kp_a")
    token_b = _make_token(private_key, sub="kp_b")
    assert verifier.verify(token_a) != verifier.verify(token_b)


def test_wrong_issuer_rejected(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    token = _make_token(private_key, issuer="https://not-us.kinde.com")
    assert verifier.verify(token) is None


def test_wrong_audience_rejected(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    token = _make_token(private_key, audience="https://some-other-api.example/mcp")
    assert verifier.verify(token) is None


def test_missing_aud_claim_accepted_as_stopgap(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    """Kinde doesn't honor RFC 8707's `resource` param (only its own
    `audience` param), so a token minted through the Claude/MCP flow has no
    usable `aud` -- see the STOPGAP note on OAuthVerifier. A missing `aud`
    must still verify; test_wrong_audience_rejected above covers that a
    *present-but-wrong* one still doesn't."""
    private_key, _ = keypair
    token = _make_token(private_key, audience=None)
    assert verifier.verify(token) == resolve_workspace_id("kp_abc123")


def test_empty_aud_list_accepted_as_stopgap(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    """The shape Kinde actually issues (observed live): aud=[] -- an empty
    array, not an omitted claim. Must be treated the same as missing."""
    private_key, _ = keypair
    now = time.time()
    token = jwt.encode(
        {
            "iss": _ISSUER,
            "aud": [],
            "sub": "kp_abc123",
            "exp": now + 300,
            "permissions": [_PERMISSION],
        },
        private_key,
        algorithm="RS256",
    )
    assert verifier.verify(token) == resolve_workspace_id("kp_abc123")


def test_expired_token_rejected(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    token = _make_token(private_key, expired=True)
    assert verifier.verify(token) is None


def test_missing_permission_rejected(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    token = _make_token(private_key, permissions=["some:other-permission"])
    assert verifier.verify(token) is None


def test_no_permissions_claim_at_all_rejected(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    now = time.time()
    token = jwt.encode(
        {"iss": _ISSUER, "aud": _AUDIENCE, "sub": "kp_abc123", "exp": now + 300},
        private_key,
        algorithm="RS256",
    )
    assert verifier.verify(token) is None


def test_missing_sub_rejected(
    verifier: OAuthVerifier, keypair: tuple[RSAPrivateKey, Any]
) -> None:
    private_key, _ = keypair
    token = _make_token(private_key, sub=None)
    assert verifier.verify(token) is None


def test_tampered_signature_rejected(verifier: OAuthVerifier) -> None:
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _make_token(other_private_key)  # signed with a key the verifier doesn't trust
    assert verifier.verify(token) is None


def test_garbage_token_rejected(verifier: OAuthVerifier) -> None:
    assert verifier.verify("not-a-jwt-at-all") is None
