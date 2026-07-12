from __future__ import annotations

import logging
from typing import Any, Protocol

import jwt

from statistician_mcp.workspace import resolve_workspace_id

logger = logging.getLogger(__name__)


class SigningKeyResolver(Protocol):
    """The one method of `jwt.PyJWKClient` this module actually depends on --
    narrowed to a Protocol so tests can substitute a static key instead of
    standing up a real JWKS-serving endpoint. Kinde's own JWKS-fetch/cache
    mechanics are PyJWT's well-tested concern, not ours; what's actually worth
    testing here is our own claim validation (issuer/audience/permission/sub)."""

    def get_signing_key_from_jwt(self, token: str) -> jwt.PyJWK: ...


class OAuthVerifier:
    """Validates Kinde-issued access tokens for a single (issuer, audience,
    required-permission) triple, and resolves a valid token to a workspace id.

    Kinde is the authorization server here; this class only ever plays the
    OAuth *resource server* role -- there is no /authorize or /token endpoint
    on this side, just signature and claim verification of tokens minted
    elsewhere. Per the MCP authorization spec, audience validation (`aud` must
    match this server) is what stops a token minted for a *different*
    Kinde-protected API from being replayed here.

    STOPGAP: Kinde only scopes a token's `aud` claim via its own proprietary
    `audience` authorization-request parameter, not RFC 8707's `resource`
    parameter that MCP clients (Claude) are required to send -- so a token
    from that flow has no `aud` claim at all. Until Kinde adds `resource`
    support (or a workaround surfaces), a missing `aud` is accepted; a
    *present-but-wrong* `aud` is still rejected, so this doesn't fully drop
    audience binding, just relaxes it to "not provably for someone else"
    rather than "provably for us." Tighten this back to a hard requirement
    once Kinde honors `resource`.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        required_permission: str,
        jwk_client: SigningKeyResolver | None = None,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._required_permission = required_permission
        self._jwk_client: SigningKeyResolver = jwk_client or jwt.PyJWKClient(
            f"{self._issuer}/.well-known/jwks"
        )

    def verify(self, raw_token: str) -> str | None:
        """Return the resolved workspace id for a valid, sufficiently-
        permissioned token, or None for any failure (bad/unknown-key signature,
        wrong issuer/audience, expired, missing the required permission,
        missing `sub`). Callers don't need to distinguish *why* it failed --
        same as `KeyStore.verify_key`, an invalid credential is just None.
        Every rejection is logged with the reason and the token's *claims*
        (never the raw token itself) to make this diagnosable in production."""
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(raw_token)
            claims = jwt.decode(
                raw_token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                options={"require": ["exp"], "verify_aud": False},
            )
        except jwt.PyJWTError as exc:
            self._log_rejection(raw_token, str(exc))
            return None

        aud = claims.get("aud")
        if aud is not None:
            aud_values = aud if isinstance(aud, list) else [aud]
            if self._audience not in aud_values:
                self._log_rejection(raw_token, "aud claim present but does not match")
                return None

        if self._required_permission not in claims.get("permissions", []):
            self._log_rejection(
                raw_token,
                f"missing required permission {self._required_permission!r}",
            )
            return None

        sub = claims.get("sub")
        if not sub:
            self._log_rejection(raw_token, "missing 'sub' claim")
            return None
        return resolve_workspace_id(sub)

    def _log_rejection(self, raw_token: str, reason: str) -> None:
        # Decoded *without* signature verification purely to log what the
        # token actually claims -- never used for any authorization decision.
        unverified: dict[str, Any]
        try:
            unverified = jwt.decode(raw_token, options={"verify_signature": False})
        except jwt.PyJWTError:
            unverified = {}
        logger.warning(
            "oauth token rejected: %s (expected issuer=%r audience=%r "
            "permission=%r; token had iss=%r aud=%r sub=%r permissions=%r)",
            reason,
            self._issuer,
            self._audience,
            self._required_permission,
            unverified.get("iss"),
            unverified.get("aud"),
            unverified.get("sub"),
            unverified.get("permissions"),
        )
