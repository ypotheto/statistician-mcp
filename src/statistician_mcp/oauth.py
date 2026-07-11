from __future__ import annotations

from typing import Protocol

import jwt

from statistician_mcp.workspace import resolve_workspace_id


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
        same as `KeyStore.verify_key`, an invalid credential is just None."""
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(raw_token)
            claims = jwt.decode(
                raw_token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
            )
        except jwt.PyJWTError:
            return None

        if self._required_permission not in claims.get("permissions", []):
            return None

        sub = claims.get("sub")
        if not sub:
            return None
        return resolve_workspace_id(sub)
