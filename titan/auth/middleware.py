"""
Titan Authentication - FastAPI Middleware

Provides authentication dependencies for FastAPI endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from titan.auth.models import User, UserRole

logger = logging.getLogger("titan.auth.middleware")

# Security schemes
http_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthenticationError(HTTPException):  # type: ignore[misc]
    """Authentication failed."""

    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):  # type: ignore[misc]
    """Authorization failed (insufficient permissions)."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


async def _authenticate_jwt(token: str) -> User:  # allow-secret
    """Authenticate using JWT token."""
    from titan.auth.jwt import JWTError, verify_token
    from titan.auth.storage import get_auth_storage

    try:
        payload = verify_token(token, expected_type="access")  # allow-secret
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e}")  # allow-secret

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token payload")

    # Get user from storage to verify they still exist and are active
    storage = await get_auth_storage()
    user_data = await storage.get_user(user_id)

    if not user_data:
        raise AuthenticationError("User not found")

    if not user_data.get("is_active", False):
        raise AuthenticationError("User account is disabled")

    return User.from_dict(user_data)


async def _authenticate_api_key(key: str) -> User:
    """Authenticate using API key."""
    from titan.auth.api_keys import hash_api_key, is_key_valid
    from titan.auth.storage import get_auth_storage

    key_hash = hash_api_key(key)
    storage = await get_auth_storage()

    # Look up key by hash
    key_data = await storage.get_api_key_by_hash(key_hash)
    if not key_data:
        raise AuthenticationError("Invalid API key")

    # Validate key
    is_valid, error = is_key_valid(
        key_hash,
        key_data["key_hash"],
        key_data.get("expires_at"),
        key_data.get("is_active", False),
    )

    if not is_valid:
        raise AuthenticationError(error or "Invalid API key")

    # Update last used timestamp (fire and forget)
    await storage.update_api_key_last_used(key_data["id"])

    # Get the user associated with this key
    user_data = await storage.get_user(key_data["user_id"])
    if not user_data:
        raise AuthenticationError("API key owner not found")

    if not user_data.get("is_active", False):
        raise AuthenticationError("API key owner account is disabled")

    return User.from_dict(user_data)


async def get_current_user(
    request: Request,
    bearer_token: HTTPAuthorizationCredentials | None = Depends(http_bearer),  # allow-secret
    api_key: str | None = Depends(api_key_header),  # allow-secret
) -> User:
    """
    Extract and validate the current user from request credentials.

    Supports both JWT Bearer tokens and API keys.

    Args:
        request: The FastAPI request
        bearer_token: Optional JWT bearer credential  # allow-secret
        api_key: Optional key from X-API-Key header  # allow-secret

    Returns:
        The authenticated User

    Raises:
        AuthenticationError: If no valid credentials provided
    """
    # Try JWT first
    if bearer_token and bearer_token.credentials:
        user = await _authenticate_jwt(bearer_token.credentials)
        # Store user in request state for downstream use
        request.state.user = user
        return user

    # Try API key
    if api_key:
        user = await _authenticate_api_key(api_key)
        request.state.user = user
        return user

    raise AuthenticationError("No valid credentials provided")


async def get_current_user_optional(
    request: Request,
    bearer_token: HTTPAuthorizationCredentials | None = Depends(http_bearer),  # allow-secret
    api_key: str | None = Depends(api_key_header),  # allow-secret
) -> User | None:
    """
    Extract the current user if credentials are provided, otherwise return None.

    Use this for endpoints that should work with or without authentication.
    """
    if not bearer_token and not api_key:
        return None

    try:
        return await get_current_user(request, bearer_token, api_key)
    except AuthenticationError:
        return None


def require_role(
    allowed_roles: list[UserRole],
) -> Callable[..., Coroutine[Any, Any, User]]:
    """
    Dependency factory for role-based access control.

    Args:
        allowed_roles: List of roles that are allowed to access the endpoint

    Returns:
        A dependency function that verifies the user has the required role

    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(user: User = Depends(require_role([UserRole.ADMIN]))):
            ...
    """

    async def role_checker(
        user: User = Depends(get_current_user),
    ) -> User:
        if user.role not in allowed_roles:
            raise AuthorizationError(
                f"This endpoint requires one of these roles: {[r.value for r in allowed_roles]}"
            )
        return user

    return role_checker


# Common role requirements as dependencies
async def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if user.role != UserRole.ADMIN:
        raise AuthorizationError("Admin access required")
    return user


async def require_user_or_admin(user: User = Depends(get_current_user)) -> User:
    """Require user or admin role (not readonly or service)."""
    if user.role not in (UserRole.ADMIN, UserRole.USER):
        raise AuthorizationError("User or admin access required")
    return user


async def require_active_user(user: User = Depends(get_current_user)) -> User:
    """Require that the user is active."""
    if not user.is_active:
        raise AuthorizationError("Account is disabled")
    return user


def scope_required(
    required_scope: str,
) -> Callable[..., Coroutine[Any, Any, User]]:
    """
    Dependency factory for scope-based access control (for API keys).

    Args:
        required_scope: The scope required to access the endpoint

    Returns:
        A dependency function that verifies the API key has the required scope
    """

    async def scope_checker(
        request: Request,
        bearer_token: HTTPAuthorizationCredentials | None = Depends(http_bearer),  # allow-secret
        api_key: str | None = Depends(api_key_header),  # allow-secret
    ) -> User:
        # First authenticate
        user = await get_current_user(request, bearer_token, api_key)

        # If using API key, check scopes
        if api_key:
            from titan.auth.api_keys import hash_api_key
            from titan.auth.storage import get_auth_storage

            key_hash = hash_api_key(api_key)
            storage = await get_auth_storage()
            key_data = await storage.get_api_key_by_hash(key_hash)

            if key_data:
                scopes = key_data.get("scopes", [])
                # Empty scopes means full access
                if scopes and required_scope not in scopes:
                    raise AuthorizationError(
                        f"API key does not have required scope: {required_scope}"
                    )

        return user

    return scope_checker


class RateLimitExceeded(HTTPException):  # type: ignore[misc]
    """Rate limit exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
