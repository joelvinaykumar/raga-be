import os
import logging
from dotenv import load_dotenv
from fastapi import Request, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
logger = logging.getLogger(__name__)

if not SUPABASE_URL or not SUPABASE_JWT_SECRET:
    raise RuntimeError("SUPABASE_URL and SUPABASE_JWT_SECRET must be configured")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_JWT_SECRET)




class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid token or expired token.")
            return credentials.credentials
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

    def verify_jwt(self, jwtoken: str) -> bool:
        isTokenValid: bool = False
        try:
            user_claims = supabase.auth.get_claims(jwtoken)
            if user_claims:
                isTokenValid = True
        except Exception as e:
            logger.warning("JWT verification failed: %s", str(e))
            isTokenValid = False
        
        return isTokenValid


def get_current_user(token: str = Depends(JWTBearer())) -> dict:
    """Resolve the authenticated user's claims from a verified Supabase JWT.

    Returns a dict with at least `user_id` (the Supabase `sub`) and `email`.
    """
    try:
        claims = supabase.auth.get_claims(token)
    except Exception as e:
        logger.warning("Could not read JWT claims: %s", str(e))
        raise HTTPException(status_code=401, detail="Invalid token or expired token.")

    if not claims:
        raise HTTPException(status_code=401, detail="Invalid token or expired token.")

    # `get_claims` returns a ClaimsResponse TypedDict: {claims, headers, signature}.
    # The actual JWT payload (with `sub`, `email`) lives under `claims`.
    if isinstance(claims, dict):
        data = claims.get("claims") or claims
    else:
        data = getattr(claims, "claims", None) or {}

    user_id = data.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject claim.")

    return {"user_id": user_id, "email": data.get("email"), "claims": data}


def require_api_key(x_api_key: str = Header(None, alias="x-api-key")) -> str:
    """Authenticate an MCP client via the `x-api-key` header.

    Validates the key against `api_keys`, records last-used telemetry, and
    returns the owning user's id. Raises 401 for missing/invalid/revoked keys.
    """
    # Imported lazily to avoid a circular import (db_utils has no auth deps).
    from db_utils import get_api_key_record, touch_api_key_last_used

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header.")

    record = get_api_key_record(x_api_key)
    if not record:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key.")

    touch_api_key_last_used(x_api_key)
    return record["user_id"]



# def verify_token_with_secret(token: str) -> Dict:
#     """
#     Verify JWT token using Supabase JWT secret.
#     This is the simpler and recommended approach.
#     """
#     try:
#         payload = jwt.decode(
#             token,
#             SUPABASE_JWT_SECRET,
#             algorithms=["HS256"],
#             audience="authenticated"
#         )
#         return payload
#     except JWTError as e:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail=f"Invalid authentication credentials: {str(e)}",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
# async def get_current_user(
#     credentials: HTTPAuthorizationCredentials = Depends(security)
# ) -> Dict:
#     """
#     Dependency to extract and verify the current user from the JWT token.
#     Use this in your route dependencies.
#     """
#     token = credentials.credentials

#     # Choose verification method (Method 1 is simpler and recommended)
#     payload = verify_token_with_secret(token)
#     # OR use: payload = verify_token_with_jwks(token)

#     return {
#         "user_id": payload.get("sub"),
#         "email": payload.get("email"),
#         "role": payload.get("role"),
#         "payload": payload
#     }
