import os
import logging
from dotenv import load_dotenv
from fastapi import Request, HTTPException
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
