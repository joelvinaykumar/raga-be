import os
import secrets
import string
from datetime import datetime, timezone
from fastapi import HTTPException
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_JWT_SECRET")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# API keys (used to authenticate MCP clients via the `x-api-key` header)
# ---------------------------------------------------------------------------

_API_KEY_ALPHABET = string.ascii_lowercase + string.digits


def _generate_api_key() -> str:
    """Return a key shaped like `raga-token-<6-8 alphanumerics>`."""
    length = secrets.choice((6, 7, 8))
    suffix = "".join(secrets.choice(_API_KEY_ALPHABET) for _ in range(length))
    return f"raga-token-{suffix}"


def get_or_create_api_key(user_id: str) -> str:
    """Return the user's existing (non-revoked) API key, creating one if absent.

    One active key per user. Generation retries on the rare suffix collision.
    """
    existing = (supabase.table("api_keys")
                .select("api_key")
                .eq("user_id", user_id)
                .eq("revoked", False)
                .limit(1)
                .execute().data or [])
    if existing:
        return existing[0]["api_key"]

    last_error: Exception | None = None
    for _ in range(5):
        api_key = _generate_api_key()
        try:
            supabase.table("api_keys").insert({
                "user_id": user_id,
                "api_key": api_key,
                "revoked": False,
            }).execute()
            return api_key
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            message = str(exc).lower()
            # Only a unique-constraint collision on api_key is worth retrying.
            if "duplicate" in message or "unique" in message or "23505" in message:
                continue
            # Anything else (e.g. RLS 42501, missing table) won't fix itself.
            break
    raise HTTPException(
        status_code=500,
        detail=f"Could not provision API key: {last_error}",
    )


def get_api_key_record(api_key: str):
    """Look up an API key. Returns the row (with user_id) or None."""
    rows = (supabase.table("api_keys")
            .select("user_id, api_key, revoked")
            .eq("api_key", api_key)
            .eq("revoked", False)
            .limit(1)
            .execute().data or [])
    return rows[0] if rows else None


def touch_api_key_last_used(api_key: str) -> None:
    try:
        (supabase.table("api_keys")
         .update({"last_used_at": datetime.now(timezone.utc).isoformat()})
         .eq("api_key", api_key)
         .execute())
    except Exception:
        # last_used_at is best-effort telemetry; never block a request on it.
        pass


def insert_application_logs(session_id, user_query, gpt_response, model,
                            citations=None, chunks=None):
    record = {
        "session_id": session_id,
        "user_query": user_query,
        "gpt_response": gpt_response,
        "model": model,
        "citations": citations or [],
        "chunks": chunks or [],
    }
    try:
        supabase.table("application_logs").insert(record).execute()
    except Exception:
        # `citations`/`chunks` columns may not exist yet on older schemas.
        # Retry without them so chat still persists.
        record.pop("citations", None)
        record.pop("chunks", None)
        supabase.table("application_logs").insert(record).execute()


def get_chat_history(session_id):
    try:
        rows = (supabase.table("application_logs")
                .select("user_query, gpt_response, citations, chunks")
                .eq("session_id", session_id)
                .order("created_at")
                .execute().data or [])
    except Exception:
        # Older schema without citations/chunks columns.
        rows = (supabase.table("application_logs")
                .select("user_query, gpt_response")
                .eq("session_id", session_id)
                .order("created_at")
                .execute().data or [])
    messages = []
    for row in rows:
        messages.append({"role": "user", "content": row["user_query"]})
        messages.append({
            "role": "assistant",
            "content": row["gpt_response"],
            "citations": row.get("citations") or [],
            "chunks": row.get("chunks") or [],
        })
    return messages


def insert_document_record(filename, filesize, session_id):
    result = supabase.table("document_store").insert({
        "filename": filename,
        "filesize": filesize,
        "session_id": session_id,
    }).execute()
    data = result.data or []
    return data[0]["id"] if data else None


def delete_document_record(file_id):
    supabase.table("document_store").delete().eq("id", file_id).execute()
    return True


def get_all_documents(session_id):
    return (supabase.table("document_store")
            .select("*")
            .eq("session_id", session_id)
            .order("upload_timestamp", desc=True)
            .execute().data or [])


def get_all_sessions():
    logs = (supabase.table("application_logs")
            .select("session_id, user_query, created_at")
            .order("created_at")
            .execute().data or [])
    kb = (supabase.table("session_knowledgebase")
          .select("session_id, knowledgebase_id")
          .execute().data or [])
    kb_map = {row["session_id"]: row["knowledgebase_id"] for row in kb}
    first_by_session = {}
    for row in logs:
        session_id = row["session_id"]
        if session_id not in first_by_session:
            first_by_session[session_id] = {
                "session_id": session_id,
                "user_query": row["user_query"],
                "knowledgebase_id": kb_map.get(session_id),
            }
    return list(first_by_session.values())


def delete_session(session_id):
    rows = (supabase.table("application_logs")
            .select("session_id")
            .eq("session_id", session_id)
            .limit(1)
            .execute().data or [])
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    supabase.table("application_logs").delete().eq("session_id", session_id).execute()
    return {"message": "Deleted successfully"}


# RAG management helper functions

def create_rag_entry(rag_id: str, name: str, description: str, top_k: float):
    supabase.table("rag_store").insert({
        "rag_id": rag_id,
        "name": name,
        "description": description,
        "top_k": top_k,
    }).execute()


def get_rag_entry(rag_id: str):
    rows = supabase.table("rag_store").select("*").eq("rag_id", rag_id).execute().data or []
    return rows[0] if rows else None


def delete_rag_entry(rag_id: str):
    supabase.table("rag_store").delete().eq("rag_id", rag_id).execute()
    return True


def list_rags():
    return (supabase.table("rag_store")
            .select("*")
            .order("created_at", desc=True)
            .execute().data or [])


def list_rags_for_user(user_id: str):
    """RAGs owned by `user_id` plus any legacy RAGs with no owner (user_id NULL).

    Option (a): NULL-owner RAGs (created before ownership existed) remain
    accessible to any authenticated key.
    """
    try:
        return (supabase.table("rag_store")
                .select("*")
                .or_(f"user_id.eq.{user_id},user_id.is.null")
                .order("created_at", desc=True)
                .execute().data or [])
    except Exception:
        # `user_id` column may not exist yet; fall back to all RAGs.
        return list_rags()


def assert_kb_access(user_id: str, knowledgebase_id: str) -> dict:
    """Ensure `user_id` may access `knowledgebase_id`; return the RAG row.

    Access is granted when the RAG is owned by the user OR has no owner
    (legacy NULL). Raises 404 if the RAG doesn't exist and 403 if it's owned
    by someone else.
    """
    rows = (supabase.table("rag_store")
            .select("*")
            .eq("rag_id", knowledgebase_id)
            .limit(1)
            .execute().data or [])
    if not rows:
        raise HTTPException(status_code=404, detail="Knowledgebase not found")
    rag = rows[0]
    owner = rag.get("user_id")
    if owner and owner != user_id:
        raise HTTPException(status_code=403, detail="You do not have access to this knowledgebase")
    return rag


# RAG-specific CRUD used by api/rag/endpoints.py

def get_all_rag_configs():
    configs = (supabase.table("rag_store")
            .select("*")
            .order("created_at")
            .execute().data or [])
    if not configs:
        return []

    # Optimize N+1 issues: fetch document records to calculate document counts in single query
    docs = (supabase.table("rag_document_store")
            .select("rag_id")
            .execute().data or [])

    counts = {}
    for doc in docs:
        r_id = doc.get("rag_id")
        if r_id:
            counts[r_id] = counts.get(r_id, 0) + 1

    for config in configs:
        config["document_count"] = counts.get(config["rag_id"], 0)

    return configs


def create_rag_record(rag_id: str, name: str, description: str, top_k: float,
                      chunk_size: int | None, embedding_model: str | None,
                      user_id: str | None = None):
    record = {
        "rag_id": rag_id,
        "name": name,
        "description": description,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "embedding_model": embedding_model,
    }
    if user_id is not None:
        record["user_id"] = user_id
    try:
        supabase.table("rag_store").insert(record).execute()
    except Exception:
        # `user_id` column may not exist yet on older schemas; retry without it.
        record.pop("user_id", None)
        supabase.table("rag_store").insert(record).execute()


def get_rag_record(rag_id: str):
    rows = (supabase.table("rag_store")
            .select("rag_id, name, description, top_k, chunk_size, embedding_model")
            .eq("rag_id", rag_id)
            .execute().data or [])
    return rows[0] if rows else None


def update_rag_record(rag_id: str, name: str | None = None, description: str | None = None):
    payload = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    supabase.table("rag_store").update(payload).eq("rag_id", rag_id).execute()
    return get_rag_record(rag_id)


def delete_rag_record(rag_id: str):
    supabase.table("rag_store").delete().eq("rag_id", rag_id).execute()
    supabase.table("rag_document_store").delete().eq("rag_id", rag_id).execute()
    supabase.table("session_knowledgebase").update(
        {"knowledgebase_id": None}
    ).eq("knowledgebase_id", rag_id).execute()
    return True


# Document store for each RAG

def insert_rag_document(rag_id: str, filename: str, filesize: int):
    result = supabase.table("rag_document_store").insert({
        "rag_id": rag_id,
        "filename": filename,
        "filesize": filesize,
    }).execute()
    data = result.data or []
    return data[0]["id"] if data else None


def get_rag_documents(rag_id: str):
    return (supabase.table("rag_document_store")
            .select("*")
            .eq("rag_id", rag_id)
            .order("upload_timestamp", desc=True)
            .execute().data or [])


def delete_rag_document(doc_id: int):
    supabase.table("rag_document_store").delete().eq("id", doc_id).execute()
    return True


# Session <-> knowledgebase mapping

def set_session_knowledgebase(session_id: str, knowledgebase_id: str | None):
    supabase.table("session_knowledgebase").upsert(
        {"session_id": session_id, "knowledgebase_id": knowledgebase_id},
        on_conflict="session_id",
    ).execute()


def get_session_knowledgebase(session_id: str) -> str | None:
    rows = (supabase.table("session_knowledgebase")
            .select("knowledgebase_id")
            .eq("session_id", session_id)
            .limit(1)
            .execute().data or [])
    return rows[0]["knowledgebase_id"] if rows else None
