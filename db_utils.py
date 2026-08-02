import os
from datetime import datetime, timezone
from fastapi import HTTPException
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_JWT_SECRET")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def insert_application_logs(session_id, user_query, gpt_response, model):
    supabase.table("application_logs").insert({
        "session_id": session_id,
        "user_query": user_query,
        "gpt_response": gpt_response,
        "model": model,
    }).execute()


def get_chat_history(session_id):
    rows = (supabase.table("application_logs")
            .select("user_query, gpt_response")
            .eq("session_id", session_id)
            .order("created_at")
            .execute().data or [])
    messages = []
    for row in rows:
        messages.append({"role": "user", "content": row["user_query"]})
        messages.append({"role": "assistant", "content": row["gpt_response"]})
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


# RAG-specific CRUD used by api/rag/endpoints.py

def get_all_rag_configs():
    return (supabase.table("rag_store")
            .select("*")
            .order("created_at")
            .execute().data or [])


def create_rag_record(rag_id: str, name: str, description: str, top_k: float,
                      chunk_size: int | None, embedding_model: str | None):
    supabase.table("rag_store").insert({
        "rag_id": rag_id,
        "name": name,
        "description": description,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "embedding_model": embedding_model,
    }).execute()


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
