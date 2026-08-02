# AGENTS.md — RAGA (Rag As A Service)

## Stack
- **FastAPI** + **LangChain** + **ChromaDB** + **OpenAI** for RAG chat
- **SQLite** (`rag_app.db`) for sessions, documents, logs
- **Supabase** JWT auth via `middlewares/auth_middleware.py`
- **watchfiles** for auto-reload in dev

## Key Files
- `main.py` — FastAPI app, `/chat`, `/upload-doc`, `/search-rag` endpoints
- `chroma_utils.py` — ChromaDB indexing, retrieval, deletion
- `db_utils.py` — SQLite helpers (sessions, documents, logs, knowledgebase mapping)
- `langchain_utils.py` — LangChain RAG chain (currently unused by `/chat`; `main.py` builds prompts manually)
- `api/rag/endpoints.py` — RAG management + proper document upload with `rag_id`
- `.env` — `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, LangChain tracing

## Running
```bash
source .venv/bin/activate
python main.py   # uvicorn on localhost:8080 with watchfiles auto-reload
```

## Architecture Notes
- **Knowledgebase ↔ RAG mapping**: `session_knowledgebase` table links `session_id` → `knowledgebase_id` (which is a `rag_id` from `rag_store`). The `/upload-doc/{session_id}` endpoint reads this and passes it as `rag_id` to ChromaDB. The `/chat` endpoint filters retrieval by this `rag_id`. If they don't match, retrieval returns 0 results.
- **Two upload paths**: `/upload-doc/{session_id}` (simple, session-aware) and `/rag/{rag_id}/documents` (RAG-specific, always passes `rag_id` correctly).
- **Streaming**: `/chat` uses SSE (`text/event-stream`) when the question has > 7 words. Short questions get a JSON `QueryResponse`.

## Known Bugs Fixed
1. `/upload-doc/{session_id}` was not passing `knowledgebase_id` to `index_document_to_chroma`, so chunks were stored with `rag_id: None` and retrieval always missed them.
2. `llm.invoke()` returns `AIMessage`, not `str` — must use `.content` before storing in SQLite or returning.
3. `get_chunks_from_chroma` returned `False` on error instead of `[]`, causing `TypeError: 'bool' object is not iterable` in the chat endpoint.
4. `session_knowledgebase.knowledgebase_id` must match `rag_store.rag_id` — a mismatch causes 0 retrieval results.

## Verification
- Check ChromaDB chunks have correct `rag_id`:
  ```python
  from chroma_utils import vectorstore
  vectorstore.get(where={"rag_id": "<knowledgebase_id>"})
  ```
- Check session's knowledgebase_id:
  ```bash
  sqlite3 rag_app.db "SELECT * FROM session_knowledgebase;"
  ```
- Check all RAGs:
  ```bash
  sqlite3 rag_app.db "SELECT * FROM rag_store;"
  ```