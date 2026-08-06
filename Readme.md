# RAGA — Rag As A Service

A production-ready **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, **LangChain**, and **ChromaDB**. Upload documents, organize them into knowledge bases (RAGs), and query them through an LLM — with per-session history, streaming responses, and multi-tenant isolation.

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Request Flow](#request-flow)
- [API Endpoints](#api-endpoints)
- [Data Model](#data-model)
- [Local Development](#local-development)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)

---

## Architecture

```
                ┌───────────────────────────────────────────────────┐
                │                  Frontend (raga-fe)                │
                └───────────────────────┬───────────────────────────┘
                                        │ HTTPS / JWT Bearer
                                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application (main.py)                    │
│                                                                         │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────────────┐   │
│  │ Auth       │   │ /chat        │   │ /upload-doc · /rag/*         │   │
│  │ Middleware │   │ (SSE stream) │   │ /rag/* · /chat-history · ... │   │
│  └────────────┘   └──────────────┘   └──────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        LangChain / OpenAI                        │  │
│  │              ChatOpenAI · OpenAIEmbeddings · Splitters           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
        ┌──────────────────┐              ┌──────────────────┐
        │      ChromaDB    │              │      SQLite      │
        │  Vector storage  │              │ (rag_app.db)     │
        │  (cloud or local)│              │ Sessions · Docs  │
        └──────────────────┘              │ Logs · RAG store │
                                          └──────────────────┘
```

The service is split into four layers:

1. **API layer** (`main.py`, `api/`) — HTTP endpoints, request/response models, routing.
2. **Service layer** (`chroma_utils.py`, `langchain_utils.py`, `db_utils.py`) — document ingestion, vector retrieval, chat completions, persistence.
3. **Middleware** (`middlewares/`) — Supabase JWT authentication and request logging.
4. **Data layer** — ChromaDB for vectors, SQLite for relational metadata.

## Project Structure

```
.
├── main.py                     # FastAPI app: /chat, /upload-doc, /rag/*, session routes
├── chroma_utils.py             # ChromaDB: indexing, retrieval, deletion; local + Chroma Cloud
├── db_utils.py                 # SQLite helpers: sessions, documents, logs, knowledgebase mapping
├── langchain_utils.py          # LangChain RAG chain (retrieval-qa with history)
├── pydantic_models.py          # Pydantic models for chat & document APIs
├── schemas.py                  # Request/response schemas for the RAG management API
├── middlewares/
│   ├── auth_middleware.py      # Supabase JWT bearer auth
│   └── middleware.py           # Request logging + CORS + trusted host
├── api/
│   ├── rag/
│   │   ├── endpoints.py        # /rag/* — RAG lifecycle + document management
│   │   └── models.py           # RAG store models
│   └── parser/
│       └── endpoints.py        # (reserved) parser endpoints
├── requirements.txt            # Pinned dependencies (resolved via uv)
├── vercel.json                 # Vercel function config (maxDuration, entrypoint)
└── runtime.txt                 # Python runtime version
```

## Request Flow

### 1. Chat (`POST /chat`)

```
Request ──► JWT auth ──► resolve session
                │
                ├── knowledgebase_id?  (session_knowledgebase → rag_store)
                │        │ yes
                │        ▼
                │   get_chunks_from_chroma(query, rag_id)  ──► ChromaDB
                │        │
                │        ▼
                │   build context + chat history ──► ChatOpenAI
                │        │
                │        ▼
                └── question > 7 words? ──► SSE streaming response
                     else ──► JSON QueryResponse
```

- Short questions return a JSON `QueryResponse`; longer questions stream via SSE (`text/event-stream`).
- Retrieval is filtered by the session's `knowledgebase_id` (`rag_id`) — a mismatch returns zero chunks.

### 2. Document Upload (`POST /upload-doc/{session_id}`)

```
Upload (pdf / docx / html)
    │
    ▼
save to temp file ──► insert_document_record ──► resolve session's rag_id
    │                                              │
    ▼                                              ▼
load_and_split_document ──► ChromaDB (chunks tagged with rag_id)
```

Two upload paths exist:

- `/upload-doc/{session_id}` — session-aware; reads the session's knowledgebase mapping.
- `/rag/{rag_id}/documents` — RAG-specific; always tags chunks with the correct `rag_id`.

## API Endpoints

| Method | Path                          | Description                                  |
| ------ | ----------------------------- | -------------------------------------------- |
| POST   | `/chat`                       | Ask a question (JSON or SSE stream)          |
| POST   | `/upload-doc/{session_id}`    | Upload & index a document                    |
| GET    | `/chat-history/{session_id}`  | Chat history for a session                   |
| GET    | `/list-docs/{session_id}`     | Documents belonging to a session             |
| GET    | `/list-sessions`              | All sessions                                 |
| DELETE | `/delete-session/{session_id}`| Delete a session                             |
| PUT    | `/sessions/{id}/knowledgebase`| Bind/unbind a session to a knowledgebase     |
| DELETE | `/delete-doc`                 | Remove a document (Chroma + SQLite)          |
| GET    | `/rag/all/`                   | List all RAG knowledge bases                 |
| POST   | `/rag/`                       | Create a RAG                                 |
| GET    | `/rag/{rag_id}`               | RAG details                                  |
| PATCH  | `/rag/{rag_id}`               | Update a RAG                                 |
| DELETE | `/rag/{rag_id}`               | Delete a RAG (store + vectors)               |
| POST   | `/rag/{rag_id}/documents`     | Upload documents to a RAG                    |
| GET    | `/rag/{rag_id}/documents`     | List a RAG's documents                       |
| DELETE | `/rag/{rag_id}/documents/{id}`| Delete one of a RAG's documents              |

## Data Model

```sql
-- Rag metadata (multi-tenant knowledge bases)
rag_store(id, rag_id, name, description, top_k, chunk_size, embedding_model, created_at, updated_at)

-- Documents uploaded per RAG
rag_document_store(id, rag_id, document_name, ...)

-- Session ↔ knowledgebase mapping (drives retrieval scoping)
session_knowledgebase(session_id, knowledgebase_id)

-- Chat logs & sessions
application_logs(id, session_id, user_query, gpt_response, model, created_at)
```

The `session_knowledgebase` table links a `session_id` to a `knowledgebase_id` (the `rag_id` from `rag_store`). Chroma chunks store the same `rag_id` as metadata so retrieval can be scoped per knowledge base.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # set OPENAI_API_KEY, SUPABASE_URL, SUPABASE_JWT_SECRET

python main.py          # uvicorn on localhost:8080 with watchfiles auto-reload
```

Interactive API docs are served at `http://localhost:8080/docs`.

## Deployment

This service deploys to **Vercel** as a single serverless function (`main.py`), configured in `vercel.json`. On Vercel the filesystem is read-only outside `/tmp`, so Chroma and SQLite data are ephemeral unless you use managed storage:

- **Vectors** — set `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` to use managed **Chroma Cloud**. Without them the app falls back to a local persistent collection.
- **Sessions/logs** — SQLite is ephemeral on serverless; wire in a hosted Postgres for production durability.

For a persistent server runtime, the app runs as a standard FastAPI service (uvicorn) on hosts like **Railway**, which is highly recommended for streaming responses (SSE).

---

## Model Context Protocol (MCP) Server

RAGA mounts a streamable-HTTP **Model Context Protocol (MCP)** server at `/mcp` (powered by FastMCP). This allows external LLM clients, such as **Claude Desktop**, Cursor, or custom agents, to connect directly to your personal knowledge bases and invoke retrieval capabilities as native skills.

### 🔑 Authentication
The MCP layer uses separate API keys from standard REST endpoints.
- Authenticators are passed in the `x-api-key` header.
- Pattern: `raga-token-[6-8 alphanumerics]`.
- API keys are automatically generated when a user logs in and calls `/me`, meaning zero-friction key provisioning.

### 🔌 Tools Exposed
The server expands Claude's abilities with three custom tools:
1. `list_knowledgebases()`: Lists RAG workspaces available to the authenticated token.
2. `search_rag(knowledgebase_id, query, top_k)`: Perform semantic vector matching over specific documents and get raw background context chunks (without calling LLMs).
3. `ask_rag(knowledgebase_id, query)`: Answer questions grounded strictly in a knowledge base with integrated citations and sources.

### ⚙️ Claude Desktop Setup
Add this entry to your Claude configuration file (located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "raga": {
      "type": "sse",
      "url": "https://your-raga-backend.railway.app/mcp",
      "headers": {
        "x-api-key": "raga-token-your_generated_key_here"
      }
    }
  }
}
```

---

## Deployment to Railway (Recommended)

Railway is excellent for deploying RAGA because its persistent container runtime fully supports long-lived Server-Sent Events (SSE) connections and doesn't suffer from Serverless Function timeouts (like Vercel).

### Detailed Steps:
1. **Prepare Database Migrations**:
   Ensure you have executed the following columns in your Supabase SQL editor:
   ```sql
   -- Enable api_keys
   create table if not exists api_keys (
     id uuid primary key default gen_random_uuid(),
     user_id uuid not null references auth.users(id) on delete cascade,
     api_key text not null unique,
     revoked boolean not null default false,
     created_at timestamptz not null default now(),
     last_used_at timestamptz
   );
   alter table api_keys disable row level security;
   create unique index if not exists api_keys_active_user on api_keys (user_id) where revoked = false;

   -- Add citations to logs
   alter table application_logs add column if not exists citations jsonb default '[]'::jsonb;
   alter table application_logs add column if not exists chunks jsonb default '[]'::jsonb;
   
   -- Add tenant user mapping to vector directories
   alter table rag_store add column if not exists user_id uuid references auth.users(id) on delete set null;
   ```

2. **Deploy Backend**:
   - Link the `raga-be` directory to a new Railway service.
   - Set the Start Command to:
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - Feed the following environment variables into Railway:
     - `OPENAI_API_KEY`: Your OpenAI credential.
     - `SUPABASE_URL`: Your Supabase API host.
     - `SUPABASE_JWT_SECRET`: Supabase JWT parser key.

3. **Deploy Frontend**:
   - Link `raga-fe` as a separate Railway or static-serving node.
   - Inject `VITE_BASE_URL` pointing to the deployed backend's HTTPS endpoint.

---

## Environment Variables

| Variable              | Required | Description                                |
| --------------------- | -------- | ------------------------------------------ |
| `OPENAI_API_KEY`      | Yes      | OpenAI API key (chat + embeddings)         |
| `SUPABASE_URL`        | Yes      | Supabase project URL (JWT auth)            |
| `SUPABASE_JWT_SECRET` | Yes      | Supabase anon/JWT secret                   |
| `CHROMA_API_KEY`      | No       | Chroma Cloud API key (managed vector store)|
| `CHROMA_TENANT`       | No       | Chroma Cloud tenant ID                     |
| `CHROMA_DATABASE`     | No       | Chroma Cloud database name                 |
| `LANGCHAIN_API_KEY`   | No       | LangSmith tracing                          |
