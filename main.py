import os
import json
import logging
import uuid
import shutil
import tempfile
import re
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from api.rag import endpoints as rag_router
from chroma_utils import index_document_to_chroma, get_chunks_from_chroma, delete_doc_from_chroma
from db_utils import insert_application_logs, get_chat_history, get_all_documents, get_all_sessions, delete_session, insert_document_record, delete_document_record, set_session_knowledgebase, get_session_knowledgebase, get_rag_record, get_or_create_api_key
from middlewares.auth_middleware import JWTBearer, get_current_user
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, SessionInfo, DeleteFileRequest, ModifySessionRequest


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


try:
    logging.basicConfig(filename='app.log', level=logging.INFO)
except OSError:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The MCP streamable-HTTP transport needs its session manager running for the
# lifetime of the app, so we drive it from the FastAPI lifespan.
from contextlib import asynccontextmanager
from mcp_server import mcp as mcp_server_instance, build_mcp_asgi_app


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    async with mcp_server_instance.session_manager.run():
        yield


app = FastAPI(
    title="RAGA - Rag As A Service",
    description="Create decentralized RAGs and plug them into your agents or LLMs. This bad boy will serach for the most relevant chunks from ChromaDB",
    swagger_ui_parameters={"docExpansion": "none"},
    lifespan=lifespan,
)
favicon_path = 'raga-favicon.png'

origins = [
    "http://localhost:5173",  # Example frontend URL
    "https://raga-fe.up.railway.app",
    "https://raga-fe.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Set to True if your frontend sends cookies/credentials
    allow_methods=["*"],     # Allow all standard HTTP methods
    allow_headers=["*"],     # Allow all standard HTTP headers
)
app.include_router(rag_router.router, prefix="/rag", tags=["RAG Config"], dependencies=[Depends(JWTBearer())])

# Mount the MCP server (streamable HTTP) at /mcp. Auth is enforced inside the
# wrapper via the `x-api-key` header, independent of the REST JWT auth.
app.mount("/mcp", build_mcp_asgi_app())

REMOVED_ROUTES = {"/search-rag"}


@app.middleware("http")
async def monitor_removed_routes(request: Request, call_next):
    if request.url.path in REMOVED_ROUTES:
        client_host = request.client.host if request.client else "unknown"
        logger.warning(
            "Removed route access attempt path=%s method=%s client_ip=%s",
            request.url.path,
            request.method,
            client_host,
        )
    return await call_next(request)

@app.get("/")
def home():
    return {"message": "Hello from RAGA Backend - RAG As A Service is fully operational!"}


@app.get("/health")
def health_endpoint():
    """Verify backend health, checking database connection and ChromaDB vectorstore."""
    db_status = "unknown"
    vector_status = "unknown"

    try:
        from db_utils import supabase
        # Test Supabase connection with a simple count select
        supabase.table("rag_store").select("id", count="exact").limit(1).execute()
        db_status = "connected"
    except Exception as e:
        logger.error("Healthcheck database error: %s", str(e))
        db_status = f"unhealthy: {str(e)}"

    try:
        from chroma_utils import vectorstore
        # Test Chroma DB heartbeat
        vectorstore._client.heartbeat()
        vector_status = "connected"
    except Exception as e:
        logger.error("Healthcheck vectorstore error: %s", str(e))
        vector_status = f"unhealthy: {str(e)}"

    is_healthy = db_status == "connected" and vector_status == "connected"
    status_code = 200 if is_healthy else 500

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "database": db_status,
            "vectorstore": vector_status,
        }
    )


@app.get("/me")
def current_user(user: dict = Depends(get_current_user)):
    """Return the authenticated user and ensure they have an MCP API key.

    The key is provisioned on first call and reused thereafter, so clients
    (and the frontend) never have to create one manually.
    """
    api_key = get_or_create_api_key(user["user_id"])
    return {
        "user_id": user["user_id"],
        "email": user.get("email"),
        "api_key": api_key,
    }

@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return FileResponse(favicon_path)

@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"Unexpected error: {str(exc)}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
        }
    )


def _normalize_preview(text: str, max_len: int = 180) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= max_len else f"{normalized[:max_len - 1]}…"


def _build_source_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_chunks: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        content = chunk.get("content") or ""
        source_chunks.append({
            "index": idx,
            "chunk_id": chunk.get("chunk_id") or f"chunk-{idx}",
            "chunk_index": chunk.get("chunk_index"),
            "file_id": chunk.get("file_id"),
            "filename": chunk.get("filename"),
            "score": chunk.get("score"),
            "source": chunk.get("source"),
            "url": chunk.get("url") or chunk.get("source"),
            "content": content,
            "preview": _normalize_preview(chunk.get("preview") or content),
        })
    return source_chunks


_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _text_overlap_ratio(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    if not left_tokens:
        return 0.0
    right_tokens = _tokenize(right)
    if not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_top_k(requested_top_k: int | None, knowledgebase_id: str | None) -> int:
    if requested_top_k is not None:
        return max(1, min(int(requested_top_k), 30))
    if not knowledgebase_id:
        return 8
    rag = get_rag_record(knowledgebase_id)
    if not rag:
        return 8
    rag_top_k = rag.get("top_k")
    try:
        return max(1, min(int(rag_top_k), 30))
    except (TypeError, ValueError):
        return 8


def _sentence_spans(answer: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?\n]+|$)", answer):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + raw.find(stripped)
        end = start + len(stripped)
        spans.append((start, end))
    if not spans and answer.strip():
        stripped = answer.strip()
        start = answer.find(stripped)
        spans.append((start, start + len(stripped)))
    return spans


def _build_citations(answer: str, source_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not answer or not source_chunks:
        return []

    spans = _sentence_spans(answer)
    if not spans:
        return []

    citations: list[dict[str, Any]] = []

    min_span_overlap = 0.08
    used_pairs: set[tuple[int, int]] = set()

    for start_char, end_char in spans:
        sentence = answer[start_char:end_char]
        ranked_sources: list[tuple[float, int, dict[str, Any]]] = []

        for source_idx, source in enumerate(source_chunks):
            source_text = source.get("content") or source.get("preview") or ""
            overlap = _text_overlap_ratio(sentence, source_text)
            retrieval_score = _to_float(source.get("score"), default=0.0)
            blended = (0.7 * overlap) + (0.3 * retrieval_score)
            ranked_sources.append((blended, source_idx, source))

        if not ranked_sources:
            continue

        ranked_sources.sort(key=lambda item: item[0], reverse=True)
        best_score, best_source_idx, source = ranked_sources[0]

        # Skip weak matches to avoid attaching irrelevant citations.
        # Always keep at least one citation if the source itself has a strong retrieval score.
        if best_score < min_span_overlap and _to_float(source.get("score"), 0.0) < 0.5:
            continue

        pair_key = (best_source_idx, start_char)
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        citations.append({
            "index": len(citations) + 1,
            "start_char": start_char,
            "end_char": end_char,
            "display_char": end_char,
            "chunk_id": source.get("chunk_id"),
            "file_id": source.get("file_id"),
            "filename": source.get("filename"),
            "quote": source.get("preview"),
            "score": source.get("score"),
            "url": source.get("url"),
        })

    return citations


def stream_chat_response(prompt, session_id, question, model_value, source_chunks=None):
    llm = ChatOpenAI(model=model_value, temperature=0.7, api_key=OPENAI_API_KEY, timeout=15.0)
    full_response = ""
    yield f"data: {json.dumps({'chunks': source_chunks or []})}\n\n"
    for chunk in llm.stream(prompt):
        content = chunk.content
        if content:
            full_response += content
            yield f"data: {json.dumps({'content': content})}\n\n"
    citations = _build_citations(full_response, source_chunks or [])
    insert_application_logs(
        session_id, question, full_response, model_value,
        citations=citations, chunks=source_chunks or [],
    )
    logging.info("Session ID: %s, AI response chars: %s", session_id, len(full_response))
    yield f"data: {json.dumps({'done': True, 'citations': citations, 'chunks': source_chunks or []})}\n\n"


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput, _token: str = Depends(JWTBearer())):
    session_id = query_input.session_id
    logging.info("Session ID: %s, User query chars: %s, Model: %s", session_id, len(query_input.question), query_input.model.value)
    if not session_id:
        session_id = str(uuid.uuid4())

    # Prefer the knowledgebase_id sent explicitly in the request body; fall back
    # to the session<->knowledgebase binding only when it isn't provided.
    knowledgebase_id = query_input.knowledgebase_id or get_session_knowledgebase(session_id)
    top_k = _resolve_top_k(query_input.top_k, knowledgebase_id)
    chat_history = get_chat_history(session_id)

    stream_mode = query_input.stream

    if not knowledgebase_id:
        logging.warning(
            "Session ID: %s has NO knowledgebase (none in request body or binding) — "
            "chunks/citations will be empty. Pass knowledgebase_id in the /chat body "
            "or bind it via PUT /sessions/%s/knowledgebase.",
            session_id, session_id,
        )

    if knowledgebase_id:
        chunks = get_chunks_from_chroma(query_input.question, knowledgebase_id, top_k=top_k)
        context = "\n\n".join([c["content"] for c in chunks if c])
        source_chunks = _build_source_chunks(chunks)
        logging.info(
            "Session ID: %s, knowledgebase_id=%s, top_k=%s, retrieved_chunks=%s",
            session_id, knowledgebase_id, top_k, len(source_chunks),
        )
        llm = ChatOpenAI(model=query_input.model.value, temperature=0.7, api_key=OPENAI_API_KEY, timeout=15.0)
        prompt = (
            f"Use the following context to answer the question. "
            f"If the context does not contain relevant information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query_input.question}"
        )
        if chat_history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
            prompt = f"Chat history:\n{history_str}\n\n{prompt}"

        should_stream = stream_mode if stream_mode is not None else (
            len(query_input.question.split()) > 7 or len(prompt.split()) > 7
        )

        if should_stream:
            return StreamingResponse(
                stream_chat_response(
                    prompt,
                    session_id,
                    query_input.question,
                    query_input.model.value,
                    source_chunks,
                ),
                media_type="text/event-stream"
            )
        answer = llm.invoke(prompt).content
        citations = _build_citations(answer, source_chunks)
    else:
        llm = ChatOpenAI(model=query_input.model.value, temperature=0.7, api_key=OPENAI_API_KEY, timeout=15.0)
        chunks = []
        source_chunks = []
        should_stream = stream_mode if stream_mode is not None else len(query_input.question.split()) > 7
        if should_stream:
            return StreamingResponse(
                stream_chat_response(
                    query_input.question,
                    session_id,
                    query_input.question,
                    query_input.model.value,
                    source_chunks,
                ),
                media_type="text/event-stream"
            )
        answer = llm.invoke(query_input.question).content
        citations = []

    insert_application_logs(
        session_id, query_input.question, answer, query_input.model.value,
        citations=citations, chunks=source_chunks,
    )
    logging.info("Session ID: %s, AI response chars: %s", session_id, len(answer))
    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=query_input.model,
        chunks=source_chunks,
        citations=citations,
    )

@app.get("/chat-history/{session_id}")
def retrieve_chat_history(session_id: str, _token: str = Depends(JWTBearer())):
    return get_chat_history(session_id)

@app.post("/upload-doc/{session_id}")
def upload_and_index_document(session_id: str, file: UploadFile = File(...), _token: str = Depends(JWTBearer())):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    fd, temp_file_path = tempfile.mkstemp(suffix=file_extension)
    os.close(fd)

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = insert_document_record(file.filename, file.size, session_id)
        knowledgebase_id = get_session_knowledgebase(session_id)
        success = index_document_to_chroma(temp_file_path, file_id, knowledgebase_id)
        
        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs/{session_id}", response_model=list[DocumentInfo])
def list_documents(session_id: str, _token: str = Depends(JWTBearer())):
    return get_all_documents(session_id)

@app.get("/list-sessions", response_model=list[SessionInfo])
def list_sessions(_token: str = Depends(JWTBearer())):
    return get_all_sessions()

@app.delete("/delete-session/{session_id}")
def delete_session_endpoint(session_id: str, _token: str = Depends(JWTBearer())):
    return delete_session(session_id)

@app.put("/sessions/{session_id}/knowledgebase")
def modify_session_knowledgebase(session_id: str, request: ModifySessionRequest, _token: str = Depends(JWTBearer())):
    set_session_knowledgebase(session_id, request.knowledgebase_id)
    return {"session_id": session_id, "knowledgebase_id": request.knowledgebase_id}


@app.delete("/delete-doc")
def delete_document(request: DeleteFileRequest, _token: str = Depends(JWTBearer())):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=int(os.getenv("PORT", default=8080)), reload=True)