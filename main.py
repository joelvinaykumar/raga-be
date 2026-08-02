import os
import json
import logging
import uuid
import shutil

from fastapi import FastAPI, File, Request, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from api.rag import endpoints as rag_router
from chroma_utils import index_document_to_chroma, get_chunks_from_chroma, delete_doc_from_chroma
from db_utils import insert_application_logs, get_chat_history, get_all_documents, get_all_sessions, delete_session, insert_document_record, delete_document_record, set_session_knowledgebase, get_session_knowledgebase
from middlewares.auth_middleware import JWTBearer
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, SessionInfo, DeleteFileRequest, ModifySessionRequest


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


try:
    logging.basicConfig(filename='app.log', level=logging.INFO)
except OSError:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="RAGA - Rag As A Service",
    description="Create decentralized RAGs and plug them into your agents or LLMs. This bad boy will serach for the most relevant chunks from ChromaDB",
    swagger_ui_parameters={"docExpansion": "none"}
)
favicon_path = 'raga-favicon.png'

origins = [
    "http://localhost:5173",  # Example frontend URL
    "https://raga-fe.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Set to True if your frontend sends cookies/credentials
    allow_methods=["*"],     # Allow all standard HTTP methods
    allow_headers=["*"],     # Allow all standard HTTP headers
)
app.include_router(rag_router.router, prefix="/rag", tags=["RAG Config"], dependencies=[Depends(JWTBearer())])

@app.get("/")
def health():
    return {"message": "Hello World"}

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

def stream_chat_response(prompt, session_id, question, model_value):
    llm = ChatOpenAI(model=model_value, temperature=0.7, api_key=OPENAI_API_KEY)
    full_response = ""
    for chunk in llm.stream(prompt):
        content = chunk.content
        if content:
            full_response += content
            yield f"data: {json.dumps({'content': content})}\n\n"
    insert_application_logs(session_id, question, full_response, model_value)
    logging.info(f"Session ID: {session_id}, AI Response: {full_response}")
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    knowledgebase_id = get_session_knowledgebase(session_id)
    chat_history = get_chat_history(session_id)

    if knowledgebase_id:
        chunks = get_chunks_from_chroma(query_input.question, knowledgebase_id)
        context = "\n\n".join([c["content"] for c in chunks if c])
        llm = ChatOpenAI(model=query_input.model.value, temperature=0.7, api_key=OPENAI_API_KEY)
        prompt = (
            f"Use the following context to answer the question. "
            f"If the context does not contain relevant information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query_input.question}"
        )
        if chat_history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
            prompt = f"Chat history:\n{history_str}\n\n{prompt}"
        if len(query_input.question.split()) > 7 or len(prompt.split()) > 7:
            return StreamingResponse(
                stream_chat_response(prompt, session_id, query_input.question, query_input.model.value),
                media_type="text/event-stream"
            )
        answer = llm.invoke(prompt).content
    else:
        llm = ChatOpenAI(model=query_input.model.value, temperature=0.7, api_key=OPENAI_API_KEY)
        if len(query_input.question.split()) > 7:
            return StreamingResponse(
                stream_chat_response(query_input.question, session_id, query_input.question, query_input.model.value),
                media_type="text/event-stream"
            )
        answer = llm.invoke(query_input.question).content
        chunks = []

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model, chunks=chunks)

@app.get("/chat-history/{session_id}")
def retrieve_chat_history(session_id: str):
    return get_chat_history(session_id)

@app.post("/upload-doc/{session_id}")
def upload_and_index_document(session_id: str, file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
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
def list_documents(session_id: str):
    return get_all_documents(session_id)

@app.get("/search-rag")
def serch_rag(query: str):
    return get_chunks_from_chroma(query)

@app.get("/list-sessions", response_model=list[SessionInfo])
def list_sessions():
    return get_all_sessions()

@app.delete("/delete-session/{session_id}")
def delete_session_endpoint(session_id: str):
    return delete_session(session_id)

@app.put("/sessions/{session_id}/knowledgebase")
def modify_session_knowledgebase(session_id: str, request: ModifySessionRequest):
    set_session_knowledgebase(session_id, request.knowledgebase_id)
    return {"session_id": session_id, "knowledgebase_id": request.knowledgebase_id}


@app.delete("/delete-doc")
def delete_document(request: DeleteFileRequest):
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