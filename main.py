from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, SessionInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, get_all_sessions, delete_session, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma,get_chunks_from_chroma, delete_doc_from_chroma
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import logging


logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

origins = [
    "http://localhost:5173",  # Example frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Set to True if your frontend sends cookies/credentials
    allow_methods=["*"],     # Allow all standard HTTP methods
    allow_headers=["*"],     # Allow all standard HTTP headers
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"Unexpected error: {str(exc)}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
        }
    )

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']
    
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

@app.get("/chat-history/{session_id}")
def retrieve_chat_history(session_id: str):
    return get_chat_history(session_id)

from fastapi import UploadFile, File, HTTPException
import os
import shutil

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
        success = index_document_to_chroma(temp_file_path, file_id)
        
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
def list_sessions(session_id: str):
    return delete_session(session_id)

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