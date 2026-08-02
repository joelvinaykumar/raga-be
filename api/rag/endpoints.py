import uuid
from fastapi import APIRouter, HTTPException, status

from .models import RagCreateModel, RagResponseModel, RagDetailModel, RagPatchModel
from fastapi import File, UploadFile
from typing import List
from schemas import RagCreateRequest, RagResponse, DocumentUploadResponse
from db_utils import get_db_connection
import os, shutil, tempfile
router = APIRouter()

@router.get("/all/", status_code=status.HTTP_200_OK)
@router.get("/all", status_code=status.HTTP_200_OK)
async def get_all_configs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM rag_store ORDER BY created_at')
    stores = cursor.fetchall()
    conn.close()
    return stores

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_rag(input: RagCreateModel):
    if input.top_k <= 0:
        raise HTTPException(status_code=400, detail="Top K must be a positive integer")
    # Ensure rag_store has chunk_size and embedding_model columns (SQLite ALTER TABLE)
    conn = get_db_connection()
    try:
        conn.execute('ALTER TABLE rag_store ADD COLUMN chunk_size INTEGER')
    except Exception:
        pass
    try:
        conn.execute('ALTER TABLE rag_store ADD COLUMN embedding_model TEXT')
    except Exception:
        pass
    conn.commit()
    rag_id = str(uuid.uuid4())
    conn.execute('INSERT INTO rag_store (rag_id, name, description, top_k, chunk_size, embedding_model) VALUES (?, ?, ?, ?, ?, ?)',
                 (rag_id, input.name, input.description, input.top_k, input.chunk_size, input.embedding_model))
    conn.commit()
    conn.close()
    return {"rag_id": rag_id, "message": "RAG created"}

@router.get("/{rag_id}", response_model=RagDetailModel, status_code=status.HTTP_200_OK)
async def get_rag(rag_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT rag_id, name, description, top_k, chunk_size, embedding_model FROM rag_store WHERE rag_id = ?', (rag_id,))
    rag = cursor.fetchone()
    conn.close()
    if not rag:
        raise HTTPException(status_code=404, detail="RAG not found")
    return {
        "rag_id": rag[0],
        "name": rag[1],
        "description": rag[2],
        "top_k": rag[3],
        "chunk_size": rag[4],
        "embedding_model": rag[5],
    }

@router.patch("/{rag_id}", response_model=RagDetailModel, status_code=status.HTTP_200_OK)
async def update_rag(rag_id: str, input: RagPatchModel):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT name, description FROM rag_store WHERE rag_id = ?', (rag_id,))
    rag = cursor.fetchone()
    if not rag:
        conn.close()
        raise HTTPException(status_code=404, detail="RAG not found")
    name = input.name if input.name is not None else rag[0]
    description = input.description if input.description is not None else rag[1]
    conn.execute('UPDATE rag_store SET name = ?, description = ?, updated_at = CURRENT_TIMESTAMP WHERE rag_id = ?',
                 (name, description, rag_id))
    conn.commit()
    cursor.execute('SELECT rag_id, name, description, top_k, chunk_size, embedding_model FROM rag_store WHERE rag_id = ?', (rag_id,))
    row = cursor.fetchone()
    conn.close()
    return {
        "rag_id": row[0],
        "name": row[1],
        "description": row[2],
        "top_k": row[3],
        "chunk_size": row[4],
        "embedding_model": row[5],
    }

@router.delete("/{rag_id}", response_model=RagResponseModel, status_code=status.HTTP_200_OK)
async def delete_rag(rag_id: str):
    from chroma_utils import delete_rag_from_chroma
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT rag_id FROM rag_store WHERE rag_id = ?', (rag_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="RAG not found")
    conn.execute('DELETE FROM rag_store WHERE rag_id = ?', (rag_id,))
    conn.execute('DELETE FROM rag_document_store WHERE rag_id = ?', (rag_id,))
    conn.execute('UPDATE session_knowledgebase SET knowledgebase_id = NULL WHERE knowledgebase_id = ?', (rag_id,))
    conn.commit()
    conn.close()
    delete_rag_from_chroma(rag_id)
    return {"message": "RAG store deleted successfully", "rag_id": rag_id}

# Document management endpoints
@router.post("/{rag_id}/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(rag_id: str, files: List[UploadFile] = File(...)):
    # Save uploaded files temporarily, insert DB record, and index into Chroma
    from db_utils import insert_rag_document, delete_rag_document
    from chroma_utils import index_document_to_chroma
    import os, shutil, tempfile
    uploaded = []
    for upload in files:
        # Write to temp file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, upload.filename)
        content = await upload.read()
        with open(file_path, "wb") as f:
            f.write(content)
        # Insert DB record
        doc_id = insert_rag_document(rag_id, upload.filename, len(content))
        # Index document
        success = index_document_to_chroma(file_path, doc_id, rag_id)
        # Cleanup
        shutil.rmtree(temp_dir)
        if not success:
            delete_rag_document(doc_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {upload.filename} into Chroma.")
        uploaded.append({"document_id": doc_id, "filename": upload.filename})
    # Return first uploaded document info
    first = uploaded[0]
    return DocumentUploadResponse(rag_id=rag_id, document_id=first["document_id"], filename=first["filename"])

@router.get("/{rag_id}/documents", response_model=List[dict], status_code=status.HTTP_200_OK)
async def list_documents(rag_id: str):
    from db_utils import get_rag_documents
    return get_rag_documents(rag_id)

@router.delete("/{rag_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(rag_id: str, doc_id: int):
    from db_utils import delete_rag_document
    from chroma_utils import delete_doc_from_chroma
    delete_rag_document(doc_id)
    delete_doc_from_chroma(doc_id)
    return {"message": "Document deleted"}