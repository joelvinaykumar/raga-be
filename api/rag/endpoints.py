import uuid
from fastapi import APIRouter, HTTPException, status

from .models import RagCreateModel, RagResponseModel, RagDetailModel, RagPatchModel
from fastapi import File, UploadFile
from typing import List
from schemas import RagCreateRequest, RagResponse, DocumentUploadResponse
from db_utils import get_all_rag_configs, create_rag_record, get_rag_record, update_rag_record, delete_rag_record
import os, shutil, tempfile
router = APIRouter()

@router.get("/all/", status_code=status.HTTP_200_OK)
@router.get("/all", status_code=status.HTTP_200_OK)
async def get_all_configs():
    return get_all_rag_configs()

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_rag(input: RagCreateModel):
    if input.top_k <= 0:
        raise HTTPException(status_code=400, detail="Top K must be a positive integer")
    rag_id = str(uuid.uuid4())
    create_rag_record(rag_id, input.name, input.description, input.top_k, input.chunk_size, input.embedding_model)
    return {"rag_id": rag_id, "message": "RAG created"}

@router.get("/{rag_id}", response_model=RagDetailModel, status_code=status.HTTP_200_OK)
async def get_rag(rag_id: str):
    rag = get_rag_record(rag_id)
    if not rag:
        raise HTTPException(status_code=404, detail="RAG not found")
    return rag

@router.patch("/{rag_id}", response_model=RagDetailModel, status_code=status.HTTP_200_OK)
async def update_rag(rag_id: str, input: RagPatchModel):
    existing = get_rag_record(rag_id)
    if not existing:
        raise HTTPException(status_code=404, detail="RAG not found")
    name = input.name if input.name is not None else existing["name"]
    description = input.description if input.description is not None else existing["description"]
    return update_rag_record(rag_id, name, description)

@router.delete("/{rag_id}", response_model=RagResponseModel, status_code=status.HTTP_200_OK)
async def delete_rag(rag_id: str):
    from chroma_utils import delete_rag_from_chroma
    if not get_rag_record(rag_id):
        raise HTTPException(status_code=404, detail="RAG not found")
    delete_rag_record(rag_id)
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