from pydantic import BaseModel, Field

class RagCreateRequest(BaseModel):
    name: str = Field(..., description="Name of the RAG collection")
    description: str | None = Field(None, description="Optional description")
    top_k: float = Field(0.5, ge=0.0, le=1.0, description="Top K similarity threshold (0-1)")

class RagResponse(BaseModel):
    rag_id: str
    name: str
    description: str | None
    top_k: float
    created_at: str
    updated_at: str

class DocumentUploadResponse(BaseModel):
    rag_id: str
    document_id: int
    filename: str
    message: str = "Document uploaded and indexed successfully"
