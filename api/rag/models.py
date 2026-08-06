from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class RagModel(BaseModel):
    id: int
    rag_id: str
    name: str
    description: str
    top_k: float
    created_at: datetime
    updated_at: datetime

class RagCreateModel(BaseModel):
    name: str
    description: str
    top_k: int = Field(..., gt=0, description="Number of top results to retrieve")
    chunk_size: int = Field(500, gt=0, description="Chunk size in characters for document splitting")
    embedding_model: str = Field("text-embedding-ada-002", description="Embedding model identifier")

class RagPatchModel(BaseModel):
    name: str | None = Field(None, description="Name of the RAG collection")
    description: str | None = Field(None, description="Optional description")

class RagDetailModel(BaseModel):
    rag_id: str
    name: str
    description: str
    top_k: int
    chunk_size: int
    embedding_model: str

class RagResponseModel(BaseModel):
    rag_id: str
    message: str


class RagPromptSuggestionsResponse(BaseModel):
    rag_id: str
    prompts: list[str]
    source: str