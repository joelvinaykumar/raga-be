from typing import Optional, List, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class UserRole(str, Enum):
    ASSISTANT = "assistant"
    USER = "user"

class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)
    knowledgebase_id: Optional[str] = Field(default=None)
    top_k: Optional[int] = Field(default=None, ge=1, le=30)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)
    stream: Optional[bool] = Field(default=None)

class Chunk(BaseModel):
    score: Union[float, str]
    content: str
    source: str | None = None
    file_id: Union[int, str, None] = None
    rag_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    filename: str | None = None
    preview: str | None = None
    url: str | None = None


class Citation(BaseModel):
    index: int
    start_char: int
    end_char: int
    display_char: int
    chunk_id: str | None = None
    file_id: Union[int, str, None] = None
    filename: str | None = None
    quote: str | None = None
    score: Union[float, str, None] = None
    url: str | None = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    chunks: List[Chunk] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    id: int
    filename: str
    filesize: int
    upload_timestamp: datetime

class SessionInfo(BaseModel):
    session_id: str
    user_query: str
    knowledgebase_id: str | None = None

class DeleteFileRequest(BaseModel):
    file_id: int

class ModifySessionRequest(BaseModel):
    knowledgebase_id: str | None = None