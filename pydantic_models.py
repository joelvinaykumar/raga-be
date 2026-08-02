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
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class Chunk(BaseModel):
    score: Union[float, str]
    content: str
    source: str
    file_id: Union[int, str]

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    chunks: List[Chunk]


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