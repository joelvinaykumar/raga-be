from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class UserRole(str, Enum):
    ASSISTANT = "asssistant"
    USER = "user"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    filesize: int
    upload_timestamp: datetime

class SessionInfo(BaseModel):
    session_id: str
    user_query: str

class DeleteFileRequest(BaseModel):
    file_id: int