from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

class PDFUploadRequest(BaseModel):
    file_path: str
    collection_name: str | None = None


class MCQOption(BaseModel):
    A: str
    B: str
    C: str
    D: str

class MCQQuestion(BaseModel):
    number: int
    question: str
    options: MCQOption
    answer: str

class MCQResponse(BaseModel):
    total_questions: int
    questions: list[MCQQuestion]

class EssayQuestion(BaseModel):
    number: int
    question: str
    answer: str
    explanation: str = None

class EssayResponse(BaseModel):
    total_questions: int
    questions: list[EssayQuestion]

class JSONResponse(BaseModel):
    answer: str
    confidence: float
    references: list[str] = []
    tags: list[str] = []

