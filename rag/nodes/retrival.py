from .llm import base_llm
from .retriever import retriever
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """relevant check on retrieved documents"""

    binary_score: str = Field(
        description = "binary score of the relevance of the retrieved documents to the query"
    )
    
retrival_llm = base_llm.with_structed_output(GradeDocuments)