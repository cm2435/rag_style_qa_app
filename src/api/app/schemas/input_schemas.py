from typing import Dict, List, Optional
from pydantic import BaseModel
from enum import StrEnum

class OpenAIProvider(StrEnum):
    GPT4 = 'gpt-4'
    GPT3_TURBO = 'gpt-3.5-turbo-16k'

class QueryMetaData(BaseModel):
    TopKResponses: int = 5
    UseReRanking: bool = False
    FilteringActNumber: Optional[int] = None
    FilteringSceneNumber: Optional[int] = None
    
class GenerationMetaData(BaseModel):
    InferenceModel : OpenAIProvider = OpenAIProvider.GPT3_TURBO 
    ReturnStream: bool = True 
    
class RagRequest(BaseModel):
    Query: str
    EmbeddingMetaData: QueryMetaData = None
    LLMGenerationMetaData: GenerationMetaData = None