from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
import os 

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from preprocessing.api_logging import logger
from preprocessing.make_text import make_chunks
from rag.embedding_handling import VectorIndex
from rag.model import QueryHandler
from schemas.input_schemas import RagRequest

app = FastAPI()
controller = APIRouter()


@app.on_event("startup")
def startup_event():
    """Method to be called when the app starts up.

    - Downloads the wrapped class for fine-tuned Tapas models.
    """
    env_path = Path(__file__).parents[1] / ".env"
    load_dotenv(dotenv_path=str(env_path))
    with open(str(Path(__file__).parent / "data" / "corpus.txt"), "r") as f:
        book_string = f.read()

    logger.info("downloading wrapped class for models")
    global vector_index
    global llm_handler
    chunks: List[Dict[str, Any]] = make_chunks(book_string)

    #These keys are hardcoded due to a bug with DOTENV load in a async startup context,
    #In a proper service I would get around this by using AWS secrets manager. This isn't common practice!
    vector_index = VectorIndex(
        cohere_api_key="55wvH2zLoYTZc287sYEnu4MW0GqQLy5dTLUoB8uJ"
    )
    llm_handler = QueryHandler(
        api_key="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"
    )

    vector_index.put_index(requests=chunks)


@controller.post("/chat", status_code=200)
async def transformation(schema: RagRequest):
    """Endpoint for performing the transformation.

    Args:
        schema: The request schema containing the questions and table.

    Returns:
        A StreamingResponse containing the predicted answer coordinates and aggregation indices.
    """
    try:
        logger.info(f"Inference on question: {schema}")
        if schema.LLMGenerationMetaData.InferenceModel: 
            if schema.LLMGenerationMetaData.InferenceModel not in llm_handler._MODEL_MAX_LENGTH.keys():
                raise ValueError("Unsupported model used as generator.")
    
        index_results = vector_index.query_index(
            schema,
            top_k=schema.EmbeddingMetaData.TopKResponses,
            use_reranking=schema.EmbeddingMetaData.UseReRanking,
        )
        return StreamingResponse(
            llm_handler.invoke_llm(
                query=schema.Query,
                retrieved_results=index_results,
                generation_model=schema.LLMGenerationMetaData.InferenceModel,
            ),
            media_type="text/plain",
        )

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(controller)

# Here for debugging only
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
