import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

sys.path.append(str(Path(__file__).parents[2] / "api" / "app"))
from preprocessing.make_text import make_chunks
from rag.embedding_handling import VectorIndex
from rag.model import QueryHandler
from schemas.input_schemas import (
    GenerationMetaData,
    OpenAIProvider,
    QueryMetaData,
    RagRequest,
)


@pytest.fixture
def llm_answer_generator():
    """Fixture to initialize the QueryHandler with an API key."""
    # These keys are hardcoded due to a bug with DOTENV load in a async startup context,
    # In a proper service I would get around this by using AWS secrets manager. This isn't common practice!
    OPENAI_TOKEN = "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"
    return QueryHandler(api_key=OPENAI_TOKEN)


@pytest.fixture
def llm_schema():
    """Fixture to initialize the QueryHandler with an API key."""
    config_path = Path(__file__).parent / "test_params.yaml"
    model_args = yaml.safe_load(config_path.read_text())["test_question_answering"]
    return RagRequest(
        Query="placeholder",
        EmbeddingMetaData=QueryMetaData(
            TopKResponses=model_args["TopKResponses"],
            UseReRanking=model_args["UseReRanking"],
        ),
        LLMGenerationMetaData=GenerationMetaData(
            InferenceModel=OpenAIProvider.GPT3_TURBO,
            ReturnStream=model_args["ReturnStream"],
        ),
    )


@pytest.fixture
def retrieval_test_args():
    """Fixture to initialize the QueryHandler with an API key."""
    config_path = Path(__file__).parent / "test_params.yaml"
    model_args = yaml.safe_load(config_path.read_text())["test_retrieval"]
    return model_args


@pytest.fixture
def vector_index():
    with open(str(Path(__file__).parent / "test_cases/book.txt"), "r") as f:
        book_string = f.read()

    chunks: List[Dict[str, Any]] = make_chunks(book_string)
    # These keys are hardcoded due to a bug with DOTENV load in a async startup context,
    # In a proper service I would get around this by using AWS secrets manager. This isn't common practice!
    COHERE_TOKEN = "55wvH2zLoYTZc287sYEnu4MW0GqQLy5dTLUoB8uJ"
    vector_index = VectorIndex(
        cohere_api_key=COHERE_TOKEN,
    )

    vector_index.put_index(requests=chunks)
    return vector_index
