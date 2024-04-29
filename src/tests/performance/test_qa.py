import pytest
import sys
import asyncio
from pathlib import Path
from typing import List
import yaml
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from .test_cases.question_answering import sample_context

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", sample_context)
async def test_customer_chatbot(test_case: dict, llm_answer_generator, llm_schema):
    llm_schema.Query = test_case['question']
    async_gen = llm_answer_generator.invoke_llm(
        query=llm_schema.Query,
        retrieved_results=test_case['vector_input'],
        generation_model=llm_schema.LLMGenerationMetaData.InferenceModel,
    )
    # Collecting output from async generator
    output = []
    async for chunk in async_gen:
        output.append(str(chunk))
    full_output = ''.join(output)  # Assuming each chunk is a string
    
    print("generated answer:", full_output)
    test_example = LLMTestCase(
        input=test_case['question'],
        actual_output=full_output,
        context=[response['vector_input']['metadata']['chunk_text'] for response in sample_context]
    )
    
    hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_example, [hallucination_metric, answer_relevancy_metric], llm_answer_generator)
