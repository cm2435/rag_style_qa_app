import json
import sys
from pathlib import Path

import numpy as np
import pytest
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).parents[2] / "api" / "app")
)

from rag.embedding_handling import VectorIndex
from schemas.input_schemas import RagRequest, GenerationMetaData, OpenAIProvider, QueryMetaData

def test_api_responses(vector_index: VectorIndex,retrieval_test_args)->None:
    response_ranks = []
    for act_number in retrieval_test_args['acts_to_search']:
        for scene_number in retrieval_test_args['scenes_to_search']:
            input_request = RagRequest(
                Query=f"What character has the most dialogue in Act {act_number} Scene {scene_number}",
                EmbeddingMetaData=QueryMetaData(
                    TopKResponses=10,
                    UseReRanking=True,
                    FilteringActNumber=None,
                    FilteringSceneNumber=None
                ),
                LLMGenerationMetaData=GenerationMetaData(
                    InferenceModel=OpenAIProvider.GPT3_TURBO,
                    ReturnStream=True 
                )
            )
            search_results = vector_index.query_index(
                input_request,
                top_k=input_request.EmbeddingMetaData.TopKResponses,
                use_reranking=input_request.EmbeddingMetaData.UseReRanking,
            )
            gold_standard_data_not_found = True
            for result_idx, result in enumerate(search_results):
                if result['metadata']['act'] == act_number and result['metadata']['scene'] == scene_number:
                    gold_standard_data_not_found = False
                    response_ranks.append(result_idx+1)
                    continue
                    
            if gold_standard_data_not_found:
                 response_ranks.append(np.inf)
    
    final_mrr = np.average([1/rank for rank in response_ranks])
    print(response_ranks)
    print("Final MMR:", final_mrr)
    assert final_mrr > retrieval_test_args['minimum_mrr']
