import cohere
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tqdm 

from schemas.input_schemas import RagRequest
from preprocessing.api_logging import logger 

class VectorIndex:
    def __init__(
        self, 
        model_name: str = "Snowflake/snowflake-arctic-embed-m-long", 
        cohere_api_key: Optional[str] = None
        ):
        self.index = None
        self.metadata = []
        self._model_name = model_name
        self._embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
        if cohere_api_key:
            self.cohere_client = cohere.Client(cohere_api_key)
            
    def query_index(self, request: RagRequest, top_k: int = 10, use_reranking: bool = False):
        assert self.index is not None, "Index has not been built, you need to put content in to search!"
        # Embed the query
        query_embedding = self._embedding_model.encode(request.Query, prompt_name="query", convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().detach().numpy()
        if len(query_embedding_np.shape) == 1:
            query_embedding_np = query_embedding_np.reshape(1, -1)

        faiss.normalize_L2(query_embedding_np)  # Normalize for cosine similarity
        distances, indices = self.index.search(query_embedding_np, top_k)
        results = []
        logger.debug(f"Vector database queried, {len(indices[0])} results returned")

        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.metadata[idx]
            if request.EmbeddingMetaData.FilteringActNumber is not None and metadata["act"] != request.EmbeddingMetaData.FilteringActNumber:
                continue
            if request.EmbeddingMetaData.FilteringSceneNumber is not None and metadata["scene"] != request.EmbeddingMetaData.FilteringSceneNumber:
                continue
            results.append({"metadata": metadata, "distance": distance})

        if use_reranking:
            results = self.rerank_results(request.Query, results, top_k)

        logger.info(f"Content retrieval completed, {len(results)} results returned. Reranking: {use_reranking}")
        return results

    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int):
        documents = [result["metadata"]["stringified_input"] for result in results]
        assert self.cohere_client, "To use reranking you must set a key in the client!"
        reranked_results = self.cohere_client.rerank(
            query=query, 
            documents=documents, 
            top_n=top_k, 
            model='rerank-english-v3.0'
        )
        for result in reranked_results.results:
            results[result.index]['relevance_score'] = result.relevance_score
        
        return sorted(results, key = lambda result : result['relevance_score'], reverse=True)

    def put_index(self, requests: List[Dict[str, Any]], verbose: bool = True):
        logger.info(f"Embedding {len(requests)} documents and putting them into the vector database.")

        # Prepare all inputs for batch processing
        inputs = [self._stringify_chunk(req) for req in requests]

        for req, inp in zip(requests, inputs):
            req['stringified_input'] = inp

        # Process all inputs in a single batch
        embeddings = self._embedding_model.encode(inputs, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().detach().numpy()

        # Ensure embeddings are in the shape (n_samples, embedding_dim)
        if len(embeddings_np.shape) == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        faiss.normalize_L2(embeddings_np)

        # Initialize the FAISS index if it's not already initialized
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings_np.shape[1])

        self.index.add(embeddings_np)
        self.metadata.extend(requests)

        if verbose:
            print("All documents have been indexed.")

    def has_documents(self) -> bool:
        """Check if any content has already been imported into the vectorstore."""
        return True if self.metadata and self.index else False
    
    @staticmethod
    def _stringify_chunk(chunk: Dict[str, Any]) -> str:
        text = chunk.pop("text", None)
        stringified_chunk_parts = [f"{k}: {v}" for k, v in chunk.items() if v is not None]
        stringified_chunk = ", ".join(stringified_chunk_parts)

        if text is not None:
            stringified_chunk += f", text: {text}"

        chunk["text"] = text
        return stringified_chunk

if __name__ == "__main__":
    # Example of model initialization
    vector_index = VectorIndex(api_key="55wvH2zLoYTZc287sYEnu4MW0GqQLy5dTLUoB8uJ")
    # Example usage of put_index
    request_data = {"text": "Example text", "act": 1, "scene": 3}
    put_data = [
        {"text": "Example text", "act": 1, "scene": 3},
        {"text": "Example text", "act": 3, "scene": 4},
        {"text": "Example text", "act": 50, "scene": 30},
        {"text": "Example text", "act": 100, "scene": 1},
        {"text": "Example text", "act": 50, "scene": 100},
    ]
    print("Putting data")
    vector_index.put_index(put_data)

    # Example usage of query_index
    print("Querying data")
    index_request = RagRequest(
        Query="Find information about Act 1 Scene 3", MetaData=QueryMetaData()
    )
    results = vector_index.query_index(index_request, top_k=5, use_reranking=True)
    print(results)



