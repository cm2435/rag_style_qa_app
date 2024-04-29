import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

import openai
import pydantic
import tiktoken
from openai import AsyncOpenAI
from schemas.input_schemas import GenerationMetaData, RagRequest
from preprocessing.api_logging import logger

class QueryHandler:
    def __init__(self, api_key: str, system_prompt: str = None,):
        self._base_inference_model = "gpt-4"
        self._client = openai.AsyncClient(
            api_key=api_key
        )
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self._MODEL_MAX_LENGTH = {"gpt-3.5-turbo-16k": 8192, "gpt-4": 4192}
        if system_prompt:
            self._system_prompt = system_prompt
        else:
            # Load default system prompt from file if not provided
            prompt_path = Path(__file__).parent / "prompts" / "question_answer.txt"
            if prompt_path.exists():
                self._system_prompt = prompt_path.read_text()
            else:
                raise FileNotFoundError("Default system prompt file not found.")

    async def invoke_llm(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        generation_model: str = None,
        max_attempts: int = 5
        ):
        # Determine the model to use
        generation_model = generation_model if generation_model else self._base_inference_model

        # Prepare the input prompt
        system_prompt = self._system_prompt.format(
            corpus=self._format_retrieved_text_chunks(retrieved_results)
        )
        attempt = 0
        while attempt < max_attempts:
            try:
                # Invoke the model with a stream to handle continuous output
                generation = await self._client.chat.completions.create(
                    model=generation_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    stream=True,
                )

                async for chunk in self._handle_returning_stream(generation):
                    yield chunk
                break  # Exit loop if successful

            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.critical(f"Failed after {max_attempts} attempts. Last error: {str(e)}")
                    break  # Exit loop after max attempts

                # Calculate exponential backoff
                sleep_time = min(30, (2 ** attempt))
                logger.warning(f"Attempt {attempt}: An error occurred. Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)

    def _format_retrieved_text_chunks(
        self, retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        current_token_offset = 0
        abridged_chunks_list = []
        if isinstance(retrieved_chunks, dict):
            retrieved_chunks = [retrieved_chunks]
            
        for chunk in retrieved_chunks:
            #Remove irrelevent sections
            chunk['metadata'].pop('text', None)
            chunk['metadata'].pop('stringified_input', None)
            if (
                len(self.encoder.encode(str(chunk))) + current_token_offset
                < self._MODEL_MAX_LENGTH[self._base_inference_model]
            ):
                abridged_chunks_list.append(chunk)

        return f"""{abridged_chunks_list}"""

    @staticmethod
    async def _handle_returning_stream(output):
        async for chunk in output:
            current_chunk = chunk.choices[0].delta.content or ""
            yield current_chunk.encode("utf-8")


# Example usage
async def main():
    handler = QueryHandler()
    example_request = RagRequest(
        Query="What is the capital of France?",
        LLMGenerationMetaData=GenerationMetaData(ReturnStream=False),
    )

    response = await handler.invoke_llm(example_request)
    print(openai.ChatCompletion)

    async for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")


if __name__ == "__main__":
    asyncio.run(main())
