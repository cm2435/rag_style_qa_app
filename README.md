# Basic RAG Style QA Bot to answer questions on 'Romeo And Juliet'
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Here is a brief implimentation of using
1. A best in class embedding model, 
2. A closed source OpenAI model as a generator given context.
3. A cohere reranker for optional ranking relevence improvement 

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Permissions](#permissions)
  - [Installation](#installation)
- [Running the Tests](#running-the-tests)
- [Project writeup](#project-writeup)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python (3.10+)
- poetry

### Permissions. 
The local development of the API requires no special CSP provider permissions. 
The 'deployment' scripts require write acess to AWS ecr and a working aws-cli toolchain to verify.
The 'tests' and running the API require two keys
  1. A openai API key to inference the generative portion of the app
  2. A cohere API key to rerank text chunks if you want best in class retrieval.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/cm2435/rag_style_qa_app
```

2. Create a new poetry environment. Navigate to the parent directory of the project and run. 
```bash
poetry install 
poetry shell
```

3. To run the API for local development, navigate to 
```bash
./src/api/app/api.py
```
And set the global variables on startup for the COHERE and OPENAI api keys, on lines 37 and 40. 
After this simply navigate to the API's parent directory 
```bash
cd src/api/app 
```
and run 
```bash
python api.py
```

### Running tests
Tests are stored under ./src/tests. There are unit tests to ensure correct working of the general API, and then 'performance' tests to asert the quality of the machine learning system.


To run the quality tests to test the information retrieval pipelime, run 
deepeval test run tests/performance/test_qa.py
```python
 pytest src/tests/performance/test_retrieval.py
```
Quality tests for the generative pipeline are done with the deepeval framework and can be run by doing the following. 
1. Navigate to 
```bash
cd src/tests/performance 
```
and set the key variables for the openai and cohere API tokens in conftest.py on lines 20, 53.
2. Run
```python
 export OPENAI_API_KEY="<KEY_HERE">

 deepeval test run src/tests/performance/test_qa.py    
```

### Project writeup
Below are some details around the design and consideration choices for the application

## 1. Book choice
I chose romeo and Juliet as it was a piece of content I was reasonably familar with. This would make it easier to sanity check LLM outputs when developing the API workflow, and also make more well grounded decisions in my preprocessing as I explored the text in the experiments folder. 

## 2. API architecture.
The API is comprised of three main machine learning components.
1. An embeddings model that takes in a listed of chunked data strings and their associated metadata, and then uses an embeddings model to create indexable representations. 
2. A reranker model that is used conditionally based on the user input schema to increase the relevence of the retrieved content chunks
3. A closed source LLM via an API to generate answers given a system prompt, the most relevent chunks of text, and the user query. 

### Machine learning rationalle: 
1. For the embeddings model I used was the Snowflake/snowflake-arctic-embed-m-long model from huggingface served via sentence_transformers. This was chosen for several reasons 
  a. At the time of writing it was the highest performing on model on the MTEB benchmark for the semantic textual similarity task https://huggingface.co/blog/mteb that was still under 1Bn parameters and could be easily run on CPU. 
  b. It has a 8192 max token context length, making it suitable for my longer chunks of text in my preprocessing (the longest of which is approximately 2500 tokens).

2. For the reranker, I chose the cohere closed source solution. This also was for afew reasons: 
  1. For some production use cases I have hand-rolled a neural reranker (in recsys or in some code applications). But papers like S-GPT require an on device LLM, and due to hardware constraints this was not feasible.
  2. For reranking the cohere model scores very highly on common benchmarks like MS-MARCO and has impressive ratelimits making it suitable for production usecases. 

3. For the closed source LLM, I allowed for the use of either the most recent version of chatgpt-3.5-turbo-16k or GPT-4 depending on the user input. This could easily be extended to use other vendors, but to reduce the complexity of the application, I went with the option I think currently provides the best in class balance between ratelimits, performance and pricing. 

Evaluation rationalle: 
To test the application, I segemented the core logic of the API into its two logical problems, information retrieval and context based generation. These are tested using pytest seperately; this allows the developer to make changes to the pipeline and mock out the specific impacts and causes of quality improvement or degridation.

### **Information retrieval** 

This involved testing for a given query, how well the embedding and reranking system work together to pull out the relevent chunks of text that contain the answer. 

I evaluated this using the industry standard Mean Reciprical Rank metric on a gold standard answer for a set of given queries. For a proper testing setup, I would write a diverse set of 20 questions and manually label the index UUID that corresponds to the 'correct answer pair', and then measure the reciprical rank of the retrieved content for each test case. 

As I was short for time however, I reframed the problem by asking the API questions about a specific act and scene of the play, and then using the metadata stored in the API to mark if it had managed to correctly identify a chunk from the right part of the play.

To make this more robust, I would also measure the precision and recall @ k for the pairwise gold data retrieval, and the NDCG@K for the content to measure diversity as well as accuracy of content retrieval (as duplicate concept discussion can hurt LLM rag generation performances)

### **LLM generation** 
To measure how well our closed source LLM could answer a question given a specific frozen piece of context, I used a style of 'model-as-a-judge' evaluation, where a closed source Judge model like GPT-4 is asked to score the quality of a generation from our pipeline out of a max_score with respect to metrics like faithfullness, relevence or hallucination. 

This is implimented using the deepeval framework with a single indicitive example, testing for hallcuination and relevence as measures of answer quality. Given more time, many more test cases covering a diverse set of user behaviour would be added. 

### **API architecture rationalle**
The full RAG pipeline is served using fastapi workers and async endpoints, allowing for async responses which is helpful since the RAG responses can be relatively long (~seconds).

The models and indexxing is all done on application startup to reduce inference latency. The models are not compiled which would improve throughput, but the book is small and thus we are not bottlenecked by embedding speed for this corpus. 

The API is served in production using docker as a containeriser running a set of uvicorn workers on a gunicorn webserver load balanced behind an nginx websocket. Since the API is set to run on CPU, this allows us to fully utilise our device resources by setting num_workers>1 if need be, allowing our model to serve greater concurrency load on a single VM. 

The deployment is done via a bash script to build the image, and then tag and push it to AWS ECR to then be deployed to EC2, fargate or sagemaker.

The CI/CD is done using github actions, for which there is a basic workflow script to lint and sort code that is checked into the dev branch. 