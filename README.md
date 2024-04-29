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

- Python (3.7+)
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
```python
 deepeval test run src/tests/performance/test_qa.py    
```

### Project writeup