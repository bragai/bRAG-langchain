# Retrieval-Augmented Generation (RAG) Project

#### 🔜 Check out [bragai.dev](https://bragai.dev) (launching soon)

---------------------

This repository contains a comprehensive exploration of Retrieval-Augmented Generation (RAG) for various applications.
Each notebook provides a detailed, hands-on guide to setting up and experimenting with RAG from an introductory level to advanced implementations, including multi-querying and custom RAG builds.

![rag_detail_v2](assets/img/rag-architecture.png)

## Project Structure

If you want to jump straight into it, check out the file `full_basic_rag.ipynb` -> this file will give you a boilerplate starter code of a fully customizable RAG chatbot.

Make sure to run your files in a virtual environment (checkout section `Get Started`)

The following notebooks can be found under the directory `notebooks/`.

### [1]\_rag_setup_overview.ipynb

This introductory notebook provides an overview of RAG architecture and its foundational setup.
The notebook walks through: 
- **Environment Setup**: Configuring the environment, installing necessary libraries, and API setups.
- **Initial Data Loading**: Basic document loaders and data preprocessing methods.
- **Embedding Generation**: Generating embeddings using various models, including OpenAI's embeddings.
- **Vector Store**: Setting up a vector store (ChromaDB/Pinecone) for efficient similarity search.
- **Basic RAG Pipeline**: Creating a simple retrieval and generation pipeline to serve as a baseline.

### [2]\_rag_with_multi_query.ipynb

Building on the basics, this notebook introduces multi-querying techniques in the RAG pipeline, exploring: 
- **Multi-Query Setup**: Configuring multiple queries to diversify retrieval.
- **Advanced Embedding Techniques**: Utilizing multiple embedding models to refine retrieval.
- **Pipeline with Multi-Querying**: Implementing multi-query handling to improve relevance in response generation.
- **Comparison & Analysis**: Comparing results with single-query pipelines and analyzing performance improvements.

### [3]_rag_routing_and_query_construction.ipynb

This notebook delves deeper into customizing a RAG pipeline.
It covers: 
- **Logical Routing:** Implements function-based routing for classifying user queries to appropriate data sources based on programming languages.
- **Semantic Routing:** Uses embeddings and cosine similarity to direct questions to either a math or physics prompt, optimizing response accuracy.
- **Query Structuring for Metadata Filters:** Defines structured search schema for YouTube tutorial metadata, enabling advanced filtering (e.g., by view count, publication date).
- **Structured Search Prompting:** Leverages LLM prompts to generate database queries for retrieving relevant content based on user input.
- **Integration with Vector Stores:** Links structured queries to vector stores for efficient data retrieval.


### [4]_rag_indexing_and_advanced_retrieval.ipynb

Continuing from the previous customization, this notebook explores:
- **Preface on Document Chunking:** Points to external resources for document chunking techniques.
- **Multi-representation Indexing:** Sets up a multi-vector indexing structure for handling documents with different embeddings and representations.
- **In-Memory Storage for Summaries:** Uses InMemoryByteStore for storing document summaries alongside parent documents, enabling efficient retrieval.
- **MultiVectorRetriever Setup:** Integrates multiple vector representations to retrieve relevant documents based on user queries.
- **RAPTOR Implementation:** Explores RAPTOR, an advanced indexing and retrieval model, linking to in-depth resources.
- **ColBERT Integration:** Demonstrates ColBERT-based token-level vector indexing and retrieval, which captures contextual meaning at a fine-grained level.
- **Wikipedia Example with ColBERT:** Retrieves information about Hayao Miyazaki using the ColBERT retrieval model for demonstration.

### [5]_rag_retrieval_and_reranking.ipynb

This final notebook brings together the RAG system components, with a focus on scalability and optimization: 
- **Document Loading and Splitting:** Loads and chunks documents for indexing, preparing them for vector storage.
- **Multi-query Generation with RAG-Fusion:** Uses a prompt-based approach to generate multiple search queries from a single input question.
- **Reciprocal Rank Fusion (RRF):** Implements RRF for re-ranking multiple retrieval lists, merging results for improved relevance.
- **Retriever and RAG Chain Setup:** Constructs a retrieval chain for answering queries, using fused rankings and RAG chains to pull contextually relevant information.
- **Cohere Re-Ranking:** Demonstrates re-ranking with Cohere’s model for additional contextual compression and refinement.
- **CRAG and Self-RAG Retrieval:** Explores advanced retrieval approaches like CRAG and Self-RAG, with links to examples.
- **Exploration of Long-Context Impact:** Links to resources explaining the impact of long-context retrieval on RAG models.

## Getting Started

### Pre-requisites

Ensure **Python 3.11.11** (preferred) is installed on your system. Follow the platform-specific instructions below to install it if not already installed.

#### macOS
1. Install [Homebrew](https://brew.sh/) if not already installed:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python 3.11.11:
   ```bash
   brew install python@3.11
   ```
3. Verify installation:
   ```bash
   python3.11 --version
   ```

#### Linux
1. Update your package manager:
   ```bash
   sudo apt update
   ```
2. Install Python 3.11.11:
   ```bash
   sudo apt install python3.11 python3.11-venv
   ```
3. Verify installation:
   ```bash
   python3.11 --version
   ```

#### Windows
1. Download the Python 3.11.11 installer from [Python.org](https://www.python.org/downloads/).
2. Run the installer and ensure you check the box **"Add Python to PATH"**.
3. Verify installation:
   ```cmd
   python --version
   ```
---

### Installation Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/bRAGAI/bRAG-langchain.git
cd bRAG-langchain
```

#### 2. Create a Virtual Environment
Use Python 3.11.11 to create a virtual environment:
```bash
python3.11 -m venv venv
```

Activate the virtual environment:
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```
- **Windows**:
  ```cmd
  venv\Scripts\activate
  ```

#### 3. Verify and Fix Python Version
If the virtual environment defaults to a different Python version (e.g., Python 3.13):
1. Verify the current Python version inside the virtual environment:
   ```bash
   python --version
   ```
2. Use Python 3.11 explicitly within the virtual environment:
   ```bash
   python3.11
   ```
3. Ensure the `python` command uses Python 3.11 by creating a symbolic link:
   ```bash
   ln -sf $(which python3.11) $(dirname $(which python))/python
   ```
4. Verify the fix:
   ```bash
   python --version
   ```

#### 4. Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

---

### Additional Steps

#### 5. Run the Notebooks
Begin with `[1]_rag_setup_overview.ipynb` to get familiar with the setup process. Proceed sequentially through the other notebooks:

- `[1]_rag_setup_overview.ipynb`
- `[2]_rag_with_multi_query.ipynb`
- `[3]_rag_routing_and_query_construction.ipynb`
- `[4]_rag_indexing_and_advanced_retrieval.ipynb`
- `[5]_rag_retrieval_and_reranking.ipynb`

#### 6. Set Up Environment Variables
1. Duplicate the `.env.example` file in the root directory and rename it to `.env`.
2. Add the following keys (replace with your actual values):

   ```env
   # LLM Model - Get key at https://platform.openai.com/api-keys
   OPENAI_API_KEY="your-api-key"

   # LangSmith - Get key at https://smith.langchain.com
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="your-api-key"
   LANGCHAIN_PROJECT="your-project-name"

   # Pinecone Vector Database - Get key at https://app.pinecone.io
   PINECONE_INDEX_NAME="your-project-index"
   PINECONE_API_HOST="your-host-url"
   PINECONE_API_KEY="your-api-key"

   # Astraflow (UCloud) - OpenAI-compatible, 200+ models
   # Global endpoint - Get key at https://astraflow.ucloud-global.com
   ASTRAFLOW_API_KEY="your-api-key"

   # Astraflow (UCloud) - China endpoint - Get key at https://astraflow.ucloud.cn
   ASTRAFLOW_CN_API_KEY="your-api-key"

      # Cohere - Get key at https://dashboard.cohere.com/api-keys
   COHERE_API_KEY=your-api-key
   ```

---

You're now ready to use the project!

## Usage

After setting up the environment and running the notebooks in sequence, you can:

1.  **Experiment with Retrieval-Augmented Generation**:
    Use the foundational setup in `[1]_rag_setup_overview.ipynb` to understand the basics of RAG.

2.  **Implement Multi-Querying**:
    Learn how to improve response relevance by introducing multi-querying techniques in `[2]_rag_with_multi_query.ipynb`.

## Star History

<a href="https://www.star-history.com/?type=date&repos=bragai%2Fbrag-langchain">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=bragai/brag-langchain&type=date&theme=dark&legend=top-left&sealed_token=Qe6d8SOeE0RWvnru_-Xma2aRcyLreotC7ALvS2gxbzJcaNVR7NCl-vBARdgSOCBxJSHocgIPhQuBdazJ_SsHN3NaRP6DS4sQSu3CT5aK1qcRI5gxXl49E52DW3t6OJWd7IOdl8Ks4pdzS4lvLM56AdwHypQeKGZlDDVlnFFfcWFZQUYIVjS7mLoYXZ9N" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=bragai/brag-langchain&type=date&legend=top-left&sealed_token=Qe6d8SOeE0RWvnru_-Xma2aRcyLreotC7ALvS2gxbzJcaNVR7NCl-vBARdgSOCBxJSHocgIPhQuBdazJ_SsHN3NaRP6DS4sQSu3CT5aK1qcRI5gxXl49E52DW3t6OJWd7IOdl8Ks4pdzS4lvLM56AdwHypQeKGZlDDVlnFFfcWFZQUYIVjS7mLoYXZ9N" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=bragai/brag-langchain&type=date&legend=top-left&sealed_token=Qe6d8SOeE0RWvnru_-Xma2aRcyLreotC7ALvS2gxbzJcaNVR7NCl-vBARdgSOCBxJSHocgIPhQuBdazJ_SsHN3NaRP6DS4sQSu3CT5aK1qcRI5gxXl49E52DW3t6OJWd7IOdl8Ks4pdzS4lvLM56AdwHypQeKGZlDDVlnFFfcWFZQUYIVjS7mLoYXZ9N" />
 </picture>
</a>

## Contact
Do you have questions or want to collaborate? Please open an issue or email Taha Ababou at taha@bragai.dev

`If this project helps you, consider buying me a coffee ☕. Your support helps me keep contributing to the open-source community!`
<p>
    <a href="https://buymeacoffee.com/bragai" target="_blank" rel="noopener noreferrer">
        <img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#white" />
    </a>
</p>

<br>

    The notebooks and visual diagrams were inspired by Lance Martin's LangChain Tutorial.

    
