# Current web context for a RAG prompt

The notebooks retrieve context from an indexed corpus. This standalone example
uses [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to retrieve current web excerpts instead. It converts each result into a LangChain
`Document` with `source` and `title` metadata, then prints a prompt containing the
question, numbered sources and excerpts. It stops before model inference.

Search uses Fast mode and is free for exploration and light use, with no Parallel
account or API key required. This example sends no authorization header, reads no
`.env` file and needs no OpenAI, Pinecone or Cohere credentials. Questions and search
queries are sent to Parallel. The anonymous service has usage limits; a service
error is raised rather than presented as an empty search.

## Run

Use a separate Python 3.11 environment for this example. Its small dependency set
does not require installing the root `requirements.txt` or the notebook stack.
From the repository root:

```bash
cd examples/parallel_web_retrieval
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python retrieve.py "How does LangChain support retrieval-augmented generation?" \
  --query "LangChain retrieval augmented generation documentation"
```

On Windows, create the environment with `py -3.11 -m venv venv` and activate it
with `venv\Scripts\activate` in Command Prompt, then run the same `python` commands.
Repeat `--query` to include related keyword searches for the same question.
The request and connection cleanup are bounded by a 60-second overall timeout;
network errors, tool errors and timeouts propagate to the caller. No automatic
retry is performed.

Output has this shape, with actual titles, URLs and excerpts supplied by search:

```text
System: Answer only from the supplied context and cite source numbers like [1]. ...
Human: Context:
[1] <page title>
Source: <page URL>
<relevant excerpts>

Question: How does LangChain support retrieval-augmented generation?
```

## Connect the context to generation

Run this Python snippet from the example directory in the same environment:

```python
import asyncio
from retrieve import build_prompt, retrieve

question = "How does LangChain support retrieval-augmented generation?"
documents = asyncio.run(retrieve(
    question, ["LangChain retrieval augmented generation documentation"]
))
if documents:
    prompt_value = build_prompt(question, documents)
    print(prompt_value.to_string())
    # With your separately configured LangChain chat model:
    # answer = llm.invoke(prompt_value)
```

This is the context-to-prompt stage used in the RAG notebooks. Supply your own
configured model to generate an answer. Review the returned sources before relying
on them: excerpts may be incomplete, and retrieved web text is untrusted input.
The example does not create embeddings, persist an index or run an agent loop.

## Offline checks

From this directory, with the environment activated:

```bash
python -m unittest discover -s tests -v
```

The tests exercise the MCP client over a local HTTP transport, including request
headers, document conversion, prompt context, errors, timeout and cancellation.
They make no live search requests.

This example is derived from **[bRAG AI](https://github.com/bragai/brag-langchain)**
and is distributed under the repository's [license](../../LICENSE).
