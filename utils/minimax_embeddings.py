"""
MiniMax Embeddings wrapper for LangChain.

MiniMax's ``embo-01`` uses a proprietary API (``texts`` + ``type`` fields)
rather than the OpenAI-compatible ``/v1/embeddings`` format.  This class
implements the LangChain ``Embeddings`` interface so it can be used as a
drop-in replacement for ``OpenAIEmbeddings`` in any LangChain pipeline.

Usage:
    from utils.minimax_embeddings import MiniMaxEmbeddings

    embeddings = MiniMaxEmbeddings()           # reads MINIMAX_API_KEY
    vecs = embeddings.embed_documents(texts)   # list[list[float]]
    vec  = embeddings.embed_query("hello")     # list[float]
"""

import os
from typing import List

import requests
from langchain_core.embeddings import Embeddings


class MiniMaxEmbeddings(Embeddings):
    """LangChain-compatible embeddings using MiniMax embo-01."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embo-01",
        base_url: str = "https://api.minimax.io/v1",
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")

        if not self.api_key:
            raise EnvironmentError(
                "Missing API key: please set the MINIMAX_API_KEY environment variable."
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, texts: List[str], embed_type: str) -> List[List[float]]:
        """Call the MiniMax embeddings endpoint.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        embed_type : ``"db"`` for storage, ``"query"`` for search queries.
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "texts": texts,
            "type": embed_type,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if data.get("base_resp", {}).get("status_code", 0) != 0:
            msg = data.get("base_resp", {}).get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax embeddings error: {msg}")

        return data["vectors"]

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (uses ``type="db"``)."""
        if not texts:
            return []
        return self._call_api(texts, embed_type="db")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (uses ``type="query"``)."""
        vectors = self._call_api([text], embed_type="query")
        return vectors[0]
