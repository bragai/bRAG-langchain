"""Integration tests for MiniMax provider.

These tests make real API calls and require MINIMAX_API_KEY to be set.
Run with: pytest tests/test_integration.py -v -m integration
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip all tests if no API key
pytestmark = pytest.mark.integration
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set",
)


@skip_no_key
class TestMiniMaxChatIntegration:
    """Integration tests for MiniMax chat model."""

    def test_basic_chat_completion(self):
        from utils.llm_provider import get_chat_model

        llm = get_chat_model(provider="minimax", temperature=0.1)
        response = llm.invoke("Say 'hello' and nothing else.")
        assert response.content
        assert len(response.content) > 0

    def test_chat_with_system_message(self):
        from langchain_core.messages import SystemMessage, HumanMessage
        from utils.llm_provider import get_chat_model

        llm = get_chat_model(provider="minimax", temperature=0.1)
        messages = [
            SystemMessage(content="You are a helpful assistant. Reply in one sentence."),
            HumanMessage(content="What is RAG?"),
        ]
        response = llm.invoke(messages)
        assert response.content
        assert len(response.content) > 10

    def test_temperature_zero_accepted(self):
        from utils.llm_provider import get_chat_model

        llm = get_chat_model(provider="minimax", temperature=0.0)
        response = llm.invoke("Say 'test'.")
        assert response.content


@skip_no_key
class TestMiniMaxEmbeddingsIntegration:
    """Integration tests for MiniMax embeddings."""

    def test_embed_single_query(self):
        from utils.minimax_embeddings import MiniMaxEmbeddings

        emb = MiniMaxEmbeddings()
        vec = emb.embed_query("Hello world")
        assert len(vec) == 1536
        assert all(isinstance(v, float) for v in vec)

    def test_embed_multiple_documents(self):
        from utils.minimax_embeddings import MiniMaxEmbeddings

        emb = MiniMaxEmbeddings()
        vecs = emb.embed_documents(["Hello", "World"])
        assert len(vecs) == 2
        assert all(len(v) == 1536 for v in vecs)

    def test_cosine_similarity_related_texts(self):
        import numpy as np
        from utils.minimax_embeddings import MiniMaxEmbeddings

        emb = MiniMaxEmbeddings()
        v1 = emb.embed_query("machine learning algorithms")
        v2 = emb.embed_query("deep learning neural networks")
        v3 = emb.embed_query("chocolate cake recipe")

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_related = cosine_sim(v1, v2)
        sim_unrelated = cosine_sim(v1, v3)

        # Related texts should have higher similarity
        assert sim_related > sim_unrelated
