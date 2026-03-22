"""Unit tests for utils.minimax_embeddings module."""

import os
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.minimax_embeddings import MiniMaxEmbeddings


class TestMiniMaxEmbeddingsInit:
    """Tests for MiniMaxEmbeddings initialization."""

    def test_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}):
            emb = MiniMaxEmbeddings()
            assert emb.api_key == "mm-test"

    def test_explicit_api_key(self):
        emb = MiniMaxEmbeddings(api_key="explicit-key")
        assert emb.api_key == "explicit-key"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="Missing API key"):
                MiniMaxEmbeddings()

    def test_default_model(self):
        emb = MiniMaxEmbeddings(api_key="test")
        assert emb.model == "embo-01"

    def test_default_base_url(self):
        emb = MiniMaxEmbeddings(api_key="test")
        assert emb.base_url == "https://api.minimax.io/v1"

    def test_custom_base_url_strips_trailing_slash(self):
        emb = MiniMaxEmbeddings(api_key="test", base_url="https://example.com/")
        assert emb.base_url == "https://example.com"


class TestMiniMaxEmbeddingsCallAPI:
    """Tests for the _call_api method."""

    @patch("utils.minimax_embeddings.requests.post")
    def test_embed_documents_calls_api(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "total_tokens": 10,
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = MiniMaxEmbeddings(api_key="test-key")
        result = emb.embed_documents(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

        # Verify API call
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "embo-01"
        assert payload["texts"] == ["hello", "world"]
        assert payload["type"] == "db"

    @patch("utils.minimax_embeddings.requests.post")
    def test_embed_query_calls_api_with_query_type(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vectors": [[0.1, 0.2, 0.3]],
            "total_tokens": 5,
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = MiniMaxEmbeddings(api_key="test-key")
        result = emb.embed_query("hello")

        assert result == [0.1, 0.2, 0.3]

        payload = mock_post.call_args[1]["json"]
        assert payload["type"] == "query"

    @patch("utils.minimax_embeddings.requests.post")
    def test_api_error_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vectors": [],
            "base_resp": {"status_code": 1001, "status_msg": "Invalid API key"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = MiniMaxEmbeddings(api_key="bad-key")
        with pytest.raises(RuntimeError, match="Invalid API key"):
            emb.embed_documents(["hello"])

    def test_embed_documents_empty_list(self):
        emb = MiniMaxEmbeddings(api_key="test-key")
        result = emb.embed_documents([])
        assert result == []

    @patch("utils.minimax_embeddings.requests.post")
    def test_authorization_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vectors": [[0.1]],
            "total_tokens": 1,
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = MiniMaxEmbeddings(api_key="my-secret-key")
        emb.embed_query("test")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    @patch("utils.minimax_embeddings.requests.post")
    def test_correct_endpoint_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vectors": [[0.1]],
            "total_tokens": 1,
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        emb = MiniMaxEmbeddings(api_key="test")
        emb.embed_query("test")

        url = mock_post.call_args[0][0]
        assert url == "https://api.minimax.io/v1/embeddings"
