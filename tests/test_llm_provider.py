"""Unit tests for utils.llm_provider module."""

import os
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_provider import (
    _resolve_provider,
    _clamp_temperature,
    get_chat_model,
    get_embeddings,
    PROVIDER_DEFAULTS,
)


# ---------------------------------------------------------------------------
# _resolve_provider
# ---------------------------------------------------------------------------


class TestResolveProvider:
    """Tests for _resolve_provider()."""

    def test_default_is_openai(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_provider("LLM_PROVIDER") == "openai"

    def test_reads_env_var(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}):
            assert _resolve_provider("LLM_PROVIDER") == "minimax"

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "MiniMax"}):
            assert _resolve_provider("LLM_PROVIDER") == "minimax"

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "  minimax  "}):
            assert _resolve_provider("LLM_PROVIDER") == "minimax"

    def test_unknown_provider_raises(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}):
            with pytest.raises(ValueError, match="Unknown provider"):
                _resolve_provider("LLM_PROVIDER")

    def test_custom_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_provider("LLM_PROVIDER", default="minimax") == "minimax"


# ---------------------------------------------------------------------------
# _clamp_temperature
# ---------------------------------------------------------------------------


class TestClampTemperature:
    """Tests for _clamp_temperature()."""

    def test_normal_value(self):
        assert _clamp_temperature(0.5) == 0.5

    def test_clamp_above_one(self):
        assert _clamp_temperature(1.5) == 1.0

    def test_clamp_below_zero(self):
        assert _clamp_temperature(-0.1) == 0.0

    def test_boundary_zero(self):
        assert _clamp_temperature(0.0) == 0.0

    def test_boundary_one(self):
        assert _clamp_temperature(1.0) == 1.0


# ---------------------------------------------------------------------------
# PROVIDER_DEFAULTS
# ---------------------------------------------------------------------------


class TestProviderDefaults:
    """Tests for provider preset configuration."""

    def test_openai_preset_exists(self):
        assert "openai" in PROVIDER_DEFAULTS

    def test_minimax_preset_exists(self):
        assert "minimax" in PROVIDER_DEFAULTS

    def test_minimax_base_url(self):
        assert PROVIDER_DEFAULTS["minimax"]["base_url"] == "https://api.minimax.io/v1"

    def test_minimax_default_model(self):
        assert PROVIDER_DEFAULTS["minimax"]["chat_model"] == "MiniMax-M2.7"

    def test_minimax_api_key_env(self):
        assert PROVIDER_DEFAULTS["minimax"]["api_key_env"] == "MINIMAX_API_KEY"

    def test_openai_base_url_is_none(self):
        assert PROVIDER_DEFAULTS["openai"]["base_url"] is None

    def test_openai_default_model(self):
        assert PROVIDER_DEFAULTS["openai"]["chat_model"] == "gpt-3.5-turbo"

    def test_minimax_embedding_model(self):
        assert PROVIDER_DEFAULTS["minimax"]["embedding_model"] == "embo-01"

    def test_openai_embedding_model(self):
        assert PROVIDER_DEFAULTS["openai"]["embedding_model"] == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# get_chat_model
# ---------------------------------------------------------------------------


class TestGetChatModel:
    """Tests for get_chat_model()."""

    @patch("utils.llm_provider.ChatOpenAI")
    def test_openai_default(self, mock_chat):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            get_chat_model()
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "gpt-3.5-turbo"
            assert "base_url" not in call_kwargs

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_provider(self, mock_chat):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=True):
            get_chat_model(provider="minimax")
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "MiniMax-M2.7"
            assert call_kwargs["base_url"] == "https://api.minimax.io/v1"

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_clamps_temperature(self, mock_chat):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=True):
            get_chat_model(provider="minimax", temperature=1.5)
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["temperature"] == 1.0

    @patch("utils.llm_provider.ChatOpenAI")
    def test_custom_model_override(self, mock_chat):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=True):
            get_chat_model(provider="minimax", model="MiniMax-M2.7-highspeed")
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "MiniMax-M2.7-highspeed"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="Missing API key"):
                get_chat_model(provider="openai")

    @patch("utils.llm_provider.ChatOpenAI")
    def test_env_var_provider_selection(self, mock_chat):
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "mm-test"}):
            get_chat_model()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "MiniMax-M2.7"

    @patch("utils.llm_provider.ChatOpenAI")
    def test_extra_kwargs_forwarded(self, mock_chat):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            get_chat_model(provider="openai", max_tokens=500)
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["max_tokens"] == 500


# ---------------------------------------------------------------------------
# get_embeddings
# ---------------------------------------------------------------------------


class TestGetEmbeddings:
    """Tests for get_embeddings()."""

    @patch("utils.llm_provider.OpenAIEmbeddings")
    def test_openai_default(self, mock_emb):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            get_embeddings()
            mock_emb.assert_called_once()
            call_kwargs = mock_emb.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-large"

    @patch("utils.llm_provider.OpenAIEmbeddings")
    def test_minimax_provider(self, mock_emb):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=True):
            get_embeddings(provider="minimax")
            mock_emb.assert_called_once()
            call_kwargs = mock_emb.call_args[1]
            assert call_kwargs["model"] == "embo-01"
            assert call_kwargs["base_url"] == "https://api.minimax.io/v1"

    @patch("utils.llm_provider.OpenAIEmbeddings")
    def test_falls_back_to_llm_provider(self, mock_emb):
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "mm-test"}):
            get_embeddings()
            call_kwargs = mock_emb.call_args[1]
            assert call_kwargs["model"] == "embo-01"

    @patch("utils.llm_provider.OpenAIEmbeddings")
    def test_embedding_provider_overrides_llm_provider(self, mock_emb):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "minimax",
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
        }):
            get_embeddings()
            call_kwargs = mock_emb.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-large"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="Missing API key"):
                get_embeddings(provider="minimax")
