"""
Multi-provider LLM and Embedding factory for bRAG-langchain.

Supports OpenAI (default) and MiniMax via environment variables.
Both providers are accessed through LangChain's ChatOpenAI / OpenAIEmbeddings
wrappers since MiniMax exposes an OpenAI-compatible API.

Usage:
    from utils.llm_provider import get_chat_model, get_embeddings

    llm = get_chat_model()          # uses LLM_PROVIDER env var
    embeddings = get_embeddings()   # uses EMBEDDING_PROVIDER env var
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": None,  # default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY",
        "chat_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-3-large",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "chat_model": "MiniMax-M3",
        "embedding_model": "embo-01",
    },
}


def _resolve_provider(env_var: str, default: str = "openai") -> str:
    """Return the lowercased provider name from *env_var*, falling back to *default*."""
    provider = os.getenv(env_var, default).strip().lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(PROVIDER_DEFAULTS)}"
        )
    return provider


def _clamp_temperature(temperature: float) -> float:
    """Clamp temperature to the MiniMax-accepted range [0, 1]."""
    return max(0.0, min(temperature, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance for the chosen provider.

    Parameters
    ----------
    provider : str, optional
        ``"openai"`` or ``"minimax"``. Defaults to the ``LLM_PROVIDER`` env
        var, then ``"openai"``.
    model : str, optional
        Override the default model for the provider.
    temperature : float
        Sampling temperature.  Clamped to [0, 1] for MiniMax.
    **kwargs
        Extra keyword arguments forwarded to ``ChatOpenAI``.
    """
    if provider is None:
        provider = _resolve_provider("LLM_PROVIDER")

    preset = PROVIDER_DEFAULTS[provider]
    api_key = os.getenv(preset["api_key_env"], "")

    if not api_key:
        raise EnvironmentError(
            f"Missing API key: please set the {preset['api_key_env']} environment variable."
        )

    model = model or preset["chat_model"]

    if provider == "minimax":
        temperature = _clamp_temperature(temperature)

    chat_kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        **kwargs,
    }

    if preset["base_url"]:
        chat_kwargs["base_url"] = preset["base_url"]

    return ChatOpenAI(**chat_kwargs)


def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> OpenAIEmbeddings:
    """Create a LangChain OpenAIEmbeddings instance for the chosen provider.

    Parameters
    ----------
    provider : str, optional
        ``"openai"`` or ``"minimax"``. Defaults to the ``EMBEDDING_PROVIDER``
        env var, then the ``LLM_PROVIDER`` env var, then ``"openai"``.
    model : str, optional
        Override the default embedding model for the provider.
    **kwargs
        Extra keyword arguments forwarded to ``OpenAIEmbeddings``.

    Notes
    -----
    MiniMax's ``embo-01`` embedding endpoint is **not** OpenAI-compatible (it
    uses ``texts`` / ``type`` fields instead of ``input``).  This helper still
    returns an ``OpenAIEmbeddings`` object that points at MiniMax so that it
    can be used as a drop-in replacement *if* MiniMax adds OpenAI-compatible
    embedding support in the future.  For production use with MiniMax
    embeddings today, consider using the ``MiniMaxEmbeddings`` class from
    ``utils.minimax_embeddings`` instead.
    """
    if provider is None:
        provider = _resolve_provider(
            "EMBEDDING_PROVIDER",
            default=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        )

    preset = PROVIDER_DEFAULTS[provider]
    api_key = os.getenv(preset["api_key_env"], "")

    if not api_key:
        raise EnvironmentError(
            f"Missing API key: please set the {preset['api_key_env']} environment variable."
        )

    model = model or preset["embedding_model"]

    emb_kwargs = {
        "model": model,
        "api_key": api_key,
        **kwargs,
    }

    if preset["base_url"]:
        emb_kwargs["base_url"] = preset["base_url"]

    return OpenAIEmbeddings(**emb_kwargs)
