from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.llm_client import AnthropicClient


def test_anthropic_client_passes_timeout_to_sdk():
    with patch("anthropic.AsyncAnthropic") as mock_async_anthropic:
        AnthropicClient(api_key="sk-test", model="claude-test", timeout=12.5)

    mock_async_anthropic.assert_called_once_with(
        api_key="sk-test", timeout=12.5
    )


def _build_client_with_stub_response(response) -> AnthropicClient:
    with patch("anthropic.AsyncAnthropic"):
        client = AnthropicClient(api_key="sk-test", model="m", timeout=30.0)

    async def fake_create(**kwargs):
        return response

    client._client = MagicMock()
    client._client.messages.create = fake_create
    return client


async def test_generate_raises_value_error_on_empty_content():
    response = SimpleNamespace(content=[])
    client = _build_client_with_stub_response(response)

    with pytest.raises(ValueError, match="no text content"):
        await client.generate(
            system="sys", user="u", max_tokens=10, temperature=0.0
        )


async def test_generate_raises_value_error_on_non_text_block():
    response = SimpleNamespace(content=[SimpleNamespace(type="tool_use")])
    client = _build_client_with_stub_response(response)

    with pytest.raises(ValueError, match="no text content"):
        await client.generate(
            system="sys", user="u", max_tokens=10, temperature=0.0
        )


async def test_generate_returns_text_when_first_block_is_text():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")]
    )
    client = _build_client_with_stub_response(response)

    result = await client.generate(
        system="sys", user="u", max_tokens=10, temperature=0.0
    )
    assert result == "hello"
