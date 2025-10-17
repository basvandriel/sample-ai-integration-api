"""Tests for the AI integration API."""

import pytest
from bootstrap import load_settings
from chat_client_factory import ChatClientFactory


class TestOllamaClient:
    """Test cases for OllamaClient."""

    def test_initialization(self):
        """Test client initialization with default values."""
        settings = load_settings()
        client = ChatClientFactory.create_client(
            provider="ollama", model=settings.CHAT_MODEL, host=settings.OLLAMA_HOST
        )
        assert client.model == settings.CHAT_MODEL
        assert settings.OLLAMA_HOST in client.url

    def test_tinyllama_initialization(self):
        """Test client initialization with tinyllama model."""
        settings = load_settings()
        client = ChatClientFactory.create_client(
            provider="ollama", model="tinyllama", host=settings.OLLAMA_HOST
        )
        assert client.model == "tinyllama"
        assert settings.OLLAMA_HOST in client.url

    @pytest.mark.integration
    def test_streaming_response_structure(self):
        """Test that streaming responses have correct structure."""
        import requests
        import time

        settings = load_settings()

        # Check if Ollama is available
        try:
            response = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.fail("Ollama server not responding")
        except requests.exceptions.RequestException:
            pytest.fail("Ollama server not available")

        client = ChatClientFactory.create_client(
            provider="ollama", model=settings.CHAT_MODEL, host=settings.OLLAMA_HOST
        )
        messages = [{"role": "user", "content": "Say 'test' and nothing else."}]

        # Collect chunks with timeout
        chunks = []
        start_time = time.time()

        try:
            for chunk in client.chat_completion(messages):
                chunks.append(chunk)
                # Limit collection to avoid hanging
                if len(chunks) >= 5 or (time.time() - start_time) > 10:
                    break
        except Exception:
            pytest.fail("Ollama streaming failed")

        if len(chunks) == 0:
            pytest.fail("No chunks received from Ollama")

        # Verify structure
        for chunk in chunks:
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            assert hasattr(chunk.choices[0], "delta")
            assert hasattr(chunk.choices[0].delta, "content")
            assert isinstance(chunk.choices[0].delta.content, str)
