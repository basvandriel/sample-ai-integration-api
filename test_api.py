"""Tests for the AI integration API."""

import pytest
from ollama_client import OllamaClient


class TestOllamaClient:
    """Test cases for OllamaClient."""

    def test_initialization(self):
        """Test client initialization with default values."""
        client = OllamaClient()
        assert client.model == "mistral"  # Default model
        assert "localhost:11434" in client.url

    def test_tinyllama_initialization(self):
        """Test client initialization with tinyllama model."""
        client = OllamaClient(model="tinyllama")
        assert client.model == "tinyllama"
        assert "localhost:11434" in client.url

    @pytest.mark.integration
    def test_streaming_response_structure(self):
        """Test that streaming responses have correct structure."""
        client = OllamaClient()
        messages = [{"role": "user", "content": "Say 'test' and nothing else."}]

        chunks = list(client.chat_completion(messages))
        assert len(chunks) > 0

        # Verify structure
        for chunk in chunks:
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            assert hasattr(chunk.choices[0], "delta")
            assert hasattr(chunk.choices[0].delta, "content")
            assert isinstance(chunk.choices[0].delta.content, str)
