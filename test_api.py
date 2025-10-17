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
        import requests
        import time

        # Check if Ollama is available
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.fail("Ollama server not responding")
        except requests.exceptions.RequestException:
            pytest.fail("Ollama server not available")

        client = OllamaClient()
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
