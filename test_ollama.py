from ollama_client import OllamaClient


def test_ollama_client_initialization():
    """Test that OllamaClient can be initialized."""
    client = OllamaClient()
    assert client.model == "mistral"
    assert client.url == "http://localhost:11434/api/chat"


def test_ollama_streaming_response():
    """Test that OllamaClient returns streaming responses."""
    client = OllamaClient()
    messages = [
        {
            "role": "system",
            "content": "Je bent een behulpzame assistent.",
        },
        {
            "role": "user",
            "content": "Zeg hallo in het Nederlands.",
        },
    ]

    # Collect response chunks
    chunks = list(client.chat_completion(messages))
    assert len(chunks) > 0

    # Check structure of first chunk
    first_chunk = chunks[0]
    assert hasattr(first_chunk, "choices")
    assert len(first_chunk.choices) > 0
    assert hasattr(first_chunk.choices[0], "delta")
    assert hasattr(first_chunk.choices[0].delta, "content")

    # Check that we got some actual content
    full_response = "".join(chunk.choices[0].delta.content for chunk in chunks)
    assert len(full_response.strip()) > 0
    print(f"Response: {full_response}")


if __name__ == "__main__":
    # For manual testing
    test_ollama_client_initialization()
    test_ollama_streaming_response()
    print("All tests passed!")
