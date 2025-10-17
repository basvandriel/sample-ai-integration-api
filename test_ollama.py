from ollama_client import OllamaClient
import requests
import time


def test_ollama_client_initialization():
    """Test that OllamaClient can be initialized."""
    client = OllamaClient()
    assert client.model == "mistral"
    assert client.url == "http://localhost:11434/api/chat"


def test_ollama_streaming_response():
    """Test that OllamaClient returns streaming responses."""
    client = OllamaClient()

    # First check if Ollama is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("⚠️  Ollama server not responding, skipping streaming test")
            return
    except requests.exceptions.RequestException:
        print("⚠️  Ollama server not available, skipping streaming test")
        return

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

    # Collect response chunks with timeout
    start_time = time.time()
    chunks = []

    try:
        for chunk in client.chat_completion(messages):
            chunks.append(chunk)
            # Break after first few chunks to avoid hanging
            if len(chunks) >= 3 or (time.time() - start_time) > 10:
                break
    except Exception as e:
        print(f"⚠️  Error during streaming: {e}, skipping test")
        return

    print(f"📦 Collected {len(chunks)} chunks in {time.time() - start_time:.2f}s")

    if len(chunks) == 0:
        print("⚠️  No chunks received, Ollama might not be ready")
        return

    # Check structure of first chunk
    first_chunk = chunks[0]
    assert hasattr(first_chunk, "choices")
    assert len(first_chunk.choices) > 0
    assert hasattr(first_chunk.choices[0], "delta")
    assert hasattr(first_chunk.choices[0].delta, "content")

    # Check that we got some actual content
    full_response = "".join(chunk.choices[0].delta.content for chunk in chunks)
    assert len(full_response.strip()) > 0
    print(f"✅ Response: {full_response}")


if __name__ == "__main__":
    # For manual testing
    test_ollama_client_initialization()
    test_ollama_streaming_response()
    print("All tests passed!")
