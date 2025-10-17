import json
from dataclasses import dataclass

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from bootstrap import load_settings
from chat_client_factory import ChatClientFactory
from chat_client import ChatClient
from settings import Settings

app = FastAPI(title="AI Integration Server", version="1.0.0")
settings = load_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins including file://
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@dataclass
class ChatRequest:
    message: str


def get_chat_client(settings: Settings = Depends(load_settings)) -> ChatClient:
    """Dependency injection for chat client based on settings."""
    return ChatClientFactory.create_client(
        provider=settings.CHAT_PROVIDER,
        api_key=settings.OPENAI_API_KEY if settings.CHAT_PROVIDER == "openai" else None,
        model=settings.CHAT_MODEL,
        base_url=settings.OPENAI_API_ENDPOINT,
        api_version=settings.OPENAI_API_VERSION,
        host=settings.OLLAMA_HOST,
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, client: ChatClient = Depends(get_chat_client)
):
    """Streaming chat endpoint using Server-Sent Events (SSE) - sends individual tokens"""

    async def generate_stream():
        try:
            # Create streaming chat completion
            messages = [
                {"role": "system", "content": "Talk like a pirate."},
                {"role": "user", "content": request.message},
            ]

            # Stream tokens as they arrive from the chat client
            for chunk in client.chat_completion(messages):
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Ensure immediate delivery with explicit flush

                    sse_chunk = f"data: {json.dumps({'content': content})}\n\n"
                    yield sse_chunk

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            error = str(e)
            print(f"Error occurred: {error}")
            yield f"data: {json.dumps({'error': error})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )
