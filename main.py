import json
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AzureOpenAI

from bootstrap import load_settings

app = FastAPI(title="AI Integration Server", version="1.0.0")
settings = load_settings()

# CORS configuration - Allow all origins for testing
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

@app.get("/test")
async def test_endpoint():
    return {"message": "Server is working", "timestamp": "2025-10-16"}


@dataclass
class ChatRequest:
    message: str


def get_azure_client() -> AzureOpenAI:
    """Create and return Azure OpenAI client"""
    return AzureOpenAI(
        api_key=settings.OPENAI_API_KEY,
        azure_endpoint=settings.OPENAI_API_ENDPOINT,
        api_version=settings.OPENAI_API_VERSION
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    client = get_azure_client()
    
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "low"},
        instructions="Talk like a pirate.",
        input=request.message,
    )
    
    return {"message": response.output_text}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events (SSE) - sends individual tokens"""
    client = get_azure_client()

    async def generate_stream():
        try:
            # Create streaming chat completion
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Talk like a pirate."},
                    {"role": "user", "content": request.message}
                ],
                stream=True,
                temperature=0.7
            )
            
            # Stream tokens as they arrive from OpenAI
            for chunk in stream:
                if (chunk.choices and 
                    chunk.choices[0].delta and 
                    chunk.choices[0].delta.content):
                    
                    content = chunk.choices[0].delta.content
                    # Ensure immediate delivery with explicit flush
                    sse_chunk = f"data: {json.dumps({'content': content})}\n\n"
                    yield sse_chunk

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
        }
    )


@app.post("/chat/stream/accumulated")
async def chat_stream_accumulated(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events (SSE) - sends accumulated text"""
    client = get_azure_client()

    async def generate_stream():
        try:
            # Create streaming chat completion
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Talk like a pirate."},
                    {"role": "user", "content": request.message}
                ],
                stream=True,
                temperature=0.7
            )
            
            accumulated_content = ""
            
            # Stream accumulated content as tokens arrive from OpenAI
            for chunk in stream:
                if (chunk.choices and 
                    chunk.choices[0].delta and 
                    chunk.choices[0].delta.content):
                    
                    accumulated_content += chunk.choices[0].delta.content
                    # Send the complete content so far
                    sse_chunk = f"data: {json.dumps({'content': accumulated_content})}\n\n"
                    yield sse_chunk
            
            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
        }
    )