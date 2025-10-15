from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import  AzureOpenAI

from bootstrap import load_settings

app = FastAPI()

settings = load_settings()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@dataclass
class ChatRequest:
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    endpoint = settings.OPENAI_API_ENDPOINT
    api_key = settings.OPENAI_API_KEY
    api_version = settings.OPENAI_API_VERSION

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )

    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "low"},
        instructions="Talk like a pirate.",
        input=request.message,
    )

    return {"message": response.output_text}