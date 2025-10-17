import requests
import json
from typing import Dict, Generator, List
from models import ChatCompletionChunk, Choice, Delta
from chat_client import ChatClient


class OllamaClient(ChatClient):
    def __init__(
        self, model: str = "mistral", host: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self.url = f"{host}/api/chat"

    def chat_completion(
        self, messages: List[Dict[str, str]]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        Mimics openai.ChatCompletion.create(messages=..., stream=...)
        """
        data = {"model": self.model, "messages": messages}
        response = requests.post(self.url, json=data, stream=True)

        # Generator yielding chunks
        for line in response.iter_lines():
            if line:
                js = json.loads(line)
                if "message" in js:
                    yield ChatCompletionChunk(
                        choices=[Choice(delta=Delta(content=js["message"]["content"]))]
                    )
