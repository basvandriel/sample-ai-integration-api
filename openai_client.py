from typing import Dict, Generator, List
from openai import OpenAI
from models import ChatCompletionChunk, Choice, Delta
from chat_client import ChatClient


class OpenAIClient(ChatClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: str = None,
        api_version: str = None,
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"api-key": api_key} if api_version else None,
        )

    def chat_completion(
        self, messages: List[Dict[str, str]]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Generate streaming chat completion using OpenAI API."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield ChatCompletionChunk(
                    choices=[
                        Choice(delta=Delta(content=chunk.choices[0].delta.content))
                    ]
                )
