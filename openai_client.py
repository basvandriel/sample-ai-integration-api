from typing import Dict, Generator, List
from openai import AzureOpenAI
from models import ChatCompletionChunk, Choice, Delta
from chat_client import ChatClient


class OpenAIClient(ChatClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1",
        base_url: str = None,
        api_version: str = None,
    ) -> None:
        self.model = model
        self.client = AzureOpenAI(
            azure_endpoint=base_url,
            api_key=api_key,
            api_version=api_version,
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
