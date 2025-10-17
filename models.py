from typing import List
from dataclasses import dataclass


@dataclass
class Delta:
    content: str


@dataclass
class Choice:
    delta: Delta


@dataclass
class ChatCompletionChunk:
    choices: List[Choice]
