from openai import OpenAI

from talkdb.llms.base import BaseLLM
from talkdb.utils.errors import TalkDBError


class OpenAILLM(BaseLLM):
    """OpenAI GPT adapter."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise TalkDBError("TALKDB-VAL-001", "OpenAI API key is required")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise TalkDBError("TALKDB-EXT-001", "OpenAI returned an empty response")
        return content

    def generate_with_history(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise TalkDBError("TALKDB-EXT-001", "OpenAI returned an empty response")
        return content

    def get_provider_name(self) -> str:
        return f"OpenAI ({self._model})"
