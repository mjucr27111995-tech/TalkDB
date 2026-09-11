import google.generativeai as genai

from talkdb.llms.base import BaseLLM
from talkdb.utils.errors import TalkDBError


class GeminiLLM(BaseLLM):
    """Google Gemini adapter."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash") -> None:
        if not api_key:
            raise TalkDBError("TALKDB-VAL-001", "Gemini API key is required")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        if not response.text:
            raise TalkDBError("TALKDB-EXT-002", "Gemini returned an empty response")
        return response.text

    def generate_with_history(self, messages: list[dict[str, str]]) -> str:
        chat = self._model.start_chat(history=[])
        last_response = None
        for msg in messages:
            if msg["role"] in ("user", "system"):
                last_response = chat.send_message(msg["content"])
        if last_response is None or not last_response.text:
            raise TalkDBError("TALKDB-EXT-002", "Gemini returned an empty response")
        return last_response.text

    def get_provider_name(self) -> str:
        return f"Google Gemini ({self._model_name})"
