from openai import OpenAI

from app import config


class LLMService:
    """OpenAI-compatible client. Pointing LLM_BASE_URL at vLLM or Ollama runs the same code fully on-prem."""

    def __init__(self):
        self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self.model = config.LLM_MODEL

    def chat(self, messages, temperature=0.1):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content
