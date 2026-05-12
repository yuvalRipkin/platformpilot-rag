from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        ...


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
