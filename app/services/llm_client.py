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
    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)

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
        if not response.content or not hasattr(response.content[0], "text"):
            block_type = (
                getattr(response.content[0], "type", "unknown")
                if response.content
                else "empty"
            )
            raise ValueError(
                f"Anthropic response had no text content (first block: {block_type})"
            )
        return response.content[0].text
