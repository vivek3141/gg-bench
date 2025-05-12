from typing import List, Optional

import openai
import os
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Local Imports
from gg_bench.utils.chat_completion.chat_completion import ChatCompletionProvider
from gg_bench.utils.chat_completion.message import Message
from gg_bench.utils.chat_completion.usage_tracker import UsageTracker


class RestAPIChatCompletion(ChatCompletionProvider):
    def __init__(self, model: str, port: int):
        self.model = model
        # self.client = openai.OpenAI(base_url=f"http://localhost:{port}/v1")
        # self.async_client = openai.AsyncOpenAI(base_url=f"http://localhost:{port}/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        assert api_key, "OPENROUTER_API_KEY environment variable must be set"

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    def chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs  # type: ignore
        )
        if usage_tracker:
            usage_tracker.update(
                usage=response.usage,
                model=self.model,
            )

        return response.choices[0].message.content

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    async def async_chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs,
    ) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, **kwargs  # type: ignore
        )
        if usage_tracker:
            usage_tracker.update(
                usage=response.usage,
                model=self.model,
            )

        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model
