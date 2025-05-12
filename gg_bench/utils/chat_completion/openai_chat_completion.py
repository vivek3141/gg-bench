import os
from typing import List, Optional

import openai
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
from gg_bench.utils.chat_completion.utils import find_config_file
from gg_bench.utils.load_yaml import load_yaml

OPENAI_CONFIG_FILE = "openai_config.yaml"


class OpenAIConfigError(Exception):
    """Raised when there's an issue with OpenAI configuration."""


class OpenAIChatCompletion(ChatCompletionProvider):
    def __init__(
        self, model: str, base_url: str | None = None, api_key: str | None = None
    ):
        self.model = model

        if api_key is None:
            openai_config_path = find_config_file(
                config_file=OPENAI_CONFIG_FILE,
                exception_to_raise=OpenAIConfigError,
                exception_message="OpenAI config not found",
            )
            config = load_yaml(openai_config_path)
            api_key = config["api_key"]
            openai.organization = config.get("organization", None)
            os.environ["OPENAI_ORG_ID"] = config.get("organization", None) or ""

        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        self.client = openai.OpenAI(base_url=base_url)
        self.async_client = openai.AsyncOpenAI(base_url=base_url)

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
        if self.model.startswith("o") and "max_tokens" in kwargs:
            del kwargs["max_tokens"]

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
        if self.model.startswith("o1") and "max_tokens" in kwargs:
            del kwargs["max_tokens"]

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
