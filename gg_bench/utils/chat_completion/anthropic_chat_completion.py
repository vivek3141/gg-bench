import os
from typing import List, Optional, Tuple

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from gg_bench.utils.chat_completion.chat_completion import (
    ChatCompletionProvider,
    Message,
)
from gg_bench.utils.chat_completion.usage_tracker import UsageTracker
from gg_bench.utils.chat_completion.utils import find_config_file
from gg_bench.utils.load_yaml import load_yaml

ANTHROPIC_CONFIG_FILE = "anthropic_config.yaml"


class AnthropicConfigError(Exception):
    """Raised when there's an issue with Anthropic configuration."""


class AnthropicChatCompletion(ChatCompletionProvider):
    model_mapping = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3.7-sonnet-thinking": "claude-3-7-sonnet-20250219",
    }

    def __init__(self, model: str):
        self.thinking = "thinking" in model
        if model in self.model_mapping:
            model = self.model_mapping[model]
        self.model = model
        self._initialize_anthropic()

    def _initialize_anthropic(self) -> None:
        anthropic_config_path = find_config_file(
            config_file=ANTHROPIC_CONFIG_FILE,
            exception_to_raise=AnthropicConfigError,
            exception_message="Anthropic config not found",
        )
        config = load_yaml(anthropic_config_path)
        api_key = config["api_key"]
        anthropic.api_key = api_key
        os.environ["ANTHROPIC_API_KEY"] = api_key

    def _split_system_and_user_messages(
        self, messages: List[Message]
    ) -> Tuple[str, List[Message]]:
        system_prompt = "".join(
            [msg["content"] for msg in messages if msg["role"] == "system"]
        )
        messages = [msg for msg in messages if msg["role"] != "system"]
        return system_prompt, messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(anthropic.RateLimitError),
    )
    def chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs
    ) -> str:
        client = anthropic.Anthropic()
        system_prompt, messages = self._split_system_and_user_messages(messages)

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096

        if self.thinking and "thinking" not in kwargs:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}

        response = client.messages.create(
            model=self.model, messages=messages, system=system_prompt, **kwargs  # type: ignore
        )
        if usage_tracker:
            usage_tracker.update(
                usage=response.usage,
                model=self.model,
            )

        return response.content[-1].text

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(anthropic.RateLimitError),
    )
    async def async_chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs
    ) -> str:
        client = anthropic.AsyncAnthropic()
        system_prompt, messages = self._split_system_and_user_messages(messages)

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096

        if "thinking" in self.model and "thinking" not in kwargs:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}

        response = await client.messages.create(
            model=self.model, messages=messages, system=system_prompt, **kwargs  # type: ignore
        )
        if usage_tracker:
            usage_tracker.update(
                usage=response.usage,
                model=self.model,
            )

        return response.content[-1].text

    def get_model_name(self) -> str:
        return self.model
