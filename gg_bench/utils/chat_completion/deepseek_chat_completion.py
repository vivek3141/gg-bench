from typing import List, Optional
import os
import re

# Local Imports
from gg_bench.utils.chat_completion.message import Message
from gg_bench.utils.chat_completion.usage_tracker import UsageTracker
from gg_bench.utils.chat_completion.openai_chat_completion import OpenAIChatCompletion


def remove_think_block(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class DeepseekChatCompletion(OpenAIChatCompletion):
    def __init__(self, model: str):
        super().__init__(
            model,
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )

    def chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs
    ) -> str:
        message = super().chat_completion(messages, usage_tracker, **kwargs)
        return remove_think_block(message)

    async def async_chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs
    ) -> str:
        message = await super().async_chat_completion(messages, usage_tracker, **kwargs)
        return remove_think_block(message)
