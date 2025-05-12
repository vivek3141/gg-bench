import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from gg_bench.utils.chat_completion.message import Message
from gg_bench.utils.chat_completion.usage_tracker import UsageTracker


class ChatCompletionProvider(ABC):
    """Abstract base class for chat completion providers."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs,
    ) -> str:
        """Synchronous chat completion method."""
        pass

    @abstractmethod
    async def async_chat_completion(
        self,
        messages: List[Message],
        usage_tracker: Optional[UsageTracker] = None,
        **kwargs,
    ) -> str:
        """Asynchronous chat completion method."""
        pass


class ModelFactory:
    """Factory class for creating chat completion providers."""

    @staticmethod
    def get_provider(model_name: str) -> ChatCompletionProvider:
        """
        Get the appropriate chat completion provider for the given model.

        Args:
            model_name (str): The name of the model.

        Returns:
            ChatCompletionProvider: An instance of the appropriate provider.

        Raises:
            ValueError: If the model is not supported.
        """
        if (
            model_name.startswith("gpt")
            or model_name.startswith("o1")
            or model_name.startswith("o3")
            or model_name.startswith("o4")
        ):
            from gg_bench.utils.chat_completion.openai_chat_completion import (
                OpenAIChatCompletion,
            )

            return OpenAIChatCompletion(model_name)
        elif model_name.startswith("claude"):
            from gg_bench.utils.chat_completion.anthropic_chat_completion import (
                AnthropicChatCompletion,
            )

            return AnthropicChatCompletion(model_name)
        elif model_name.startswith("meta-llama") or model_name.startswith("llama"):
            model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            from gg_bench.utils.chat_completion.openai_chat_completion import (
                OpenAIChatCompletion,
            )

            return OpenAIChatCompletion(
                model_name,
                base_url="https://api.together.xyz/v1",
                api_key=os.environ.get("TOGETHER_API_KEY"),
            )
        elif model_name.startswith("deepseek"):
            from gg_bench.utils.chat_completion.deepseek_chat_completion import (
                DeepseekChatCompletion,
            )

            model_name = "deepseek-ai/DeepSeek-R1"
            return DeepseekChatCompletion(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")


def chat_completion(
    model: str,
    messages: List[Message],
    usage_tracker: Optional[UsageTracker] = None,
    **kwargs: Any,
) -> str:
    """
    Perform a synchronous chat completion.

    Args:
        model (str): The name of the model to use.
        messages (List[Message]): The list of messages for the conversation.
        usage_tracker (Optional[UsageTracker]): A tracker for token usage and costs.
        **kwargs: Additional keyword arguments to pass to the chat completion method.

    Returns:
        str: The generated response.

    Raises:
        ValueError: If an unsupported model is specified.
        RuntimeError: If there's an error during chat completion.
    """
    provider = ModelFactory.get_provider(model)
    output = provider.chat_completion(messages, usage_tracker=usage_tracker, **kwargs)
    return output


async def chat_completion_async(
    model: str,
    messages: List[Message],
    usage_tracker: Optional[UsageTracker] = None,
    **kwargs: Any,
) -> str:
    """
    Perform an asynchronous chat completion.

    Args:
        model (str): The name of the model to use.
        messages (List[Message]): The list of messages for the conversation.
        usage_tracker (Optional[UsageTracker]): A tracker for token usage and costs.
        **kwargs: Additional keyword arguments to pass to the chat completion method.

    Returns:
        str: The generated response.

    Raises:
        ValueError: If an unsupported model is specified.
        RuntimeError: If there's an error during chat completion.
    """
    provider = ModelFactory.get_provider(model)
    output = await provider.async_chat_completion(
        messages, usage_tracker=usage_tracker, **kwargs
    )
    return output


if __name__ == "__main__":
    # Usage example
    messages: List[Message] = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Tell me a joke."),
    ]
    try:
        usage_tracker = UsageTracker()
        model = "o1"
        response_llama = chat_completion(
            model,
            messages,
            usage_tracker=usage_tracker,
        )
        print(f"Response from {model}: {response_llama}")
        print(f"Usage tracker: {usage_tracker}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
