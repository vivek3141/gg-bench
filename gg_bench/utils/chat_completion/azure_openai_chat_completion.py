
import openai
import os
from openai import AzureOpenAI

# Local Imports
from gg_bench.utils.chat_completion.openai_chat_completion import OpenAIChatCompletion, OpenAIConfigError
from gg_bench.utils.chat_completion.utils import find_config_file
from gg_bench.utils.load_yaml import load_yaml

OPENAI_CONFIG_FILE = "openai_config.yaml"

class AzureOpenAIChatCompletion(OpenAIChatCompletion):
    def __init__(
        self, model: str, base_url: str | None = None, api_key: str | None = None
    ):
        self.model = model

        openai_config_path = find_config_file(
            config_file=OPENAI_CONFIG_FILE,
            exception_to_raise=OpenAIConfigError,
            exception_message="OpenAI config not found",
        )
        config = load_yaml(openai_config_path)
        config = config.get("azure_openai", config)

        api_key = config.get("api_key", api_key)
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        api_version = config.get("api_version", None)

        azure_endpoint = config.get("endpoint", base_url)

        self.client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
