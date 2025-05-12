import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Union

import tiktoken
import yaml
from anthropic.types import Usage
from openai.types import CompletionUsage

# Internal imports
from gg_bench.utils.chat_completion.message import Message

# Model costs per 1K tokens
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-4o": {
        "in_cost": 0.0025,
        "out_cost": 0.01,
    },
    "gpt-4o-mini": {
        "in_cost": 0.00015,
        "out_cost": 0.00045,
    },
    "claude-3.5-sonnet": {
        "in_cost": 0.003,
        "out_cost": 0.015,
    },
    "claude-3-5-sonnet-20240620": {
        "in_cost": 0.003,
        "out_cost": 0.015,
    },
    "claude-3.7-sonnet": {
        "in_cost": 0.003,
        "out_cost": 0.015,
    },
    "claude-3-7-sonnet-20250219": {
        "in_cost": 0.003,
        "out_cost": 0.015,
    },
    "o1": {
        "in_cost": 0.015,
        "out_cost": 0.06,
    },
    "o3-mini": {
        "in_cost": 0.0011,
        "out_cost": 0.0044,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "in_cost": 0,
        "out_cost": 0,
    },
}
tokenizer = tiktoken.encoding_for_model("gpt-4o")


@dataclass
class UsageTracker:
    """Tracks token usage and associated costs."""

    in_tokens: int = 0
    out_tokens: int = 0
    cost: float = 0.0

    def update(self, usage: Union[CompletionUsage, Usage], model: str) -> None:
        """
        Update the usage tracker with the cost of the input and output tokens.

        Args:
            usage_tracker (UsageTracker): The current usage tracker.
            usage (CompletionUsage): The completion usage.
            model (str): The model used for the completion.

        Raises:
            ValueError: If the model is not found in MODEL_COSTS.
        """
        if model not in MODEL_COSTS:
            return

        if isinstance(usage, CompletionUsage):
            in_tokens, out_tokens = usage.prompt_tokens, usage.completion_tokens
        elif isinstance(usage, Usage):
            in_tokens, out_tokens = usage.input_tokens, usage.output_tokens
        else:
            raise ValueError(f"Unsupported usage type: {type(usage)}")

        in_cost = in_tokens / 1000 * MODEL_COSTS[model]["in_cost"]
        out_cost = out_tokens / 1000 * MODEL_COSTS[model]["out_cost"]

        self.in_tokens += in_tokens
        self.out_tokens += out_tokens
        self.cost += in_cost + out_cost

    def update_from_messages(
        self,
        in_messages: List[Message],
        out_message: str,
        model: str,
    ) -> None:
        """
        Update the usage tracker with the cost of the input and output tokens.
        Uses the GPT-4o tokenizer, so might be inaccurate for other models.

        Args:
            usage_tracker (UsageTracker): The current usage tracker.
            in_messages (List[Message]): The input messages.
            out_message (str): The output message.
            model (str): The model used for the completion.

        Returns:
            UsageTracker: The updated usage tracker.

        Raises:
            ValueError: If the model is not found in MODEL_COSTS.
        """
        if model not in MODEL_COSTS:
            return

        in_tokens = sum(len(tokenizer.encode(msg["content"])) for msg in in_messages)
        out_tokens = len(tokenizer.encode(out_message))

        in_cost = in_tokens / 1000 * MODEL_COSTS[model]["in_cost"]
        out_cost = out_tokens / 1000 * MODEL_COSTS[model]["out_cost"]
        total_cost = in_cost + out_cost

        self.in_tokens += in_tokens
        self.out_tokens += out_tokens
        self.cost += total_cost

    def update_with_tracker(self, tracker: "UsageTracker") -> None:
        """Update the usage tracker with the values from another tracker."""
        self.in_tokens += tracker.in_tokens
        self.out_tokens += tracker.out_tokens
        self.cost += tracker.cost

    def save(self, path: str) -> None:
        """Save the usage tracker to a file."""
        if path.endswith(".yaml"):
            with open(path, "w") as f:
                yaml.dump(asdict(self), f)
        elif path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(asdict(self), f)
        else:
            raise ValueError("Unsupported file format.")

    @staticmethod
    def load(path: str) -> "UsageTracker":
        """Load a usage tracker from a file."""
        if path.endswith(".yaml"):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        elif path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format.")

        return UsageTracker(**data)

    @staticmethod
    def from_json(data: Union[Dict, str]) -> "UsageTracker":
        """Load a usage tracker from a JSON string."""
        if isinstance(data, str):
            data = json.loads(data)
        assert isinstance(data, dict)
        return UsageTracker(**data)

    def to_json(self) -> dict:
        """Return a JSON representation of the usage tracker."""
        return asdict(self)

    def __str__(self):
        """Return a string representation of the usage tracker."""
        return f"Usage Tracker:\nInput tokens: {self.in_tokens}\nOutput tokens: {self.out_tokens}\nTotal cost: {self.cost}"
