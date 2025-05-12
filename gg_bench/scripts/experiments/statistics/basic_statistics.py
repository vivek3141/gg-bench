import json
import numpy as np
from enum import Enum
import tiktoken
import importlib
import gymnasium as gym
from gymnasium import spaces
import io
import contextlib

gpt_tokenizer = tiktoken.get_encoding("o200k_base")


class DataType(Enum):
    DESCRIPTIONS = "descriptions"
    ENVS = "envs"
    ACTIONS = "actions"


def get_valid_envs() -> list[int]:
    with open("gg_bench/data/splits/valid_envs.json") as f:
        valid_envs = json.load(f)
    valid_envs = [int(env[0]) for env in valid_envs]
    assert all(isinstance(env, int) for env in valid_envs)
    return valid_envs


def get_data(data_type: DataType, valid_envs: list[int]) -> list[str]:
    data = []
    for env_id in valid_envs:
        file_name = (
            f"env_{env_id}.py" if data_type == DataType.ENVS else f"{env_id}.txt"
        )
        with open(f"gg_bench/data/{data_type.value}/{file_name}", "r") as f:
            data.append(f.read().strip())
    return data


def get_basic_statistics(data: list[str], count_lines: bool = False) -> dict:
    lengths = [
        len(i.split("\n")) if count_lines else len(gpt_tokenizer.encode(i))
        for i in data
    ]
    return {
        "mean": np.mean(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
    }


def print_basic_statistics(
    name: str,
    data: list[str],
    count_lines: bool = False,
    round_decimals: int = 3,
    unit: str = "",
) -> None:
    stats = get_basic_statistics(data, count_lines)
    print(f"{name} statistics:")
    print(f"Mean: {round(stats['mean'], round_decimals)} {unit}".strip())
    print(f"Std: {round(stats['std'], round_decimals)} {unit}".strip())
    print(f"Min: {round(stats['min'], round_decimals)} {unit}".strip())
    print(f"Max: {round(stats['max'], round_decimals)} {unit}".strip())
    print("")


def get_action_space_size(env: gym.Env) -> int:
    action_space = env.action_space

    if isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(np.prod(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        return 2**action_space.n
    elif isinstance(action_space, spaces.Box):
        raise ValueError(
            "Cannot determine discrete size of continuous Box action space"
        )
    else:
        raise NotImplementedError(
            f"Action space type {type(action_space)} not supported"
        )


def get_action_space_statistics(env_ids: list[int]) -> dict:
    action_space_sizes = []
    for env_id in env_ids:
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                module = importlib.import_module(f"gg_bench.data.envs.env_{env_id}")
                env = module.CustomEnv()
                action_space_sizes.append(get_action_space_size(env))
        except Exception:
            continue

    return {
        "mean": np.mean(action_space_sizes),
        "std": np.std(action_space_sizes),
        "min": np.min(action_space_sizes),
        "max": np.max(action_space_sizes),
    }


def print_action_space_statistics(env_ids: list[int], round_decimals: int = 3):
    stats = get_action_space_statistics(env_ids)
    print("Action Space Size statistics:")
    print(f"Mean: {round(stats['mean'], round_decimals)}")
    print(f"Std: {round(stats['std'], round_decimals)}")
    print(f"Min: {round(stats['min'], round_decimals)}")
    print(f"Max: {round(stats['max'], round_decimals)}")
    print("")


def print_statistics_for_envs(valid_envs: list[int]) -> None:
    descriptions = get_data(DataType.DESCRIPTIONS, valid_envs)
    envs = get_data(DataType.ENVS, valid_envs)
    actions = get_data(DataType.ACTIONS, valid_envs)

    print_basic_statistics(
        "Descriptions", descriptions, count_lines=False, unit="tokens"
    )
    print_basic_statistics("Envs", envs, count_lines=True, unit="lines of code")
    print_basic_statistics("Actions", actions, count_lines=False, unit="tokens")
    print_action_space_statistics(valid_envs)


def main() -> None:
    print("STATISTICS BEFORE FILTERING:\n")
    all_valid_envs = list(range(1, 1001))
    print_statistics_for_envs(all_valid_envs)

    print("STATISTICS AFTER FILTERING:\n")
    valid_envs = get_valid_envs()
    print_statistics_for_envs(valid_envs)


if __name__ == "__main__":
    main()
