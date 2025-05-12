import importlib
import re
import signal
import warnings
from typing import Any, Dict, List

import tqdm
import json
from stable_baselines3.common.env_checker import check_env

# Internal Imports


def timeout_handler(signum: int, frame: Any) -> None:
    """
    Handler for timeout signal.

    Args:
        signum (int): The signal number.
        frame (Any): The current stack frame.

    Raises:
        TimeoutError: When execution times out.
    """
    raise TimeoutError("Execution timed out")


def execution_filter(env_idx: int, game_length: int = 5, timeout: float = 5.0) -> bool:
    """
    Filter environments based on execution. This function checks for
    the following:
    1. The environment runs without any errors.
    2. The action space and observation space are expected, and work with env.step
    3. The environment doesn't divide by zero when computing rewards.
    4. The environment doesn't have an infinite loop.
    5. The action space is not exponential.

    Args:
        env_idx (int): The index of the environment.
        game_length (int): The number of steps to run the environment for.
        timeout (float): The maximum execution time allowed.

    Returns:
        bool: True if the environment passes the filter, False otherwise.
    """
    # First, filter out any games with an exponential action space
    with open(f"gg_bench/data/envs/env_{env_idx}.py", "r") as f:
        env_code = f.read()

    pattern = r"spaces\.Discrete\(([^)]*)\)"
    match = re.search(pattern, env_code)

    if match and "**" in match.group(1):
        return False

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            env_module = importlib.import_module(f"gg_bench.data.envs.env_{env_idx}")
            env = env_module.CustomEnv()
            check_env(env)

            for _ in range(game_length):
                valid_moves = env.valid_moves()
                env.step(valid_moves[0])
                out_str = env.render()
                assert out_str is not None and out_str != ""

            for warning in w:
                if issubclass(
                    warning.category, RuntimeWarning
                ) and "divide by zero" in str(warning.message):
                    return False
        return True
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Error in env {env_idx}: {e}")
        return False
    finally:
        signal.alarm(0)  # Cancel the alarm


def run_execution_filtering(valid_envs_file: str) -> None:
    """
    Run execution filtering on a list of environments provided in a file.

    Args:
        valid_envs_file (str): The path to the file containing valid environments.
    """
    valid_envs_data: Dict[str, List[int]] = json.loads(open(valid_envs_file).read())

    filtered_valid_envs: List[int] = []
    failed_envs: List[int] = []
    for env_idx in tqdm.tqdm(valid_envs_data["valid_envs"]):
        if execution_filter(env_idx):
            filtered_valid_envs.append(env_idx)
        else:
            failed_envs.append(env_idx)

    num_filtered_envs = len(valid_envs_data["valid_envs"]) - len(filtered_valid_envs)
    print(f"Execution filtering completed. Filtered {num_filtered_envs} environments")

    valid_envs_data["valid_envs"] = filtered_valid_envs
    valid_envs_data["envs_failed_execution_filtering"] = failed_envs
    with open(valid_envs_file, "w") as f:
        json.dump(valid_envs_data, f)  # default_flow_style=False, width=50)
