import importlib
from typing import Dict, List, Optional

import numpy as np
import tqdm
import json

# Internal Imports
from gg_bench.utils.chat_completion import UsageTracker, chat_completion
from gg_bench.utils.load_yaml import load_yaml


def make_gpt_move(
    board_state: str,
    instruction_prompt: str,
    turn_prompt: str,
    valid_moves: List[int],
    model: str = "gpt-4o-mini",
    verbose: bool = False,
    usage_tracker: Optional[UsageTracker] = None,
):
    """
    Makes a GPT move based on the current board state and valid moves.

    Args:
        board_state (str): Current state of the game board.
        instruction_prompt (str): Prompt with game instructions.
        turn_prompt (str): Prompt for the current turn.
        valid_moves (List[int]): List of valid moves.
        model (str): GPT model to use. Default is "gpt-4o-mini".
        verbose (bool): If True, print debug information. Default is False.
        usage_tracker (Optional[UsageTracker]): Tracker for API usage.

    Returns:
        int: The chosen move.
    """
    turn_prompt = turn_prompt.replace("<BoardState>", board_state)
    turn_prompt = turn_prompt.replace("<ValidMoves>", str(valid_moves))

    move = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": turn_prompt},
        ],
        usage_tracker=usage_tracker,
    )

    try:
        move = int(move)
    except ValueError:
        if verbose:
            print(f"Couldn't parse {move} as integer. Taking random move")
        return np.random.choice(valid_moves)

    if move in valid_moves or str(move) in valid_moves:
        return move
    else:
        if verbose:
            print(
                f"Invalid move {move} detected from {valid_moves}. Taking random move"
            )
        return np.random.choice(valid_moves)


def timeout_filter(
    env_id: int,
    verbose: bool = False,
    usage_tracker: Optional[UsageTracker] = None,
):
    """
    Timeout filtering. Runs a random agent against GPT-4o-mini and filters out any
    games that take too long to complete.

    Args:
        env_id (int): The ID of the environment to test.
        verbose (bool): If True, print debug information. Default is False.
        usage_tracker (Optional[UsageTracker]): Tracker for API usage.

    Returns:
        bool: True if the game completes within the maximum number of turns,
              False otherwise.
    """
    run_eval_config: Dict = load_yaml("gg_bench/configs/run_eval.yaml")

    with open(f"gg_bench/data/descriptions/{env_id}.txt") as f:
        game_description = f.read().strip()

    with open(f"gg_bench/data/actions/{env_id}.txt") as f:
        action_description = f.read().strip()

    instruction_prompt: str = run_eval_config["instruction_prompt"]
    instruction_prompt = instruction_prompt.replace(
        "<GameDescription>", game_description
    )
    instruction_prompt = instruction_prompt.replace(
        "<MoveDescription>", action_description
    )

    env_module = importlib.import_module(f"gg_bench.data.envs.env_{env_id}")
    try:
        env = env_module.CustomEnv()
    except Exception as e:
        if verbose:
            print(f"Error creating env for {env_id}: {e}")
        return False

    for _ in range(run_eval_config["max_turns_per_game"]):
        try:
            valid_moves = env.valid_moves()
        except Exception as e:
            if verbose:
                print(f"Error getting valid moves for {env_id}: {e}")
            return False
        if verbose:
            print(env.render())
            print(f"Random Move: {valid_moves[0]}")

        try:
            _, _, done, _, _ = env.step(valid_moves[0])
        except Exception as e:
            if verbose:
                print(f"Error taking random move for {env_id}: {e}")
            return False
        if done:
            return True

        try:
            board_state = env.render()
        except Exception as e:
            if verbose:
                print(f"Error getting board state for {env_id}: {e}")
            return False

        move = make_gpt_move(
            board_state=board_state,
            instruction_prompt=instruction_prompt,
            turn_prompt=run_eval_config["turn_prompt"],
            valid_moves=valid_moves,
            usage_tracker=usage_tracker,
            verbose=verbose,
        )
        if verbose:
            print(env.render())
            print(f"GPT Move: {move}")

        try:
            _, _, done, _, _ = env.step(move)
        except Exception as e:
            if verbose:
                print(f"Error taking GPT move for {env_id}: {e}")
            return False
        if done:
            return True

    return False


def run_timeout_filtering(valid_envs_file: str, verbose: bool = False) -> None:
    """
    Run timeout filtering on a list of environments provided in a file.

    Args:
        valid_envs_file (str): The path to the file containing valid environments.
    """
    valid_envs_data: Dict = load_yaml(valid_envs_file)

    filtered_valid_envs: List[int] = []
    usage_tracker = UsageTracker()
    for env_id in tqdm.tqdm(valid_envs_data["valid_envs"]):
        if timeout_filter(env_id=env_id, usage_tracker=usage_tracker, verbose=verbose):
            filtered_valid_envs.append(env_id)

    num_filtered_envs = len(valid_envs_data["valid_envs"]) - len(filtered_valid_envs)
    print(f"Timeout filtering completed. Filtered {num_filtered_envs} environments")

    valid_envs_data["valid_envs"] = filtered_valid_envs
    with open(valid_envs_file, "w") as f:
        json.dump(valid_envs_data, f, default_flow_style=True, width=50)
    print(usage_tracker)
