import argparse
import fcntl
import importlib
import json
import logging
import os
import subprocess
import numpy as np
import sys

import multiprocessing as mp

from typing import Any, Optional, Tuple
from datetime import datetime
from sbx import DQN, PPO
from tqdm.rich import tqdm

# Internal Imports
from gg_bench.utils.game_stats import GameStats
from gg_bench.utils.chat_completion import Message, chat_completion, UsageTracker
from gg_bench.utils.load_yaml import load_yaml
from gg_bench.utils.env_wrappers import MetadataEnv, TimeoutEnv
from gg_bench.utils.inference import get_mcts_prediction

config = load_yaml("gg_bench/configs/run_eval.yaml")
instruction_prompt = config["instruction_prompt"]
turn_prompt = config["turn_prompt"]
num_games_per_env = config["num_games_per_env"]
max_turns_per_game = config["max_turns_per_game"]


def setup_logger():
    """Set up the logger."""
    logger = logging.getLogger("train_all")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def get_turn_prompt(board: str, legal_moves: list) -> str:
    """Generate the turn prompt for the GPT model."""
    return turn_prompt.replace("<BoardState>", board).replace(
        "<LegalMoves>", str(legal_moves)
    )


def get_formatted_instruction_prompt(env_id: int) -> str:
    """Load and format the instruction prompt for a given environment."""
    with open(f"gg_bench/data/descriptions/{env_id}.txt", "r") as f:
        game_description = f.read().strip()
    with open(f"gg_bench/data/actions/{env_id}.txt", "r") as f:
        move_description = f.read().strip()
    return instruction_prompt.replace("<GameDescription>", game_description).replace(
        "<MoveDescription>", move_description
    )


def make_agent_move(
    *,
    obs: Any,
    env: Any,
    agent: Any,
    stats: GameStats,
    verbose: bool = False,
    deterministic: bool = False,
) -> Tuple[Any, bool, Optional[float]]:
    """Make a move for the agent and return the new state."""
    action = get_mcts_prediction(env, agent, max_iter=300)

    if verbose:
        print(f"Agent move: {action}")
        print(env.render())

    try:
        obs, reward, done, _, _ = env.step(action)
    except Exception:
        stats.env_faults += 1
        return obs, True, None

    if done and verbose:
        print(env.render())

    return obs, done, reward


def make_gpt_move(
    *,
    obs: Any,
    env: Any,
    stats: GameStats,
    formatted_instruction_prompt: str,
    model: str,
    max_attempts: int = 5,
    verbose: bool = False,
    usage_tracker: Optional[UsageTracker] = None,
) -> Tuple[Any, bool, Optional[float]]:
    """Make a move for the GPT model and return the new state."""
    sorted_valid_moves = sorted(env.valid_moves())
    turn_prompt = get_turn_prompt(env.render(), sorted_valid_moves)

    if not sorted_valid_moves:
        stats.env_faults += 1
        return obs, True, None
    elif len(sorted_valid_moves) == 1:
        # If there is only one valid move, make it. LMs tend to get confused
        # between the list of valid moves and other numbers present in the prompt
        action = sorted_valid_moves[0]
    else:
        action = None
        for _ in range(max_attempts):
            messages = [
                Message(role="system", content=formatted_instruction_prompt),
                Message(role="user", content=turn_prompt),
            ]
            response = chat_completion(model, messages, usage_tracker=usage_tracker)
            if not response.isnumeric():
                continue

            action = int(response)
            if action in sorted_valid_moves:
                break

        if action not in sorted_valid_moves:
            return obs, True, -10

    if verbose:
        print(f"GPT move: {action}")
        print(env.render())

    try:
        obs, reward, done, _, _ = env.step(action)
    except Exception:
        stats.env_faults += 1
        return obs, True, None

    if done and verbose:
        print(env.render())

    return obs, done, reward


def play_game(
    *,
    env: Any,
    agent: Any,
    stats: GameStats,
    formatted_instruction_prompt: str,
    model: str,
    llm_first: bool = False,
    verbose: bool = False,
    usage_tracker: Optional[UsageTracker] = None,
) -> int:
    """Play a single game and return the number of turns played."""
    obs, _ = env.reset()
    turns = 0

    if verbose:
        print(f"Starting new game... LLM First: {llm_first}")

    if llm_first:
        obs, done, reward = make_gpt_move(
            obs=obs,
            env=env,
            stats=stats,
            model=model,
            formatted_instruction_prompt=formatted_instruction_prompt,
            verbose=verbose,
            usage_tracker=usage_tracker,
        )
        turns += 1
        if done:
            stats.update(reward or 0, "gpt", turns)
            return turns

    for _ in range(max_turns_per_game):
        obs, done, reward = make_agent_move(
            obs=obs, env=env, agent=agent, stats=stats, verbose=verbose
        )
        turns += 1
        if done:
            stats.update(reward or 0, "agent", turns)
            return turns

        obs, done, reward = make_gpt_move(
            obs=obs,
            env=env,
            stats=stats,
            model=model,
            formatted_instruction_prompt=formatted_instruction_prompt,
            verbose=verbose,
            usage_tracker=usage_tracker,
        )
        turns += 1
        if done:
            stats.update(reward or 0, "gpt", turns)
            return turns

    stats.env_faults += 1
    return turns


def main_single_env(args):
    """Run the evaluation for a specific environment."""
    env_id = args.env_id
    checkpoint_step = args.checkpoint_step
    verbose = args.verbose
    out_file = args.out_file

    with open(out_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    if str(env_id) in data:
        stats = GameStats.from_json(data[str(env_id)])
    else:
        stats = GameStats()

    usage_tracker = UsageTracker()

    env_module = importlib.import_module(f"gg_bench.data.envs.env_{env_id}")
    env = env_module.CustomEnv()
    env = TimeoutEnv(env)
    env = MetadataEnv(env)

    formatted_instruction_prompt = get_formatted_instruction_prompt(env_id)
    agent = PPO.load(
        f"gg_bench/data/models/env_{env_id}/agent_{checkpoint_step}", device="cpu"
    )
    if not hasattr(agent.policy, "actor_state"):
        print("Using DQN agent")
        agent = DQN.load(f"gg_bench/data/models/env_{env_id}/best_model", device="cpu")

    for game in tqdm(range(num_games_per_env - stats.total_games)):
        if verbose:
            print(f"Starting game {game}!")

        num_turns = play_game(
            env=env,
            agent=agent,
            stats=stats,
            formatted_instruction_prompt=formatted_instruction_prompt,
            model=args.model,
            llm_first=game % 2 == 0,
            verbose=verbose,
            usage_tracker=usage_tracker,
        )

        with open(out_file, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = json.load(f)
                data[env_id] = json.loads(str(stats))
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        if verbose:
            print(f"Game ended after {num_turns} turns")
            print()

    if args.out_file:
        with open(out_file, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = json.load(f)
                data[env_id] = json.loads(str(stats))

                old_usage_tracker = UsageTracker.from_json(data["usage_tracker"])
                usage_tracker.update_with_tracker(old_usage_tracker)
                data["usage_tracker"] = usage_tracker.to_json()

                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    if verbose:
        print(stats)
        print(usage_tracker)


def batch(iterable, batch_size):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def launch_single_env(
    env_id: int,
    checkpoint_step: int,
    model: str,
    out_file: str,
    verbose: bool,
    sem,
) -> None:
    with sem:
        assert os.path.exists(
            f"gg_bench/data/models/env_{env_id}/agent_{checkpoint_step}.zip"
        )

        command = [
            sys.executable,
            "gg_bench/scripts/eval/eval.py",
            f"--env_id={env_id}",
            f"--out_file={out_file}",
            f"--model={model}",
            f"--checkpoint_step={checkpoint_step}",
        ] + (["--verbose"] if verbose else [])

        process = subprocess.Popen(command)
        process.wait()


def main_multi_env(args):
    if not os.path.exists("logs/eval"):
        os.makedirs("logs/eval")

    if not os.path.exists(args.out_file):
        with open(args.out_file, "w") as f:
            json.dump({"usage_tracker": UsageTracker().to_json()}, f)
        envs_to_exclude = []
    else:
        with open(args.out_file, "r") as f:
            data = json.load(f)
            envs_to_exclude = [
                int(key)
                for key in data.keys()
                if key.isnumeric() and data[key]["total_games"] >= num_games_per_env
            ]

    with open("gg_bench/data/splits/valid_envs.json") as f:
        valid_envs = json.load(f)

    valid_envs = [
        env
        for env in valid_envs
        if int(env[0]) not in envs_to_exclude
        and (not args.id_range or args.id_range[0] <= int(env[0]) <= args.id_range[1])
    ]

    logger.info(f"Starting evaluation for {len(valid_envs)} environments...")
    sem = mp.Semaphore(args.batch_size)

    processes = [
        mp.Process(
            target=launch_single_env,
            args=(
                env_id,
                ckpt,
                args.model,
                args.out_file,
                args.verbose,
                sem,
            ),
        )
        for env_id, ckpt in valid_envs
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    logger.info("All environments evaluated!")

    with open(args.out_file, "r") as f:
        data = json.load(f)

    consolidated_game_stats = GameStats()
    for key, game_stats in data.items():
        if (
            key.isnumeric()
            and game_stats["agent_wins"] + game_stats["gpt_wins"]
            >= 0.8 * game_stats["total_games"]
        ):
            consolidated_game_stats.update_with_other(GameStats.from_json(game_stats))

    data["consolidated"] = json.loads(str(consolidated_game_stats))
    with open(args.out_file, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser(
        description="Run evaluation for a specific environment."
    )
    parser.add_argument("--env_id", type=int, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--id_range", type=int, nargs=2, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        # default=f"results/{model}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    # Check if only one of env_id and id_range is provided, not both or none
    if not (args.env_id is None) ^ (args.id_range is None):
        raise ValueError("Please provide either env_id or id_range")

    if args.out_file is None:
        args.out_file = (
            f"results/{args.model}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    if args.env_id is not None:
        main_single_env(args)
    else:
        main_multi_env(args)
