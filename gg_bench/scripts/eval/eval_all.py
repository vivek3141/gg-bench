import argparse
import json
import logging
import logging.handlers
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime

# Internal Imports
from gg_bench.utils.game_stats import GameStats
from gg_bench.utils.chat_completion import UsageTracker


def setup_child_logger(queue):
    logger = logging.getLogger("eval_all")
    queue_handler = logging.handlers.QueueHandler(queue)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(queue_handler)
    return logger


def setup_listener_logger():
    logger = logging.getLogger("eval_all")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        f"logs/eval/eval_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def run_script(env_id, verbose, out_file, queue):
    logger = setup_child_logger(queue)
    logger.info(f"Starting eval for environment {env_id} writing to {out_file}")

    command = [
        sys.executable,
        "gg_bench/scripts/eval/eval.py",
        f"--env_id={env_id}",
        f"--out_file={out_file}",
    ] + (["--verbose"] if verbose else [])

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Finished eval for environment {env_id}")
        logger.info(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred for environment {env_id}: {e.stderr}")


def listener_process(queue):
    logger = setup_listener_logger()
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger.handle(record)
        except Exception:
            break


if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("logs/eval"):
        os.makedirs("logs/eval")

    parser = argparse.ArgumentParser()
    parser.add_argument("--id_range", type=int, nargs=2, required=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--out_file",
        type=str,
        default=f"results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    args = parser.parse_args()

    if os.path.exists(args.out_file):
        raise ValueError(f"Output file {args.out_file} already exists!")

    with open(args.out_file, "w") as f:
        json.dump({"usage_tracker": UsageTracker().to_json()}, f)

    manager = multiprocessing.Manager()
    queue = manager.Queue(-1)

    listener = multiprocessing.Process(target=listener_process, args=(queue,))
    listener.start()

    logger = setup_child_logger(queue)
    logger.info(f"Running env_id: {args.id_range[0]} to {args.id_range[1]}")

    processes = []
    for env_id in range(args.id_range[0], args.id_range[1] + 1):
        if os.path.exists(f"gg_bench/data/models/env_{env_id}/agent_1000000.zip"):
            command = [
                sys.executable,
                "gg_bench/scripts/eval/eval.py",
                f"--env_id={env_id}",
                f"--out_file={args.out_file}",
            ] + (["--verbose"] if args.verbose else [])
            process = subprocess.Popen(command)
            processes.append(process)

    for process in processes:
        process.wait()

    logger.info("All environments evaluated!")
    queue.put(None)
    listener.join()

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
