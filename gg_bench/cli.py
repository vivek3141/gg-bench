import sys
import os
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: gg-bench <command> [options]")
        print("Available commands: train, eval, generate")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "train":
        subprocess.run(["python", "gg_bench/scripts/train/train.py", *args])
    elif command == "eval":
        subprocess.run(["python", "gg_bench/scripts/eval/eval.py", *args])
    elif command == "generate":
        assert (
            len(args) == 1
        ), f"Expected one argument, got {len(args)}. Usage: gg-bench generate <pipeline_type>"
        assert args[0] in {
            "descriptions",
            "actions",
            "envs",
        }, f"Invalid pipeline type '{args[0]}'. Must be one of: descriptions, actions, envs."

        if args[0] == "descriptions":
            subprocess.run(
                ["python", "gg_bench/scripts/generate/generate_descriptions.py"]
            )
        elif args[0] == "actions":
            subprocess.run(["python", "gg_bench/scripts/generate/generate_actions.py"])
        elif args[0] == "envs":
            subprocess.run(["python", "gg_bench/scripts/generate/generate_envs.py"])
    elif command == "filter":
        subprocess.run(["python", "gg_bench/scripts/filter/filter.py", *args])
    elif command == "upper-bound-filter":
        assert (
            len(args) == 0
        ), f"Expected no arguments, got {len(args)}. Usage: gg-bench upper-bound-filter"
        os.system(
            "python gg_bench/scripts/experiments/agent_rank_analysis/agent_rank_analysis.py"
        )
        os.system(
            "python gg_bench/scripts/experiments/agent_rank_analysis/find_games.py"
        )
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval, generate")
        sys.exit(1)
