import argparse
import os

import json

# Internal Imports
from gg_bench.scripts.filter.execution import run_execution_filtering
from gg_bench.scripts.filter.timeout import run_timeout_filtering

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_envs_file", type=str, default="gg_bench/data/valid_envs.json"
    )
    parser.add_argument("--execution", action="store_true")
    parser.add_argument("--timeout", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    num_of_envs = max(
        [int(file.split(".")[0]) for file in os.listdir("gg_bench/data/descriptions")]
    )

    if not os.path.exists(args.valid_envs_file):
        with open(args.valid_envs_file, "w") as f:
            json.dump(
                {"valid_envs": list(range(1, num_of_envs + 1))},
                f,
                # default_flow_style=False,
                # width=50,
                # indent=4,
            )

    if not any([args.execution, args.timeout, args.all]):
        args.all = True

    if args.execution or args.all:
        run_execution_filtering(valid_envs_file=args.valid_envs_file)

    if args.timeout or args.all:
        run_timeout_filtering(valid_envs_file=args.valid_envs_file)
