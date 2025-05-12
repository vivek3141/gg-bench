from typing import List
from gg_bench.utils.load_yaml import load_yaml
import json

consolidate_config = load_yaml("gg_bench/configs/consolidate.yaml")

# Non Inclusive Thresholds
EASY_THRESHOLD = 0.0
MEDIUM_THRESHOLD = 4.0
HARD_THRESHOLD = 7.0

# Initial Eval
GPT_4O_MINI_EVAL = "results/gpt-4o-mini_eval_20241117_203351.json"


def get_ids_by_difficulty(difficulty: str, file_path: str) -> List[int]:
    assert difficulty in ["easy", "medium", "hard"]
    if difficulty == "easy":
        left_threshold = EASY_THRESHOLD
        right_threshold = MEDIUM_THRESHOLD
    elif difficulty == "medium":
        left_threshold = MEDIUM_THRESHOLD
        right_threshold = HARD_THRESHOLD
    else:
        left_threshold = HARD_THRESHOLD
        right_threshold = float("inf")

    with open(file_path, "r") as f:
        data = json.load(f)

    ids = []
    for game_idx, stats in data.items():
        if not game_idx.isnumeric():
            continue

        successful = stats["gpt_wins"] + stats["agent_wins"] + stats["draws"]
        if (
            left_threshold <= stats["avg_turns_per_game"] < right_threshold
            and successful
            >= consolidate_config["acceptance_threshold"] * stats["total_games"]
        ):
            ids.append(int(game_idx))

    return ids


def main():
    easy_ids = get_ids_by_difficulty("easy", GPT_4O_MINI_EVAL)
    medium_ids = get_ids_by_difficulty("medium", GPT_4O_MINI_EVAL)
    hard_ids = get_ids_by_difficulty("hard", GPT_4O_MINI_EVAL)

    with open("gg_bench/data/splits/easy.json", "w") as f:
        json.dump(easy_ids, f)

    with open("gg_bench/data/splits/medium.json", "w") as f:
        json.dump(medium_ids, f)

    with open("gg_bench/data/splits/hard.json", "w") as f:
        json.dump(hard_ids, f)

    print(f"Easy: {len(easy_ids)}")
    print(f"Medium: {len(medium_ids)}")
    print(f"Hard: {len(hard_ids)}")


if __name__ == "__main__":
    main()
