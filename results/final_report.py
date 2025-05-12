import json
import numpy as np
from typing import List, Tuple
import math


MODEL_EVAL_FILES = {
    "gpt-4o": "results/gpt-4o_eval_20250305_232011.json",
    "gpt-4o-mini": "results/gpt-4o-mini_eval_20250304_204004.json",
    "o1": "results/o1_eval_20250422_061810.json",
    "o3-mini": "results/o3-mini_eval_20250307_021443.json",
    "claude-3.7-sonnet": "results/claude-3.7-sonnet_eval_20250307_021235.json",
    "llama3.3-70b": "results/llama3.3-70b_eval_20250323_223649.json",
    "deepseek-r1": "results/deepseek-reasoner_eval_20250326_041314.json",
}


def mean_confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    n = len(data)
    if n == 0:
        raise ValueError("Data list is empty.")

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_error = math.sqrt(variance / n)

    # Z-score for 95% confidence
    z = 1.96

    margin = z * std_error
    return (mean - margin, mean + margin)


def compute_statistics(file_path: str):
    """Computes statistics for a given evaluation file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    gpt_wins = agent_wins = draws = 0
    gpt_faults = agent_faults = env_faults = 0
    gpt_win_vec = []
    environments = set()

    for game_idx, stats in data.items():
        if not game_idx.isnumeric():
            continue

        total = stats["gpt_wins"] + stats["agent_wins"] + stats["draws"]
        if total > 0:
            gpt_win_vec.append(stats["gpt_wins"] / total)

        gpt_wins += stats["gpt_wins"]
        agent_wins += stats["agent_wins"]
        draws += stats["draws"]

        gpt_faults += stats["gpt_faults"]
        agent_faults += stats["agent_faults"]
        env_faults += stats["env_faults"]
        
        if stats["total_games"] != 30:
            print(f"Faulty game {file_path} {game_idx}: {stats['total_games']} games")
            
        environments.add(game_idx)

    total_games = gpt_wins + agent_wins + draws
    gpt_win_rate = (gpt_wins / total_games) if total_games > 0 else 0

    if gpt_win_vec:
        mean = np.mean(gpt_win_vec)
        std = np.std(gpt_win_vec)
    else:
        mean = std = 0

    return {
        # "win_rate": gpt_win_rate,
        "win_rate": mean,
        "win_rate_std": std,
        "confidence_interval": mean_confidence_interval(gpt_win_vec),
        "total_games": total_games,
        "gpt_wins": gpt_wins,
        "agent_wins": agent_wins,
        "draws": draws,
        "gpt_faults": gpt_faults,
        "agent_faults": agent_faults,
        "env_faults": env_faults,
        "num_environments": len(environments),
    }


def print_final_report():
    """Prints a nicely formatted report of all models."""
    print("\n=== Final Report ===\n")

    for model, file_path in MODEL_EVAL_FILES.items():
        stats = compute_statistics(file_path)
        total_games = stats["total_games"]
        gpt_win_pct = (stats["gpt_wins"] / total_games) * 100 if total_games > 0 else 0
        agent_win_pct = (
            (stats["agent_wins"] / total_games) * 100 if total_games > 0 else 0
        )
        draw_pct = (stats["draws"] / total_games) * 100 if total_games > 0 else 0

        print(
            f"{model.upper()} Win Rate: {stats['win_rate']:.2%} std={stats['win_rate_std']:.2%}"
        )
        print(
            f"  Breakdown: GPT Wins={stats['gpt_wins']} ({gpt_win_pct:.2f}%), Agent Wins={stats['agent_wins']} ({agent_win_pct:.2f}%), Draws={stats['draws']} ({draw_pct:.2f}%), Total Games={stats['total_games']}"
        )
        print(
            f"  Confidence Interval: {stats['confidence_interval'][0]:.2%} - {stats['confidence_interval'][1]:.2%}, +- = {(abs(stats['confidence_interval'][1] - stats['confidence_interval'][0])/2):.2%}"
        )
        print(
            f"  Faults: GPT={stats['gpt_faults']}, Agent={stats['agent_faults']}, Env={stats['env_faults']}"
        )
        print(f"  Number of Environments: {stats['num_environments']}\n")


if __name__ == "__main__":
    print_final_report()
