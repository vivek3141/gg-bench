import json
import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def consolidate_results(
    input_dir: str,
    output_file: str,
    split: str = "hard",
) -> Dict[str, Any]:
    """
    Consolidate results from multiple JSON files.

    Args:
        input_dir (str): Directory containing input JSON files.
        output_file (str): Path to save consolidated results.
        acceptance_threshold (float): Threshold for accepting environment results.
        num_games_per_env (int): Number of games per environment.

    Returns:
        Dict[str, Any]: Consolidated results.

    Raises:
        FileNotFoundError: If the input directory is not found.
        json.JSONDecodeError: If there's an error parsing a JSON file.
    """
    if split == "hard":
        valid_envs = json.load(open("gg_bench/data/splits/hard.json"))
    elif split == "medium":
        valid_envs = json.load(open("gg_bench/data/splits/medium.json"))
    elif split == "easy":
        valid_envs = json.load(open("gg_bench/data/splits/easy.json"))
    else:
        raise ValueError(f"Invalid split: {split}")

    results = {}

    try:
        for file in os.listdir(input_dir):
            if file.endswith(".json"):
                model = file.split("_")[0]
                current_results = {
                    "agent_wins": 0,
                    "gpt_wins": 0,
                    "draws": 0,
                    "num_valid_envs": 0,
                    "valid_envs": [],
                }

                with open(os.path.join(input_dir, file), "r") as f:
                    current_model_results = json.load(f)

                for idx in valid_envs:
                    if str(idx) not in current_model_results:
                        continue

                    current_env_results = current_model_results[str(idx)]
                    for key in ["agent_wins", "gpt_wins", "draws"]:
                        current_results[key] += current_env_results[key]

                    current_results["num_valid_envs"] += 1

                # successful_games = sum(
                #     current_env_results[key]
                #     for key in ["agent_wins", "gpt_wins", "draws"]
                # )
                # if successful_games >= acceptance_threshold * num_games_per_env:
                #     for key in ["agent_wins", "gpt_wins", "draws"]:
                #         results[key] += current_env_results[key]
                #     valid_envs.append(int(file.split("_")[1].split(".")[0]))

                # current_results["num_valid_envs"] = len(valid_envs)
                # current_results["valid_envs"] = sorted(valid_envs)

                results[model] = current_results

        # with open(output_file, "w") as f:
        #     json.dump(results, f, indent=2)

        return results
    except FileNotFoundError:
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error parsing JSON file: {e}")


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the consolidated results.

    Args:
        results (Dict[str, Any]): Consolidated results dictionary.
    """
    # print(f"Consolidated results for {results['num_valid_envs']} environments:")
    # total_games = sum(results[key] for key in ["agent_wins", "gpt_wins", "draws"])

    for model, model_results in results.items():
        # Just print model: win rate
        print(f"Model: {model}")
        print(f"Num Envs: {model_results['num_valid_envs']}")
        if model_results["num_valid_envs"] == 0:
            print("LLM Win Rate: 0.00%")
        else:
            total_games = (
                model_results["agent_wins"]
                + model_results["gpt_wins"]
                + model_results["draws"]
            )
            win_rate = model_results["gpt_wins"] / total_games
            print(f"LLM Win Rate: {win_rate:.2%}")
        print()

        # print(f"Number of valid environments: {model_results['num_valid_envs']}")
        # if model_results["num_valid_envs"] == 0:
        #     print(f"{model}: 0.00%")
        # else:
        #     print(
        #         f"{model}: {model_results['agent_wins'] / model_results['num_valid_envs']:.2%}"
        #     )

    # titles = {
    #     "agent_wins": "Agent Win Rate",
    #     "gpt_wins": "GPT Win Rate",
    #     "draws": "Draw Rate",
    # }
    # for key in ["agent_wins", "gpt_wins", "draws"]:
    #     if total_games == 0:
    #         print(f"{titles[key]}: 0.00%")
    #     else:
    #         print(f"{titles[key]}: {results[key] / total_games:.2%}")


def main(split: str = "hard") -> None:
    """
    Main function to run the consolidation process.
    """
    config = load_config("gg_bench/configs/consolidate.yaml")
    acceptance_threshold = config["acceptance_threshold"]
    num_games_per_env = config["num_games_per_env"]

    results = consolidate_results(
        input_dir="results",
        output_file="results/consolidated_results.json",
        # acceptance_threshold=acceptance_threshold,
        # num_games_per_env=num_games_per_env,
        split=split,
    )

    print(f"Results for split: {split}")
    print_summary(results)


if __name__ == "__main__":
    for split in ["easy", "medium", "hard"]:
        main(split)
