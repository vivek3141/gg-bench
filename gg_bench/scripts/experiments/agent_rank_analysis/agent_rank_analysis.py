import os
import importlib
import json
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from sbx import PPO
from gg_bench.utils.env_wrappers import MetadataEnv, TimeoutEnv, EnvTimeoutError
from gg_bench.utils.inference import get_prediction, get_mcts_prediction


USE_MCTS = True
GAMES_PER_MATCH = 6
models_base_dir = "gg_bench/data/models"


def load_checkpointed_models(checkpoint_dir):
    """
    Scans 'checkpoint_dir' for .zip files,
    loads each as a PPO model,
    returns a dict {step_count_str: PPO_model}.
    """
    models = {}
    if not os.path.isdir(checkpoint_dir):
        return models  # no directory => no models found

    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".zip"):
            model_path = os.path.join(checkpoint_dir, filename)
            model = PPO.load(model_path)
            # parse step count from e.g. "agent_250000.zip"
            step_str = filename.split("_")[1].split(".")[0]
            models[step_str] = model
    return models


def play_one_game(model1, model2, env, player1_first=True, use_mcts=False):
    """
    Plays a single game of model1 vs. model2 in 'env'.
    Returns +1 if model1 wins, -1 if model2 wins, 0 if draw.
    """
    obs, _ = env.reset()
    done = False

    def choose_action(model, obs):
        if use_mcts:
            # MCTS approach: run a tree search from current state
            return get_mcts_prediction(env, model, max_iter=100)
        else:
            # PPO approach: direct inference
            return get_prediction(model, obs, env.valid_moves())

    while not done:
        if player1_first:
            action = choose_action(model1, obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                return reward  # from model1's perspective
            action = choose_action(model2, obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                return -reward
        else:
            action = choose_action(model2, obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                return -reward
            action = choose_action(model1, obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                return reward

    return 0  # edge case: draw


def pairwise_matchups_unified(
    models, env, games_per_match=GAMES_PER_MATCH, use_mcts=False
):
    """
    For each distinct pair (i, j), i < j, of model keys:
      - plays 'games_per_match' games
      - alternates which model goes first
      - returns a nested dict: results[i][j] = fraction of times i beats j
    """
    # Sort model keys numerically (assuming they are step counts)
    model_keys = sorted(models.keys(), key=lambda x: int(x))
    results = {}
    for i_idx in range(len(model_keys)):
        m1_key = model_keys[i_idx]
        results.setdefault(m1_key, {})

        for j_idx in range(i_idx + 1, len(model_keys)):
            m2_key = model_keys[j_idx]
            model1 = models[m1_key]
            model2 = models[m2_key]

            i_wins = 0
            total_games = 0

            for game_idx in range(games_per_match):
                # Alternate who goes first
                player1_first = game_idx % 2 == 0
                reward = play_one_game(
                    model1, model2, env, player1_first=player1_first, use_mcts=use_mcts
                )
                if reward > 0:
                    i_wins += 1
                total_games += 1

            results[m1_key][m2_key] = i_wins / total_games if total_games > 0 else 0.0

    return results


def process_env(env_id):
    """
    Process a single environment matchup.
    Returns a tuple (env_id, result_dict) where result_dict has keys 'results' and 'error'.
    """
    try:
        env_module_path = f"gg_bench.data.envs.env_{env_id}"
        try:
            env_module = importlib.import_module(env_module_path)
        except ModuleNotFoundError:
            print(f"No environment code found for env_{env_id}. Skipping.")
            return env_id, {"results": None, "error": "ModuleNotFound"}

        env = env_module.CustomEnv()
        env = TimeoutEnv(env)
        env = MetadataEnv(env)

        checkpoint_dir = os.path.join(models_base_dir, f"env_{env_id}")
        models_dict = load_checkpointed_models(checkpoint_dir)
        if not models_dict:
            print(f"No models found in {checkpoint_dir}; skipping env_{env_id}.")
            return env_id, {"results": None, "error": "No models found"}

        print(f"\n=== Processing env_{env_id} (MCTS={USE_MCTS}) ===")

        results = pairwise_matchups_unified(
            models_dict, env, GAMES_PER_MATCH, use_mcts=USE_MCTS
        )

        for m1 in sorted(results.keys(), key=lambda x: int(x)):
            for m2 in sorted(results[m1].keys(), key=lambda x: int(x)):
                wr = results[m1][m2]
                print(f"  Agent {m1} vs Agent {m2}: {wr*100:.1f}%")

        # Report high win pairs
        high_win_pairs = []
        for m1 in results:
            for m2, wr in results[m1].items():
                if wr >= 0.8:
                    high_win_pairs.append((m1, m2, wr))
        if high_win_pairs:
            print("-- Agents with >=80% win rate --")
            for m1, m2, wr in high_win_pairs:
                print(f"  Agent {m1} beats Agent {m2} with {wr*100:.1f}% win rate!")
        else:
            print("-- No agent pairs had >=80% win rate --")

        normal_results = {m1: dict(results[m1]) for m1 in results}
        return env_id, {"results": normal_results, "error": None}

    except EnvTimeoutError as e:
        print(f"Env_{env_id} raised EnvTimeoutError: {e}. Skipping environment.")
        return env_id, {"results": None, "error": f"Timeout: {str(e)}"}
    except Exception as e:
        print(f"Env_{env_id} encountered an error: {e}. Skipping.")
        return env_id, {"results": None, "error": str(e)}


if __name__ == "__main__":
    # Load list of valid environments
    valid_envs_path = os.path.join("gg_bench", "data", "splits", "valid_envs.json")
    with open(valid_envs_path, "r") as f:
        valid_envs = json.load(f)

    master_results = {}

    with ProcessPoolExecutor() as executor:
        pbar = tqdm.tqdm(total=len(valid_envs))
        future_to_env = {
            executor.submit(process_env, env_id): env_id for env_id in valid_envs
        }
        for future in as_completed(future_to_env):
            env_id, result = future.result()
            master_results[env_id] = result
            pbar.update(1)

    json_results = {}
    for eid, content in master_results.items():
        if content["results"] is not None:
            res_dict = {}
            for agent1, subdict in content["results"].items():
                float_subdict = {agent2: float(wr) for agent2, wr in subdict.items()}
                res_dict[agent1] = float_subdict
            json_results[str(eid)] = {"results": res_dict, "error": content["error"]}
        else:
            json_results[str(eid)] = {"results": None, "error": content["error"]}

    output_path = "results/analysis/mcts_ranking_results.json"
    if not os.path.exists("results/analysis"):
        os.makedirs("results/analysis")

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nSaved ranking results to {output_path}")
    print("Finished all environment matchups!")
