import os
import importlib

from collections import defaultdict

from sbx import PPO
from tqdm.rich import tqdm

# Internal Imports
from gg_bench.utils.env_wrappers import MetadataEnv, TimeoutEnv
from gg_bench.utils.elo_system import EloSystem
from gg_bench.utils.inference import get_prediction, get_mcts_prediction


def load_checkpointed_models(checkpoint_dir):
    models = {}
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".zip"):
            model_path = os.path.join(checkpoint_dir, filename)
            model = PPO.load(model_path)

            step_count = filename.split("_")[1].split(".")[0]
            models[step_count] = model
    return models


def play_game(model1, model2, env, player1_first=True, use_mcts=False):
    obs, _ = env.reset()
    done = False
    reward = float("inf")
    while not done:
        if player1_first:
            if use_mcts:
                action = get_mcts_prediction(env, model1)
            else:
                action = get_prediction(model1, obs, env.valid_moves())
            obs, reward, done, _, _ = env.step(action)
            if done:
                assert reward in [0, 1]
                return reward
            if use_mcts:
                action = get_mcts_prediction(env, model2)
            else:
                action = get_prediction(model2, obs, env.valid_moves())
        else:
            if use_mcts:
                action = get_mcts_prediction(env, model2)
            else:
                action = get_prediction(model2, obs, env.valid_moves())
            obs, reward, done, _, _ = env.step(action)
            if done:
                assert reward in [0, 1]
                return -reward
            if use_mcts:
                action = get_mcts_prediction(env, model1)
            else:
                action = get_prediction(model1, obs, env.valid_moves())

        obs, reward, done, _, _ = env.step(action)
        if done:
            assert reward in [0, 1]
            return ((-1) ** player1_first) * reward


def tournament(models, env, games_per_match=5, use_mcts=False):
    elo_system = EloSystem()
    results = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0})
    n = len(models)
    total_steps = n * (n - 1) * games_per_match

    with tqdm(total=total_steps, desc="Tournament") as pbar:
        for model1_name, model1 in models.items():
            for model2_name, model2 in models.items():
                if model1_name != model2_name:
                    for i in range(games_per_match):
                        reward = play_game(
                            model1=model1,
                            model2=model2,
                            env=env,
                            player1_first=i % 2 == 0,
                            use_mcts=use_mcts,
                        )
                        assert reward in [1, -1, 0], f"Invalid reward: {reward}"
                        if reward > 0:
                            results[model1_name]["wins"] += 1
                            results[model2_name]["losses"] += 1
                            elo_system.update_ratings(model1_name, model2_name, 1)
                        elif reward < 0:
                            results[model1_name]["losses"] += 1
                            results[model2_name]["wins"] += 1
                            elo_system.update_ratings(model1_name, model2_name, 0)
                        else:
                            results[model1_name]["draws"] += 1
                            results[model2_name]["draws"] += 1
                            elo_system.update_ratings(model1_name, model2_name, 0.5)
                        pbar.update(1)

    return elo_system, results


def select_best_model(checkpoint_dir, env):
    models = load_checkpointed_models(checkpoint_dir)
    elo_system, results = tournament(models, env)

    for model_name, stats in results.items():
        print(f"{model_name}:")
        print(
            f"  Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}"
        )
        print(f"  Elo rating: {elo_system.ratings[model_name]:.2f}")
        print()

    best_model_name = max(elo_system.ratings, key=elo_system.ratings.get)
    best_model = models[best_model_name]

    print(f"Best model: {best_model_name}")
    print(f"Best model Elo rating: {elo_system.ratings[best_model_name]:.2f}")


if __name__ == "__main__":
    env_id = 5
    env_module = importlib.import_module(f"gg_bench.data.envs.env_{env_id}")
    env = env_module.CustomEnv()
    env = TimeoutEnv(env)
    env = MetadataEnv(env)
    select_best_model(f"gg_bench/data/models/env_{env_id}", env)
