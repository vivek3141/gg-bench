import argparse
import importlib
import logging
import os
import subprocess
import sys

from sbx import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from tqdm.rich import tqdm


# Internal Imports
from gg_bench.utils.callbacks import ProgressBarCallback
from gg_bench.utils.env_wrappers import (
    AlternatingAgentEnv,
    EnvTimeoutError,
    MetadataEnv,
    TimeoutEnv,
)

# Constants
DEVICE = "cpu"


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


def setup_environment(env_id):
    """Load and set up the environment."""
    env_module = importlib.import_module(f"gg_bench.data.envs.env_{env_id}")
    env = env_module.CustomEnv()
    env = TimeoutEnv(env)
    env = MetadataEnv(env)
    env = AlternatingAgentEnv(env)
    logger.info("Environment loaded and checked")
    return env


def create_callbacks(env, args, log_dir):
    """Create training callbacks."""
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=1.0, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=stop_train_callback,
        best_model_save_path=f"gg_bench/data/models/env_{args.env_id}/",
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=15,
    )
    return eval_callback


def train_model(env, args, log_dir):
    """Train the PPO model."""
    logger.info("Starting model training")
    try:
        pbar = tqdm(total=args.timesteps)
        pbar_callback = ProgressBarCallback(pbar)

        initial_epsilon = 1.0
        final_epsilon = 0.1
        epsilon_decay = (initial_epsilon - final_epsilon) / args.timesteps

        save_interval = args.timesteps // 4
        swap_interval = args.timesteps // 4

        agent = PPO("MlpPolicy", env, verbose=1)

        for i in range(0, args.timesteps, save_interval):
            steps_remaining = save_interval
            while steps_remaining > 0:
                steps_to_learn = min(swap_interval, steps_remaining)

                current_epsilon = max(
                    final_epsilon, initial_epsilon - epsilon_decay * i
                )
                env.set_epsilon(current_epsilon)

                agent_params = agent.get_parameters().copy()
                new_agent = PPO("MlpPolicy", env, verbose=1)
                new_agent.set_parameters(agent_params)
                env.add_opposing_agent(new_agent)

                eval_callback = create_callbacks(env, args, log_dir)
                agent.learn(
                    total_timesteps=steps_to_learn,
                    callback=CallbackList([eval_callback, pbar_callback]),
                )
                agent = PPO.load(
                    f"gg_bench/data/models/env_{args.env_id}/best_model", env=env
                )
                steps_remaining -= steps_to_learn

            agent.save(
                f"gg_bench/data/models/env_{args.env_id}/agent_{i+save_interval}"
            )
            print(
                f"Saved checkpoint at {i+save_interval} steps, epsilon: {current_epsilon:.3f}"
            )

        os.system(f"rm gg_bench/data/models/env_{args.env_id}/best_model.zip")

    except Exception as e:
        if isinstance(e, EnvTimeoutError):
            raise e
        logger.info(f"Training stopped early: {str(e)}")
    logger.info("Model training completed")


def main_single_env(args):
    """Main function to run the training process."""
    logger.info(f"Starting training for environment {args.env_id}")

    # Setup environment
    env = setup_environment(args.env_id)

    # Create log directory
    log_dir = f"gg_bench/data/logs/env_{args.env_id}"
    os.makedirs(log_dir, exist_ok=True)

    try:
        # Train the model
        train_model(env, args, log_dir)
    except EnvTimeoutError:
        logger.info("Training stopped due to timeout")
        if os.path.exists(f"gg_bench/data/models/env_{args.env_id}/"):
            os.system(f"rm -r gg_bench/data/models/env_{args.env_id}/")


def main_multi_env(args):
    """Main function to run the training process for multiple environments."""
    if not os.path.exists("gg_bench/data/models"):
        os.makedirs("gg_bench/data/models")

    if not os.path.exists("logs/train"):
        os.makedirs("logs/train")

    logger.info(f"Going to train env_id: {args.id_range[0]} to {args.id_range[1]}")

    processes = []
    for env_id in range(args.id_range[0], args.id_range[1] + 1):
        command = [
            sys.executable,
            "gg_bench/scripts/train/train.py",
            f"--env_id={env_id}",
            f"--timesteps={args.timesteps}",
        ] + (["--verbose"] if args.verbose else [])
        process = subprocess.Popen(command)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            logger.error(f"Process failed with return code {process.returncode}")
            sys.exit(1)

    logger.info("All training processes completed")


if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=int, default=None)
    parser.add_argument("--id_range", type=int, nargs=2, default=None)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # Check if only one of env_id and id_range is provided, not both or none
    if not (args.env_id is None) ^ (args.id_range is None):
        raise ValueError("Please provide either env_id or id_range")
    if args.env_id is not None:
        main_single_env(args)
    else:
        main_multi_env(args)
