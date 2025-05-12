import asyncio
import os

import black
import tqdm

# Internal Imports
from gg_bench.utils.chat_completion import Message, UsageTracker, chat_completion_async
from gg_bench.utils.load_yaml import load_yaml
from gg_bench.utils.markdown import extract_python_code


async def generate_env_and_save(
    game_idx: int,
    game_description: str,
    prompt: str,
    model: str,
    max_tokens: int,
    usage_tracker: UsageTracker,
    pbar: tqdm.tqdm,
) -> None:
    """
    Generate a game environment using the specified model and save it to a file.

    Args:
        game_idx (int): The index of the game being generated.
        game_description (str): The description of the game.
        prompt (str): The instruction prompt for generating the environment.
        model (str): The name of the model to use for chat completion.
        max_tokens (int): The maximum number of tokens to generate.
        usage_tracker (UsageTracker): The usage tracker object.
        pbar (tqdm.tqdm): A progress bar to update after generation.

    Returns:
        None

    This function generates a game environment using the chat_completion_async function,
    saves the environment to a file, and updates the progress bar.
    """
    prompt = prompt.replace("<GameDescription>", game_description)
    try:
        response = await chat_completion_async(
            model=model,
            messages=[Message(role="user", content=prompt)],
            max_tokens=max_tokens,
            usage_tracker=usage_tracker,
        )
    except Exception as e:
        print(f"Error generating env for idx {game_idx}: {e}")
        return
    env_code = extract_python_code(response.strip())
    if not env_code:
        raise ValueError(f"No Python code found for idx {game_idx}")
    try:
        env_code = black.format_str(env_code, mode=black.FileMode())
    except:
        pass
    with open(f"gg_bench/data/envs/env_{game_idx}.py", "w") as f:
        f.write(env_code)
    pbar.update(1)


async def main() -> None:
    config = load_yaml("gg_bench/configs/generate_envs.yaml")

    prompt = config["prompt"]
    model = config["model"]
    max_tokens = config["max_tokens"]
    num_games = config["num_games"]

    if not os.path.exists("gg_bench/data/envs"):
        os.makedirs("gg_bench/data/envs")
    if not os.path.exists("gg_bench/data/usage_trackers"):
        os.makedirs("gg_bench/data/usage_trackers")

    tasks = []
    usage_tracker = UsageTracker()
    game_indices = [
        i
        for i in range(1, num_games + 1)
        if not os.path.exists(f"gg_bench/data/envs/env_{i}.py")
        and os.path.exists(f"gg_bench/data/descriptions/{i}.txt")
    ]

    with tqdm.tqdm(total=num_games) as pbar:
        for game_idx in game_indices:
            with open(f"gg_bench/data/descriptions/{game_idx}.txt", "r") as f:
                game_description = f.read().strip()

            tasks.append(
                generate_env_and_save(
                    game_idx=game_idx,
                    game_description=game_description,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    usage_tracker=usage_tracker,
                    pbar=pbar,
                )
            )
        await asyncio.gather(*tasks)

    print("All game environments have been coded and saved.")
    print("Usage tracker:")
    print(f"In Tokens: {usage_tracker.in_tokens}")
    print(f"Out Tokens: {usage_tracker.out_tokens}")
    print(f"Cost: {usage_tracker.cost}")

    usage_tracker.save("gg_bench/data/usage_trackers/envs.yaml")


if __name__ == "__main__":
    asyncio.run(main())
