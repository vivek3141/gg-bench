import asyncio
import os

import tqdm

# Internal imports
from gg_bench.utils.chat_completion import Message, chat_completion_async, UsageTracker
from gg_bench.utils.load_yaml import load_yaml


async def generate_actions_and_save(
    game_idx: int,
    description: str,
    env_code: str,
    prompt: str,
    model: str,
    max_tokens: int,
    usage_tracker: UsageTracker,
    pbar: tqdm.tqdm,
) -> None:
    """
    Generate action descriptions for a game environment and save them to a file.

    Args:
        game_idx (int): The index of the game being processed.
        description (str): The description of the game.
        env_code (str): The Python code for the game environment.
        prompt (str): The instruction prompt for generating actions.
        model (str): The name of the model to use for chat completion.
        max_tokens (int): The maximum number of tokens to generate.
        usage_tracker (UsageTracker): The usage tracker object.
        pbar (tqdm.tqdm): A progress bar to update after generation.

    Returns:
        None
    """
    prompt = prompt.replace("<GameDescription>", description)
    prompt = prompt.replace("<PythonCode>", env_code)
    try:
        action_description = await chat_completion_async(
            model=model,
            messages=[Message(role="user", content=prompt)],
            max_tokens=max_tokens,
            usage_tracker=usage_tracker,
        )
    except Exception as e:
        print(f"Error generating actions for idx {game_idx}: {e}")
        return

    with open(f"gg_bench/data/actions/{game_idx}.txt", "w") as f:
        f.write(action_description.strip())
    pbar.update(1)


async def main() -> None:
    config = load_yaml("gg_bench/configs/generate_actions.yaml")

    prompt = config["instruction_prompt"]
    model = config["model"]
    max_tokens = config["max_tokens"]
    num_games = config["num_games"]

    if not os.path.exists("gg_bench/data/actions"):
        os.makedirs("gg_bench/data/actions")
    if not os.path.exists("gg_bench/data/usage_trackers"):
        os.makedirs("gg_bench/data/usage_trackers")

    tasks = []
    usage_tracker = UsageTracker()
    game_indices = [
        i
        for i in range(1, num_games + 1)
        if not os.path.exists(f"gg_bench/data/actions/{i}.txt")
        and os.path.exists(f"gg_bench/data/descriptions/{i}.txt")
        and os.path.exists(f"gg_bench/data/envs/env_{i}.py")
    ]

    with tqdm.tqdm(total=num_games) as pbar:
        for game_idx in game_indices:
            with open(f"gg_bench/data/descriptions/{game_idx}.txt", "r") as f:
                description = f.read().strip()

            with open(f"gg_bench/data/envs/env_{game_idx}.py", "r") as f:
                env_code = f.read().strip()

            tasks.append(
                generate_actions_and_save(
                    game_idx=game_idx,
                    description=description,
                    env_code=env_code,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    usage_tracker=usage_tracker,
                    pbar=pbar,
                )
            )
        await asyncio.gather(*tasks)

    print("All game actions have been generated and saved.")
    print("Usage tracker:")
    print(f"In Tokens: {usage_tracker.in_tokens}")
    print(f"Out Tokens: {usage_tracker.out_tokens}")
    print(f"Cost: {usage_tracker.cost}")

    usage_tracker.save("gg_bench/data/usage_trackers/actions.yaml")


if __name__ == "__main__":
    asyncio.run(main())
