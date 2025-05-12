import asyncio
import os

import tqdm

# Internal Imports
from gg_bench.utils.chat_completion import Message, UsageTracker, chat_completion_async
from gg_bench.utils.load_yaml import load_yaml


async def generate_description_and_save(
    game_idx: int,
    prompt: str,
    model: str,
    max_tokens: int,
    usage_tracker: UsageTracker,
    pbar: tqdm.tqdm,
) -> None:
    """
    Generate a game description using the specified model and save it to a file.

    Args:
        game_idx (int): The index of the game being generated.
        prompt (str): The prompt to use for generating the game description.
        model (str): The name of the model to use for chat completion.
        max_tokens (int): The maximum number of tokens to generate.
        pbar (tqdm.tqdm): A progress bar to update after generation.

    Returns:
        None

    This function generates a game description using the chat_completion_async function,
    saves the description to a file, and updates the progress bar.
    """
    description = await chat_completion_async(
        model=model,
        messages=[Message(role="user", content=prompt)],
        max_tokens=max_tokens,
        usage_tracker=usage_tracker,
    )
    with open(f"gg_bench/data/descriptions/{game_idx}.txt", "w") as f:
        f.write(description.strip())
    pbar.update(1)


async def main() -> None:
    config = load_yaml("gg_bench/configs/generate_descriptions.yaml")

    prompts = config["prompts"]
    model = config["model"]
    max_tokens = config["max_tokens"]
    num_games = config["num_games"]

    if not os.path.exists("gg_bench/data/descriptions"):
        os.makedirs("gg_bench/data/descriptions")
    if not os.path.exists("gg_bench/data/usage_trackers"):
        os.makedirs("gg_bench/data/usage_trackers")

    tasks = []
    usage_tracker = UsageTracker()

    with tqdm.tqdm(total=num_games) as pbar:
        for game_idx in range(1, num_games + 1):
            if os.path.exists(f"gg_bench/data/descriptions/{game_idx}.txt"):
                pbar.update(1)
                continue
            tasks.append(
                generate_description_and_save(
                    game_idx=game_idx,
                    prompt=prompts[game_idx % len(prompts)].strip(),
                    model=model,
                    max_tokens=max_tokens,
                    usage_tracker=usage_tracker,
                    pbar=pbar,
                )
            )
        await asyncio.gather(*tasks)

    print("All game descriptions have been generated and saved.")
    print("Usage tracker:")
    print(f"In Tokens: {usage_tracker.in_tokens}")
    print(f"Out Tokens: {usage_tracker.out_tokens}")
    print(f"Cost: {usage_tracker.cost}")

    usage_tracker.save("gg_bench/data/usage_trackers/descriptions.yaml")


if __name__ == "__main__":
    asyncio.run(main())
