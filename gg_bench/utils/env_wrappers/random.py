from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, Wrapper


class RandomAgentEnv(Wrapper):
    """
    A wrapper for environments that adds a random agent opponent.

    This wrapper alternates turns between the player and a random agent,
    modifying the environment to create a two-player game scenario.
    """

    def __init__(self, env: Env):
        """
        Initialize the RandomAgentEnv wrapper.

        Args:
            env (Env): The environment to wrap.
        """
        super().__init__(env)
        self.random_first: bool = False

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and determine if the random agent goes first.

        Args:
            seed (Optional[int]): The seed for the random number generator.
            options (Optional[Dict[str, Any]]): Additional options for resetting the environment.

        Returns:
            Tuple[Any, Dict[str, Any]]: The initial observation and info dictionary.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.random_first = np.random.choice([True, False])
        if self.random_first:
            action = np.random.choice(self.valid_moves())
            obs, _, _, _, info = self.env.step(action)
        return obs, info

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment, including the random agent's move if applicable.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: The observation, reward, terminated flag,
                                                           truncated flag, and info dictionary.
        """
        # Player's turn
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if not done:
            # Random agent's turn
            valid_moves = self.valid_moves()
            if not valid_moves:
                return obs, 1, True, False, info
            random_action = np.random.choice(valid_moves)
            obs, reward, terminated, truncated, info = self.env.step(random_action)
            done = terminated or truncated
            reward = -reward if reward in [-1, 1] else reward

        return obs, reward, terminated, truncated, info
