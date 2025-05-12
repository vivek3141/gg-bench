from typing import Any, Dict, Tuple, Union

import numpy as np
from gymnasium import Wrapper, spaces


class MetadataEnv(Wrapper):
    """
    A wrapper for Gymnasium environments that adds metadata to the observation space.

    This wrapper extends the observation space by adding a player indicator,
    and provides additional methods for game state management.
    """

    def __init__(self, env: Any):
        """
        Initialize the MetadataEnvWrapper.

        Args:
            env (Any): The Gymnasium environment to wrap.

        Raises:
            ValueError: If the observation space is not of type Box.
        """
        super().__init__(env)
        self.last_obs: np.ndarray | None = None
        self.last_reward: int | float | None = None
        self.current_player_indicator: int = 1
        self.is_game_over: bool = False

        if not isinstance(self.observation_space, spaces.Box):
            raise ValueError("Observation space must be of type Box")

        old_shape = self.observation_space.shape
        new_shape = (old_shape[0] + 1,) + old_shape[1:]

        self.observation_space = spaces.Box(
            low=np.append(self.observation_space.low, 0),
            high=np.append(self.observation_space.high, 1),
            shape=new_shape,
            dtype=np.float32,
        )

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return the initial observation.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The initial observation and info dictionary.
        """
        self.is_game_over = False
        self.current_player_indicator = 1

        obs, info = self.env.reset(*args, **kwargs)
        obs = self._add_player_indicator(obs)
        self.last_obs = obs
        self.last_reward = 0

        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, Any, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                The observation, reward, done flag, truncated flag, and info dictionary.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_player_indicator = -self.current_player_indicator
        obs = self._add_player_indicator(obs)
        self.last_obs = obs
        self.last_reward = reward
        if done:
            self.is_game_over = True
        return obs, reward, done, truncated, info

    def get_obs(self) -> np.ndarray:
        """
        Get the last observation.

        Returns:
            np.ndarray: The last observation.

        Raises:
            ValueError: If no observation is available.
        """
        if self.last_obs is None:
            raise ValueError("Observation is None. Call reset() or step() first.")
        return self.last_obs

    def get_reward(self) -> Union[float, int]:
        """
        Get the last reward.

        Returns:
            float: The last reward.

        Raises:
            ValueError: If no reward is available.
        """
        if self.last_reward is None:
            raise ValueError("Reward is None. Call reset() or step() first.")
        return self.last_reward

    def get_current_player(self) -> int:
        """
        Get the current player indicator.

        Returns:
            int: The current player indicator (1 or -1).
        """
        return self.current_player_indicator

    def game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.is_game_over

    def valid_moves(self) -> Any:
        """
        Get the valid moves for the current state.

        Returns:
            Any: The valid moves, as defined by the underlying environment.

        Raises:
            NotImplementedError: If the underlying environment does not implement valid_moves().
        """
        if hasattr(self.env.unwrapped, "valid_moves"):
            return self.env.unwrapped.valid_moves()  # type: ignore
        raise NotImplementedError(
            "The underlying environment does not implement valid_moves()"
        )

    def _add_player_indicator(self, obs: np.ndarray) -> np.ndarray:
        """
        Add the player indicator to the observation.

        Args:
            obs (np.ndarray): The original observation.

        Returns:
            np.ndarray: The observation with the player indicator added.

        Raises:
            ValueError: If the observation is not 1D.
        """
        if len(obs.shape) != 1:
            raise ValueError("Observation must be 1D")
        value = (self.current_player_indicator + 1) // 2
        extended_obs = np.append(obs.flatten(), value)
        obs_shape = self.observation_space.shape
        assert isinstance(obs_shape, tuple), "Observation space does not exist"
        return extended_obs.reshape(obs_shape)
