from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, Wrapper

from gg_bench.utils.inference import get_prediction


class AlternatingAgentEnv(Wrapper):
    """
    A wrapper for having an agent play against an opposing agent.

    This wrapper alternates turns between the player and an opposing agent,
    modifying the environment to create a two-player game scenario.
    """

    def __init__(self, env: Env, opposing_agent_first: Optional[bool] = None):
        """
        Initialize the AlternatingAgentEnv.

        Args:
            env (Env): The environment to wrap.
            opposing_agent_first (Optional[bool]): Whether the opposing agent goes first.
                If None, it will be randomly decided for each episode.
        """
        super().__init__(env)
        self.opposing_agents: Any = []
        self.random_agent_first: bool = opposing_agent_first is None
        self.opposing_agent_first: bool = (
            np.random.choice([True, False])
            if not opposing_agent_first
            else opposing_agent_first
        )
        self.epsilon: float = 1.0
        self.last_obs: Any = None

    def add_opposing_agent(self, agent: Any, resample: bool = True) -> None:
        """
        Set the opposing agent.

        Args:
            agent (Any): The opposing agent to be used.
        """
        self.opposing_agents.append(agent)
        if resample:
            self.opposing_agent = np.random.choice(self.opposing_agents)

    def set_epsilon(self, epsilon: float) -> None:
        """
        Set the epsilon value for exploration.

        Args:
            epsilon (float): The epsilon value to set.
        """
        self.epsilon = epsilon

    def _step_opposing_agent(
        self, obs: Any
    ) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """
        Take a step with the opposing agent.

        Args:
            obs (Any): The current observation.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: The step results (obs, reward, done, truncated, info).

        Raises:
            AssertionError: If the opposing agent is not set.
        """
        assert (
            self.opposing_agent is not None
        ), "Opposing agent must be set before calling this method"

        # Note: Assuming the environment has a 'valid_moves' method. If not, you'll need to modify this part.
        action = get_prediction(
            agent=self.opposing_agent,
            obs=obs,
            valid_moves=self.env.unwrapped.valid_moves(),  # type: ignore
            epsilon=self.epsilon,
        )
        return self.env.step(action)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and determine which agent goes first.

        Returns:
            Tuple[Any, Dict[str, Any]]: The initial observation and info dictionary.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self.last_obs = obs
        self.opposing_agent = np.random.choice(self.opposing_agents)

        if self.random_agent_first:
            self.opposing_agent_first = np.random.choice([True, False])

        if self.opposing_agent_first:
            obs, _, done, _, info = self._step_opposing_agent(obs)
            assert not done
            return obs, info
        else:
            return obs, info

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment, alternating between the main agent and the opposing agent.

        Args:
            action (Any): The action to take.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: The step results (obs, reward, done, truncated, info).
        """
        obs, reward, done, truncated, info = self.env.step(action)
        if done:
            return obs, reward, done, truncated, info

        obs, reward, done, truncated, info = self._step_opposing_agent(obs)
        reward = -reward if reward in [-1, 1] else reward

        return obs, reward, done, truncated, info
