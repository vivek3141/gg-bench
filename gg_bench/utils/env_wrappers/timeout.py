from typing import Any, Dict, Optional, Tuple

from gymnasium import Env, Wrapper


class EnvTimeoutError(Exception):
    """Custom exception raised when the environment reaches the maximum number of steps."""

    pass


class TimeoutEnv(Wrapper):
    """
    A wrapper for environments that imposes a maximum number of steps.

    This wrapper raises an EnvTimeoutError if the number of steps exceeds
    the specified timeout value.
    """

    def __init__(self, env: Env, timeout: int = 100):
        """
        Initialize the TimeoutEnv wrapper.

        Args:
            env (Env): The environment to wrap.
            timeout (int): The maximum number of steps allowed before raising a timeout error.
        """
        super().__init__(env)
        self.timeout: int = timeout
        self.steps: int = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and the step counter.

        Args:
            seed (Optional[int]): The seed for the random number generator.
            options (Optional[Dict[str, Any]]): Additional options for resetting the environment.

        Returns:
            Tuple[Any, Dict[str, Any]]: The initial observation and info dictionary.
        """
        self.steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment and check if the timeout has been reached.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: The observation, reward, terminated flag,
                                                           truncated flag, and info dictionary.

        Raises:
            EnvTimeoutError: If the number of steps exceeds the timeout value.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if self.steps >= self.timeout:
            raise EnvTimeoutError("Max steps reached")
        return obs, reward, terminated, truncated, info
