import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.start_number = 1
        self.target_number = 20
        self.allowed_operations = [("+1", lambda x: x + 1), ("×2", lambda x: x * 2)]

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.allowed_operations))
        self.observation_space = spaces.Box(
            low=0, high=self.target_number, shape=(1,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.start_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        # Apply the selected operation
        _, operation = self.allowed_operations[action]
        try:
            result = operation(self.current_number)
            if not isinstance(result, int):
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                else:
                    self.done = True
                    return self._get_obs(), -10, True, False, {}  # Invalid result
            if result <= 0 or result > self.target_number:
                self.done = True
                return self._get_obs(), -10, True, False, {}  # Exceeds target
        except Exception:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Exception occurred

        self.current_number = result

        # Check for win condition
        if self.current_number == self.target_number:
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Switch players
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return self._get_obs(), 0, False, False, {}  # Continue game

    def render(self):
        return (
            f"Current Number: {self.current_number}\n"
            f"Target Number: {self.target_number}\n"
            f"Player {self.current_player}'s turn\n"
            f"Allowed Operations: {', '.join(op[0] for op in self.allowed_operations)}"
        )

    def valid_moves(self):
        valid_actions = []
        for index, (name, operation) in enumerate(self.allowed_operations):
            try:
                result = operation(self.current_number)
                if not isinstance(result, int):
                    if isinstance(result, float) and result.is_integer():
                        result = int(result)
                    else:
                        continue  # Result is not an integer
                if result <= 0 or result > self.target_number:
                    continue  # Invalid range
                valid_actions.append(index)
            except Exception:
                continue  # Operation caused an exception
        return valid_actions

    def _get_obs(self):
        return np.array([self.current_number], dtype=np.float32)
