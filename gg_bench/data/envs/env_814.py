import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0: Add 1, 1: Add 2, 2: Add 3, 3: Multiply by 2
        self.action_space = spaces.Discrete(4)

        # Define observation space: [current_number, last_operation]
        # - current_number ranges from 1 to 100 (sufficient upper limit for the game)
        # - last_operation: 0 (None), 1 (Addition), 2 (Multiplication)
        self.observation_space = spaces.Box(
            low=np.array([1, 0], dtype=np.float32),
            high=np.array([100, 2], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1  # Starting number
        self.current_player = 1  # Player 1 starts
        self.last_operation = 0  # No last operation (None)
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def _get_obs(self):
        return np.array([self.current_number, self.last_operation], dtype=np.float32)

    def step(self, action):
        if self.done:
            # The game has already ended
            return self._get_obs(), -10, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Perform the action
        reward = 0
        operation = None
        if action == 0:
            operation = "add"
            self.current_number += 1
        elif action == 1:
            operation = "add"
            self.current_number += 2
        elif action == 2:
            operation = "add"
            self.current_number += 3
        elif action == 3:
            operation = "multiply"
            self.current_number *= 2

        # Update last_operation
        if operation == "add":
            self.last_operation = 1
        elif operation == "multiply":
            self.last_operation = 2

        # Check for victory
        if self.current_number >= 31:
            self.done = True
            reward = 1  # Current player wins
            return self._get_obs(), reward, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_obs(), reward, False, False, {}

    def render(self):
        info = f"Current Number: {self.current_number}\n"
        info += f"Player {self.current_player}'s Turn\n"
        last_op = (
            "None"
            if self.last_operation == 0
            else "Addition" if self.last_operation == 1 else "Multiplication"
        )
        info += f"Last Operation: {last_op}\n"
        action_map = {0: "Add 1", 1: "Add 2", 2: "Add 3", 3: "Multiply by 2"}
        actions = self.valid_moves()
        actions_desc = [action_map[a] for a in actions]
        info += f"Available Actions: {actions_desc}\n"
        return info

    def valid_moves(self):
        if self.done:
            return []  # No valid moves if the game is over

        # Determine valid actions based on last operation
        if self.last_operation == 0:
            # No restriction on the first move
            return [0, 1, 2, 3]
        elif self.last_operation == 1:
            # Opponent added last turn, cannot add this turn
            return [3]
        elif self.last_operation == 2:
            # Opponent multiplied last turn, cannot multiply this turn
            return [0, 1, 2]
        else:
            # Should not occur, but return all actions just in case
            return [0, 1, 2, 3]
