import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 10 discrete actions (toggle lights 1-10)
        self.action_space = spaces.Discrete(10)
        # Observation space: 10 lights with states 0 (OFF) or 1 (ON)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lights = np.zeros(10, dtype=np.int8)  # All lights start OFF
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game is not over
        return self.lights.copy(), {}  # Return initial observation and empty info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.lights.copy(), 0, True, False, {}

        # Check if the action is valid
        if action not in self.valid_moves():
            self.done = True
            return self.lights.copy(), -10, True, False, {}  # Invalid move

        # Toggle the selected light
        self.lights[action] = 1 - self.lights[action]  # Toggle between 0 and 1

        # Check for a win condition: three consecutive ON lights
        for i in range(8):  # Positions 0 to 7
            if (
                self.lights[i] == 1
                and self.lights[i + 1] == 1
                and self.lights[i + 2] == 1
            ):
                self.done = True
                return self.lights.copy(), 1, True, False, {}  # Current player wins

        # No win, switch to the other player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return self.lights.copy(), 0, False, False, {}  # Continue game

    def render(self):
        # Generate a string representation of the current game state
        board_str = ""
        for i in range(10):
            state = "ON " if self.lights[i] == 1 else "OFF"
            board_str += f"[{i + 1}] {state}  "
        return board_str.strip()

    def valid_moves(self):
        # All lights (0-9) are valid moves if the game is not over
        if self.done:
            return []
        return list(range(10))
