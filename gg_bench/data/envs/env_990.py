import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Actions are cell indices 0-3
        # Observation is the current values of the 4 cells
        # Each cell has a capacity equal to its cell number (1-based)
        self.capacities = np.array([1, 2, 3, 4], dtype=np.int8)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=self.capacities, shape=(4,), dtype=np.int8
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cells = np.zeros(4, dtype=np.int8)  # Cells start at 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.cells.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.cells.copy(), 0, True, False, {}  # Game is over

        valid_actions = self.valid_moves()

        if action not in valid_actions and len(valid_actions) > 0:
            # Invalid move when valid moves are available
            self.done = True
            reward = -10
            return (
                self.cells.copy(),
                reward,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Add 1 token to the selected cell
        self.cells[action] += 1

        # Check for overload
        if self.cells[action] > self.capacities[action]:
            # Current player loses
            self.done = True
            reward = -10
            return self.cells.copy(), reward, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if opponent has any valid moves
        opponent_valid_actions = self.valid_moves()
        if len(opponent_valid_actions) == 0:
            # Opponent has no valid moves, current player wins
            self.done = True
            reward = 1
            return self.cells.copy(), reward, True, False, {}

        # Game continues
        reward = 0
        return self.cells.copy(), reward, False, False, {}

    def render(self):
        # Generate a string representing the current state
        state_lines = []
        for i in range(4):
            cell_num = i + 1
            state_lines.append(
                f"Cell {cell_num} [{self.cells[i]}/{self.capacities[i]}]"
            )
        return " | ".join(state_lines)

    def valid_moves(self):
        # Return a list of valid actions (cells where value < capacity)
        return [i for i in range(4) if self.cells[i] < self.capacities[i]]
