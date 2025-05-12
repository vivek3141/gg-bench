import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 -> move 1 step, 1 -> move 2 steps, 2 -> move 3 steps
        self.action_space = spaces.Discrete(3)

        # Define observation space: positions of both players and current player
        self.observation_space = spaces.Box(low=0, high=7, shape=(3,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize player positions: both start at cell 0
        self.positions = np.array([0, 0], dtype=np.int32)
        # Current player: 0 for Player 1, 1 for Player 2
        self.current_player = 0
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no further moves are allowed
            return self._get_obs(), 0, True, False, {}

        # Validate the action
        if action not in [0, 1, 2]:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Map action to move (1, 2, or 3 steps)
        move = action + 1

        # Get the current player's position
        position = self.positions[self.current_player]

        # Calculate new position, ensuring it does not exceed cell 7
        new_position = position + move
        if new_position > 7:
            new_position = 7

        # Update the player's position
        self.positions[self.current_player] = new_position

        # Check for pushing the opponent
        opponent = 1 - self.current_player
        if new_position == self.positions[opponent]:
            # Push the opponent back to cell 0
            self.positions[opponent] = 0

        # Check for win condition
        if new_position == 7:
            self.done = True
            reward = 1  # Current player wins
            return self._get_obs(), reward, True, False, {}
        else:
            # No reward for a regular move
            reward = 0

        # Swap to the next player
        self.current_player = opponent

        return self._get_obs(), reward, False, False, {}

    def render(self):
        # Create a list representing the grid cells
        grid = ["-" for _ in range(8)]
        # Mark the positions of the players on the grid
        symbols = ["X", "O"]
        for idx, pos in enumerate(self.positions):
            grid[pos] = symbols[idx]

        # Build the visual representation of the grid
        grid_str = "[Start] "
        for i, cell in enumerate(grid):
            grid_str += f"{cell} " if i < 7 else f"{cell} [Finish]"
        return grid_str

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1, 2]

    def _get_obs(self):
        # Return the observation including player positions and current player
        return np.array(
            [self.positions[0], self.positions[1], self.current_player], dtype=np.int32
        )
