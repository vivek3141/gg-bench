import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 => Remove 1 block, 1 => Remove 2 blocks
        self.action_space = spaces.Discrete(2)

        # Observation: 10 elements, 1 if block is present, 0 if removed
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32)

        # Initialize the tower and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tower = np.ones(10, dtype=np.int32)  # All blocks present at start
        self.current_player = 1  # Player 1 starts
        self.previous_move = None  # No previous move
        self.done = False
        return self.tower.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.tower.copy(), 0, True, False, {}
        number_of_blocks_to_remove = action + 1  # Map action to blocks to remove

        # Determine allowed moves based on previous move
        if self.previous_move is None:
            allowed_moves = [1, 2]
        elif self.previous_move == 1:
            allowed_moves = [1, 2]
        elif self.previous_move == 2:
            allowed_moves = [1]
        else:
            allowed_moves = [1, 2]  # Should not happen, default to both moves allowed

        # Cannot remove more blocks than are remaining
        blocks_remaining = np.sum(self.tower)
        if number_of_blocks_to_remove > blocks_remaining:
            allowed_moves = [move for move in allowed_moves if move <= blocks_remaining]

        # Check if the action is valid
        if number_of_blocks_to_remove not in allowed_moves:
            # Invalid move
            self.done = True
            reward = -10
            return (
                self.tower.copy(),
                reward,
                True,
                False,
                {},
            )  # Observation, reward, done, truncated, info

        # Valid move, remove blocks
        indices_to_remove = np.where(self.tower == 1)[0][:number_of_blocks_to_remove]
        self.tower[indices_to_remove] = 0

        # Update previous move
        self.previous_move = number_of_blocks_to_remove

        # Check if the game is over (no blocks remaining)
        if np.sum(self.tower) == 0:
            self.done = True
            reward = 1  # Current player wins
            return self.tower.copy(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            self.current_player = -self.current_player  # Switch player
            return self.tower.copy(), reward, False, False, {}

    def render(self):
        tower_levels = np.where(self.tower == 1)[0] + 1  # Levels are from 1 to 10
        representation = (
            f"Current Player: {'Player 1' if self.current_player ==1 else 'Player 2'}\n"
        )
        representation += f"Tower Levels: {tower_levels.tolist()}\n"
        return representation

    def valid_moves(self):
        number_of_blocks_remaining = np.sum(self.tower)
        if self.previous_move is None:
            allowed_moves = [1, 2]
        elif self.previous_move == 1:
            allowed_moves = [1, 2]
        elif self.previous_move == 2:
            allowed_moves = [1]
        else:
            allowed_moves = [1, 2]
        # Cannot remove more blocks than are remaining
        allowed_moves = [
            move for move in allowed_moves if move <= number_of_blocks_remaining
        ]
        # Map allowed moves to action indices (action = move -1)
        valid_actions = [move - 1 for move in allowed_moves]
        return valid_actions
