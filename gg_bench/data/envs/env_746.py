import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_height=20):
        super(CustomEnv, self).__init__()

        self.target_height = target_height

        # Define action and observation space
        # There are 9 possible Number Blocks to choose from, numbered 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation space consists of:
        # - Tower Height (scalar)
        # - Player A's Number Blocks remaining (9 binary values)
        # - Player B's Number Blocks remaining (9 binary values)
        # - Current Player indicator (0 or 1)
        low = np.array([0] + [0] * 9 + [0] * 9 + [0], dtype=np.int32)
        high = np.array([self.target_height] + [1] * 9 + [1] * 9 + [1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tower_height = 0
        self.current_player = 0  # 0 for Player A, 1 for Player B
        self.player_blocks = [
            np.ones(9, dtype=np.int32),  # Player A's blocks
            np.ones(9, dtype=np.int32),  # Player B's blocks
        ]
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_obs(), 0, True, False, {}

        # Check if action is valid
        current_blocks = self.player_blocks[self.current_player]
        block_value = action + 1  # Number Blocks are from 1 to 9

        if action < 0 or action >= 9:
            # Invalid action index
            self.done = True
            return self._get_obs(), -10, True, False, {}

        if current_blocks[action] == 0:
            # Block has already been used
            self.done = True
            return self._get_obs(), -10, True, False, {}

        if self.tower_height + block_value > self.target_height:
            # Move exceeds target height
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move
        self.tower_height += block_value
        current_blocks[action] = 0  # Remove the block from player's set

        if self.tower_height == self.target_height:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Check if next player has valid moves
        next_player = (self.current_player + 1) % 2
        next_player_blocks = self.player_blocks[next_player]
        valid = False
        for idx, available in enumerate(next_player_blocks):
            if available == 1:
                next_block_value = idx + 1
                if self.tower_height + next_block_value <= self.target_height:
                    valid = True
                    break
        if not valid:
            # Next player cannot make a move, current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch to next player
        self.current_player = next_player

        return self._get_obs(), 0, False, False, {}

    def render(self):
        render_str = "Current Tower Height: {}\n".format(self.tower_height)
        render_str += "Current Player: {}\n".format(
            "A" if self.current_player == 0 else "B"
        )
        render_str += "Player A's Remaining Blocks: {}\n".format(
            [i + 1 for i, x in enumerate(self.player_blocks[0]) if x == 1]
        )
        render_str += "Player B's Remaining Blocks: {}\n".format(
            [i + 1 for i, x in enumerate(self.player_blocks[1]) if x == 1]
        )
        print(render_str)

    def valid_moves(self):
        current_blocks = self.player_blocks[self.current_player]
        valid_moves = []
        for idx, available in enumerate(current_blocks):
            if available == 1:
                block_value = idx + 1
                if self.tower_height + block_value <= self.target_height:
                    valid_moves.append(idx)
        return valid_moves

    def _get_obs(self):
        obs = np.concatenate(
            (
                np.array([self.tower_height], dtype=np.int32),
                self.player_blocks[0],
                self.player_blocks[1],
                np.array([self.current_player], dtype=np.int32),
            )
        )
        return obs
