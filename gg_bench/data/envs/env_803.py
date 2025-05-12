import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 possible blocks to choose from, numbered 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation space consists of two parts:
        # - Available blocks: positions 0-8 (1 if available, 0 if used)
        # - Tower state: positions 9-17 (numbers of blocks placed, 0 if empty)
        self.observation_space = spaces.Box(low=0, high=9, shape=(18,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_blocks = np.ones(9, dtype=np.int8)  # Blocks 1-9 are available
        self.tower = np.zeros(9, dtype=np.int8)  # Tower can have up to 9 blocks
        self.current_player = 1  # Player 1 starts (can be 1 or -1)
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action to block number (1-9)
        block_number = action + 1

        # Check if block is available
        if self.available_blocks[action] != 1:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid move

        # Check placement rules
        tower_height = np.count_nonzero(self.tower)
        if tower_height == 0:
            # First move, any block can be placed
            valid_move = True
        else:
            block_below = self.tower[tower_height - 1]
            # Check if block_number is a divisor or multiple of block_below
            if block_below % block_number == 0 or block_number % block_below == 0:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid move

        # Valid move, place the block
        self.tower[tower_height] = block_number
        self.available_blocks[action] = 0  # Mark block as used

        # Check if opponent has any valid moves
        opponent_moves = self.valid_moves()
        if not opponent_moves:
            # Opponent cannot move, current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        return self._get_observation(), 0, False, False, {}

    def render(self):
        tower_blocks = self.tower[self.tower != 0]
        tower_str = "Current Tower: [{}]".format(
            ", ".join(map(str, tower_blocks)) if tower_blocks.size > 0 else "Empty"
        )
        available_blocks = np.where(self.available_blocks == 1)[0] + 1
        available_str = "Available Blocks: [{}]".format(
            ", ".join(map(str, available_blocks))
        )
        player_str = "Current Player: Player {}".format(
            1 if self.current_player == 1 else 2
        )
        return "{}\n{}\n{}".format(tower_str, available_str, player_str)

    def valid_moves(self):
        valid_moves = []
        tower_height = np.count_nonzero(self.tower)
        if tower_height == 0:
            # First move, any available block can be placed
            valid_moves = list(np.where(self.available_blocks == 1)[0])
        else:
            block_below = self.tower[tower_height - 1]
            for i in range(9):
                if self.available_blocks[i] == 1:
                    block_number = i + 1
                    if (
                        block_below % block_number == 0
                        or block_number % block_below == 0
                    ):
                        valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        # Concatenate available_blocks and tower to form the observation
        observation = np.concatenate((self.available_blocks, self.tower))
        return observation
