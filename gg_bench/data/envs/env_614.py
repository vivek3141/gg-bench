import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # There are 9 blocks numbered from 1 to 9 (actions 0 to 8)
        self.action_space = spaces.Discrete(9)
        # Observation is an array of size 18:
        # - First 9 elements represent the tower (blocks placed so far)
        # - Next 9 elements represent available blocks (1 if available, 0 if not)
        self.observation_space = spaces.Box(low=0, high=9, shape=(18,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize game state
        self.tower = np.zeros(9, dtype=np.int32)  # From bottom (index 0) to top
        self.available_blocks = np.ones(
            9, dtype=np.int32
        )  # 1 if block (1-9) is available
        self.current_player = 1  # Player 1 starts
        self.turn_count = 0  # Number of blocks placed
        self.done = False
        # Return initial observation and info dict
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_observation(), 0, True, False, {}

        # Check if current player has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player cannot make any valid moves, they lose
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if action is valid
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Process the action
        block_number = action + 1
        # Place the block on the tower
        self.tower[self.turn_count] = block_number
        self.available_blocks[action] = 0  # Mark the block as used
        self.turn_count += 1

        # After the move, check if opponent has any valid moves
        opponent_valid_moves = self._get_opponent_valid_moves(block_number)
        if not opponent_valid_moves:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}
        else:
            # Switch player
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        tower_blocks = self.tower[: self.turn_count]
        tower_str_list = [
            "| " + str(block) + " |" for block in tower_blocks[::-1]
        ]  # From top to bottom
        tower_str = "\n".join(tower_str_list)
        available_blocks_str = ", ".join(
            [str(i + 1) for i in range(9) if self.available_blocks[i]]
        )
        render_str = f"Current Player: Player {self.current_player}\n"
        render_str += f"Available Blocks: {available_blocks_str}\n"
        render_str += "Tower (top to bottom):\n" + tower_str
        return render_str

    def valid_moves(self):
        # Return a list of valid actions for the current player
        valid_actions = []
        for action in range(9):
            if self.available_blocks[action]:
                block_number = action + 1
                if self.turn_count == 0:
                    # On first move, any block can be placed
                    valid_actions.append(action)
                else:
                    last_block = self.tower[self.turn_count - 1]
                    if last_block % block_number == 0 or block_number % last_block == 0:
                        valid_actions.append(action)
        return valid_actions

    def _get_opponent_valid_moves(self, last_block_number):
        # Determine if opponent has any valid moves
        opponent_valid_actions = []
        for action in range(9):
            if self.available_blocks[action]:
                block_number = action + 1
                if (
                    last_block_number % block_number == 0
                    or block_number % last_block_number == 0
                ):
                    opponent_valid_actions.append(action)
        return opponent_valid_actions

    def _get_observation(self):
        # Return the observation combining the tower and available blocks
        observation = np.concatenate((self.tower, self.available_blocks))
        return observation
