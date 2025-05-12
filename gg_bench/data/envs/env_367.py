import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Blocks numbered 1-9

        # Observation space consists of:
        # - Available blocks: array of size 9, values {-1, 0, 1}
        # - Current player's tower height: scalar
        # - Current player's top block: scalar
        # - Opponent's tower height: scalar
        # - Opponent's top block: scalar
        self.observation_space = spaces.Box(low=-1, high=15, shape=(13,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize available blocks: 0 for available, 1/-1 for used by player
        self.available_blocks = np.zeros(9, dtype=np.int8)

        # Initialize player towers
        self.player_towers = {
            1: {"height": 0, "top_block": 0},
            -1: {"height": 0, "top_block": 0},
        }

        self.current_player = 1  # Player 1 starts
        self.done = False

        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        block_number = action + 1  # Block numbers are 1-9
        current_tower = self.player_towers[self.current_player]
        opponent_tower = self.player_towers[-self.current_player]

        # Place the block
        self.available_blocks[action] = self.current_player
        current_tower["height"] += block_number
        current_tower["top_block"] = block_number

        # Check for win condition
        if current_tower["height"] == 15:
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Check for loss condition
        if current_tower["height"] > 15:
            self.done = True
            return self._get_obs(), -1, True, False, {}

        # Check if no valid moves remain for both players
        if not self.valid_moves() and not self._opponent_valid_moves():
            self.done = True
            # Determine winner based on tower heights
            if current_tower["height"] > opponent_tower["height"]:
                return self._get_obs(), 1, True, False, {}
            elif current_tower["height"] < opponent_tower["height"]:
                return self._get_obs(), -1, True, False, {}
            else:
                return self._get_obs(), 0, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        return self._get_obs(), 0, False, False, {}

    def render(self):
        player = "Player 1" if self.current_player == 1 else "Player 2"
        output = f"Current Player: {player}\n"
        output += "Available Blocks:\n"
        for i in range(9):
            if self.available_blocks[i] == 0:
                output += f"{i+1} "
        output += "\n"
        output += f"Player 1's Tower Height: {self.player_towers[1]['height']} units\n"
        output += f"Player 1's Tower Blocks: {self._get_tower_blocks(1)}\n"
        output += f"Player 2's Tower Height: {self.player_towers[-1]['height']} units\n"
        output += f"Player 2's Tower Blocks: {self._get_tower_blocks(-1)}\n"
        return output

    def valid_moves(self):
        valid_actions = []
        current_tower = self.player_towers[self.current_player]
        for action in range(9):
            block_number = action + 1
            # Check if block is available
            if self.available_blocks[action] != 0:
                continue
            # Check stacking rules
            if (
                current_tower["top_block"] != 0
                and block_number > current_tower["top_block"]
            ):
                continue
            # Check tower height limit
            if current_tower["height"] + block_number > 15:
                continue
            valid_actions.append(action)
        return valid_actions

    def _opponent_valid_moves(self):
        valid_actions = []
        opponent_tower = self.player_towers[-self.current_player]
        for action in range(9):
            block_number = action + 1
            # Check if block is available
            if self.available_blocks[action] != 0:
                continue
            # Check stacking rules
            if (
                opponent_tower["top_block"] != 0
                and block_number > opponent_tower["top_block"]
            ):
                continue
            # Check tower height limit
            if opponent_tower["height"] + block_number > 15:
                continue
            valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        obs = np.zeros(13, dtype=np.int8)
        obs[0:9] = self.available_blocks
        current_tower = self.player_towers[self.current_player]
        opponent_tower = self.player_towers[-self.current_player]
        obs[9] = current_tower["height"]
        obs[10] = current_tower["top_block"]
        obs[11] = opponent_tower["height"]
        obs[12] = opponent_tower["top_block"]
        return obs

    def _get_tower_blocks(self, player):
        blocks = []
        used_blocks = np.where(self.available_blocks == player)[0]
        sorted_blocks = sorted([block + 1 for block in used_blocks], reverse=True)
        return sorted_blocks
