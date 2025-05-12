import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-9 for blocks 1-10, 10 for 'pass'
        self.action_space = spaces.Discrete(11)  # actions 0 to 10

        # Define observation space
        # Block counts: 0 to 2 (length 10)
        # Tower heights: 0 to 15 (length 2)
        # Current player: 1 or 2 (length 1)
        # Total observation length: 13
        self.low = np.array([0] * 10 + [0, 0, 1], dtype=np.int32)
        self.high = np.array([2] * 10 + [15, 15, 2], dtype=np.int32)

        self.observation_space = spaces.Box(
            low=self.low, high=self.high, dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.block_pool = np.array([2] * 10, dtype=np.int32)  # Two copies of each block
        self.tower_heights = np.array(
            [0, 0], dtype=np.int32
        )  # Player 1 and Player 2 tower heights
        self.current_player = 0  # 0 for player 1, 1 for player 2
        self.done = False
        observation = self._get_obs()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_moves = self.valid_moves()

        if action not in valid_moves:
            # Invalid action
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        reward = 0

        if action == 10:
            # Pass
            self.current_player = 1 - self.current_player
            # Check if next player can move
            if 10 in self.valid_moves():
                # Neither player can move
                self.done = True
                # Determine winner
                if self.tower_heights[0] > self.tower_heights[1]:
                    winner = 0
                elif self.tower_heights[1] > self.tower_heights[0]:
                    winner = 1
                else:
                    winner = -1  # Tie
                if winner == self.current_player:
                    reward = 1
                elif winner == -1:
                    reward = 0
                else:
                    reward = -1
                return self._get_obs(), reward, True, False, {}
            else:
                return self._get_obs(), reward, False, False, {}
        else:
            # Action is block selection
            block_number = action + 1  # action 0 corresponds to block 1
            self.block_pool[action] -= 1
            new_height = self.tower_heights[self.current_player] + block_number
            self.tower_heights[self.current_player] = new_height

            # Check if adding block causes tower to exceed 15
            if new_height > 15:
                # Player loses immediately
                self.done = True
                reward = -10
                return self._get_obs(), reward, True, False, {}

            # Check for victory
            if new_height == 15:
                self.done = True
                reward = 1
                return self._get_obs(), reward, True, False, {}

            # Switch to next player
            self.current_player = 1 - self.current_player
            return self._get_obs(), reward, False, False, {}

    def render(self):
        block_pool_str = "Block Pool: "
        for i in range(10):
            block_number = i + 1
            count = self.block_pool[i]
            if count > 0:
                block_pool_str += f"{block_number}({count}) "
        player1_height = self.tower_heights[0]
        player2_height = self.tower_heights[1]
        current_player_str = f"Current Player: Player {self.current_player + 1}"
        return f"{block_pool_str}\nPlayer 1 Tower Height: {player1_height}\nPlayer 2 Tower Height: {player2_height}\n{current_player_str}"

    def valid_moves(self):
        moves = []
        # Calculate valid block selections
        for i in range(10):  # block numbers 1 to 10, actions 0 to 9
            if self.block_pool[i] > 0:
                block_number = i + 1
                if self.tower_heights[self.current_player] + block_number <= 15:
                    moves.append(i)
        if len(moves) == 0:
            moves.append(10)  # Add 'pass' action if no other moves
        return moves

    def _get_obs(self):
        observation = np.concatenate(
            (self.block_pool, self.tower_heights, np.array([self.current_player + 1]))
        )
        return observation
