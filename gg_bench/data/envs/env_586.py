import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 10 inclusive, mapped to indices 0-9
        self.action_space = spaces.Discrete(10)

        # Observation space: counts of numbers selected by both players
        # Indices 0-9: counts for player 1
        # Indices 10-19: counts for player 2
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(20,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize counts of selected numbers for both players
        self.counts = np.zeros((2, 10), dtype=np.int32)  # Players 1 and 2, numbers 1-10
        # Initialize products for both players
        self.player_products = [1, 1]  # Player 1 and Player 2
        # Player 1 starts
        self.current_player = 1
        # Game over flag
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no further actions should be taken
            return self._get_obs(), 0, True, False, {}

        if action < 0 or action >= 10:
            # Invalid action
            return self._get_obs(), -10, True, False, {}

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move (would cause product to exceed 100)
            return self._get_obs(), -10, True, False, {}

        selected_number = action + 1  # Map action index to number 1-10

        # Update counts
        player_idx = self.current_player - 1  # 0 for player 1, 1 for player 2
        self.counts[player_idx, action] += 1

        # Update product
        self.player_products[player_idx] *= selected_number

        # Check for win condition
        if self.player_products[player_idx] == 100:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}
        else:
            # Check if no valid moves left (deadlock)
            if not self.valid_moves():
                # Current player loses
                self.done = True
                reward = -1
                return self._get_obs(), reward, True, False, {}
            else:
                # Switch to other player
                self.current_player = 2 if self.current_player == 1 else 1
                reward = 0
                return self._get_obs(), reward, False, False, {}

    def render(self):
        output = "Player 1's set: {}\n".format(self._player_set(1))
        output += "Player 1's product: {}\n".format(self.player_products[0])

        output += "Player 2's set: {}\n".format(self._player_set(2))
        output += "Player 2's product: {}\n".format(self.player_products[1])

        output += "Current player: Player {}\n".format(self.current_player)

        return output

    def valid_moves(self):
        # Return list of valid action indices (0-9)
        current_product = self.player_products[self.current_player - 1]
        valid_moves = []
        for action in range(10):
            selected_number = action + 1
            new_product = current_product * selected_number
            if new_product <= 100:
                valid_moves.append(action)
        return valid_moves

    def _get_obs(self):
        # Concatenate counts of both players into single array
        observation = np.concatenate((self.counts[0], self.counts[1]))
        return observation

    def _player_set(self, player_num):
        # Return the list of numbers selected by the player
        player_idx = player_num - 1
        set_list = []
        for action in range(10):
            count = self.counts[player_idx, action]
            number = action + 1
            set_list.extend([number] * count)
        return set_list
