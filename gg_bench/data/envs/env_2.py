import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Numbers 1-9 mapped to actions 0-8

        # Observation is the state of the numbers: 1 (current player), -1 (opponent), 0 (available)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.available_numbers = set(range(1, 10))  # Numbers 1 to 9
        self.player_collections = {1: [], -1: []}
        self.current_player = 1  # Start with player 1
        self.done = False

        self.board = np.zeros(9, dtype=np.int8)  # Represents the numbers 1-9

        return self._get_obs(), {}  # Observation and info dict

    def step(self, action):
        selected_number = action + 1  # Map action 0-8 to number 1-9

        # Check if move is valid
        if selected_number not in self.available_numbers or self.done:
            reward = -10
            self.done = True
            return self._get_obs(), reward, self.done, False, {}

        # Valid move
        self.available_numbers.remove(selected_number)
        self.player_collections[self.current_player].append(selected_number)
        self.board[action] = self.current_player

        # Check for a win
        if self._check_win(self.player_collections[self.current_player]):
            reward = 1
            self.done = True
            return self._get_obs(), reward, self.done, False, {}

        # Check for sudden death
        if not self.available_numbers:
            # Sudden death
            player_sum = sum(self.player_collections[self.current_player])
            opponent_sum = sum(self.player_collections[-self.current_player])

            if player_sum > opponent_sum:
                reward = 1
            elif player_sum < opponent_sum:
                reward = -1
            else:
                # Sums equal, compare highest individual numbers
                player_max = max(self.player_collections[self.current_player])
                opponent_max = max(self.player_collections[-self.current_player])

                if player_max > opponent_max:
                    reward = 1
                else:
                    reward = -1  # Opponent wins

            self.done = True
            return self._get_obs(), reward, self.done, False, {}

        # Switch player
        self.current_player *= -1

        reward = -10  # Penalty for a valid move (as per instruction)

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        output = (
            "Available Numbers: "
            + ", ".join(map(str, sorted(self.available_numbers)))
            + "\n"
        )
        output += f"Player {self.current_player}'s collection: {sorted(self.player_collections[self.current_player])}\n"
        output += f"Opponent's collection: {sorted(self.player_collections[-self.current_player])}\n"
        return output

    def valid_moves(self):
        return [
            num - 1 for num in self.available_numbers
        ]  # actions are indices from 0 to 8

    def _get_obs(self):
        # From the current player's perspective
        obs = self.board * self.current_player
        return obs

    def _check_win(self, collection):
        if len(collection) >= 3:
            from itertools import combinations

            for combo in combinations(collection, 3):
                combo = sorted(combo)
                if combo[2] - combo[1] == combo[1] - combo[0]:
                    return True
        return False
