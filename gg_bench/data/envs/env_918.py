import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_product=100):
        super(CustomEnv, self).__init__()

        self.target_product = target_product

        # Define action space: numbers from 1 to 9 represented as indices 0 to 8
        self.action_space = spaces.Discrete(9)

        # Define observation space
        # Observation is a vector of length 11:
        # - First 9 entries: availability of numbers 1-9 (1 if available, 0 if not)
        # - Entry 10: current player's product
        # - Entry 11: opponent's product
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [1, 1]),
            high=np.array([1] * 9 + [362880, 362880]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.availability = np.ones(9, dtype=np.int32)  # Numbers 1 to 9 available
        self.player_products = [1, 1]  # Player 1 and Player 2 products
        self.current_player = 0  # Player index: 0 for Player 1, 1 for Player 2
        self.done = False

        # Construct the observation
        observation = np.concatenate(
            (
                self.availability,
                [
                    self.player_products[self.current_player],
                    self.player_products[1 - self.current_player],
                ],
            )
        )

        return observation, {}

    def step(self, action):
        if self.done:
            observation = np.concatenate(
                (
                    self.availability,
                    [
                        self.player_products[self.current_player],
                        self.player_products[1 - self.current_player],
                    ],
                )
            )
            return observation, 0, True, False, {}

        # Invalid action
        if action < 0 or action >= 9 or self.availability[action] == 0:
            self.done = True
            observation = np.concatenate(
                (
                    self.availability,
                    [
                        self.player_products[self.current_player],
                        self.player_products[1 - self.current_player],
                    ],
                )
            )
            return observation, -10, True, False, {}

        # Valid action
        number_chosen = action + 1
        self.availability[action] = 0
        self.player_products[self.current_player] *= number_chosen

        # Check if current player reaches target
        if self.player_products[self.current_player] >= self.target_product:
            self.done = True
            observation = np.concatenate(
                (
                    self.availability,
                    [
                        self.player_products[self.current_player],
                        self.player_products[1 - self.current_player],
                    ],
                )
            )
            return observation, 1, True, False, {}

        # Check if all numbers have been used
        if np.sum(self.availability) == 0:
            self.done = True
            if (
                self.player_products[self.current_player]
                > self.player_products[1 - self.current_player]
            ):
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses
            observation = np.concatenate(
                (
                    self.availability,
                    [
                        self.player_products[self.current_player],
                        self.player_products[1 - self.current_player],
                    ],
                )
            )
            return observation, reward, True, False, {}

        # Switch to next player
        self.current_player = 1 - self.current_player
        observation = np.concatenate(
            (
                self.availability,
                [
                    self.player_products[self.current_player],
                    self.player_products[1 - self.current_player],
                ],
            )
        )
        return observation, 0, False, False, {}

    def render(self):
        available_numbers = [i + 1 for i in range(9) if self.availability[i] == 1]
        state_str = f"Available Numbers: {available_numbers}\n"
        state_str += f"Player {self.current_player + 1}'s Product: {self.player_products[self.current_player]}\n"
        state_str += f"Player {((self.current_player + 1) % 2) + 1}'s Product: {self.player_products[1 - self.current_player]}\n"
        return state_str

    def valid_moves(self):
        return [i for i in range(9) if self.availability[i] == 1]
