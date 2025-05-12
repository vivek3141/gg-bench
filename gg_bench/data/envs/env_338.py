import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space: integers 2 to 9 inclusive, mapped to action indices 0-7
        self.action_space = spaces.Discrete(8)

        # Observation space: cumulative products of both players
        self.observation_space = spaces.Box(
            low=1, high=1000, shape=(2,), dtype=np.int32
        )

        # Initialize the environment
        self.current_player = 1  # Player 1 starts
        self.cumulative_products = [1, 1]
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.cumulative_products = [1, 1]
        self.done = False
        observation = np.array(self.cumulative_products, dtype=np.int32)
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                np.array(self.cumulative_products, dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Map action index to number choice
        number_choice = action + 2  # actions are 0 to 7 corresponding to numbers 2 to 9

        current_index = self.current_player - 1
        current_product = self.cumulative_products[current_index]
        new_product = current_product * number_choice

        # Update cumulative products
        self.cumulative_products[current_index] = new_product

        # Determine the reward and done flag
        if new_product == 1000:
            reward = 1
            self.done = True
        elif new_product > 1000:
            reward = -100
            self.done = True
        else:
            reward = -10
            self.done = False

        # Prepare observation
        observation = np.array(self.cumulative_products, dtype=np.int32)

        # Switch to the next player
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        return observation, reward, self.done, False, {}

    def render(self):
        output = f"Player 1's cumulative product: {self.cumulative_products[0]}\n"
        output += f"Player 2's cumulative product: {self.cumulative_products[1]}\n"
        output += f"Current turn: Player {self.current_player}\n"
        return output

    def valid_moves(self):
        if self.done:
            return []
        current_index = self.current_player - 1
        current_product = self.cumulative_products[current_index]
        valid_actions = []
        for action in range(8):  # action indices from 0 to 7
            number_choice = action + 2  # number choices from 2 to 9
            new_product = current_product * number_choice
            if new_product <= 1000:
                valid_actions.append(action)
        return valid_actions
