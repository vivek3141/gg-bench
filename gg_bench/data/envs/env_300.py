import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Default target number (N)
        self.target_number = 60

        # Define the factors and action space
        self.factors = np.arange(2, 10)  # Factors from 2 to 9 inclusive
        self.num_actions = len(self.factors)
        self.action_space = spaces.Discrete(
            self.num_actions
        )  # Actions 0 to 7 correspond to factors 2 to 9

        # Observation space:
        # [current player's product, opponent's product, current player (1 or 2)]
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 1]),
            high=np.array([self.target_number, self.target_number, 2]),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Optionally allow setting a custom target number
        if options is not None and "target_number" in options:
            self.target_number = options["target_number"]

        self.player_products = {1: 1, 2: 1}  # Both players start with a product of 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_obs()
        return observation, {}  # Return initial observation and empty info

    def step(self, action):
        if self.done:
            # If the game is already over, any action is invalid
            return self._get_obs(), -10, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action played
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        factor = self.factors[action]
        current_product = self.player_products[self.current_player]
        new_product = current_product * factor

        # Update the current player's product
        self.player_products[self.current_player] = new_product

        if new_product == self.target_number:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}
        elif new_product > self.target_number:
            # Current player loses by exceeding the target
            self.done = True
            reward = -1
            return self._get_obs(), reward, True, False, {}
        else:
            # Game continues, switch to the other player
            self.current_player = (
                3 - self.current_player
            )  # Switch player between 1 and 2
            reward = 0
            return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        # Observation: [current player's product, opponent's product, current player]
        return np.array(
            [
                self.player_products[self.current_player],
                self.player_products[3 - self.current_player],
                self.current_player,
            ],
            dtype=np.int32,
        )

    def render(self):
        # Generate a string representation of the current state
        s = f"Target Number (N): {self.target_number}\n"
        s += f"Player {self.current_player}'s Turn:\n"
        s += f"Player {self.current_player}'s Current Product: {self.player_products[self.current_player]}\n"
        s += f"Player {3 - self.current_player}'s Current Product: {self.player_products[3 - self.current_player]}\n"
        s += f"Valid Moves: {self.valid_moves()} (Corresponding Factors: {[self.factors[a] for a in self.valid_moves()]})\n"
        return s

    def valid_moves(self):
        # Return a list of valid action indices based on the current state
        current_product = self.player_products[self.current_player]
        valid_actions = []
        for idx, factor in enumerate(self.factors):
            if current_product * factor <= self.target_number:
                valid_actions.append(idx)
        return valid_actions
