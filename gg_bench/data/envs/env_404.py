import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the target number
        self.target_number = 100

        # Define action and observation space
        # There are 8 possible actions corresponding to numbers 2-9
        self.action_space = spaces.Discrete(8)

        # Observation space consists of:
        # - Current player's product (1 to target_number)
        # - Opponent's product (1 to target_number)
        # - Available numbers (0 or 1 for each number from 2 to 9)
        # Total size: 2 (products) + 8 (availability of numbers) = 10
        self.observation_space = spaces.Box(
            low=np.array([1, 1] + [0] * 8),
            high=np.array([self.target_number, self.target_number] + [1] * 8),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the shared pool of numbers from 2 to 9 inclusive
        self.shared_pool = list(range(2, 10))

        # Initialize players' products
        self.player_products = {1: 1, 2: 1}

        # Randomly select which player starts
        self.current_player = self.np_random.choice([1, 2])

        # Set the game as not done
        self.done = False

        # Return the initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Map action index to selected number (2-9)
        selected_number = action + 2

        # Check if the selected number is in the shared pool
        if selected_number not in self.shared_pool:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Remove the selected number from the shared pool
        self.shared_pool.remove(selected_number)

        # Multiply the current player's product by the selected number
        self.player_products[self.current_player] *= selected_number
        current_product = self.player_products[self.current_player]

        # Check for win condition
        if current_product == self.target_number:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check for loss condition
        if current_product > self.target_number:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if there are no valid moves left for the next player
        if not self.valid_moves():
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        # Return the observation with zero reward and not done
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Generate a string representation of the game state
        obs = self._get_observation()
        output = "----------------------------------------\n"
        output += f"It's Player {self.current_player}'s turn.\n"
        output += f"Current Products: Player 1 = {self.player_products[1]}, Player 2 = {self.player_products[2]}\n"
        available_numbers = [
            str(num) for num, available in zip(range(2, 10), obs[2:]) if available == 1
        ]
        output += f"Available Numbers: {', '.join(available_numbers)}\n"
        output += "----------------------------------------\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for idx, num in enumerate(range(2, 10)):
            if num in self.shared_pool:
                # Check if multiplying by this number would exceed the target
                potential_product = self.player_products[self.current_player] * num
                if potential_product <= self.target_number:
                    valid_actions.append(idx)
        return valid_actions

    def _get_observation(self):
        # Generate the observation array
        # Elements 0 and 1: current player's product and opponent's product
        # Elements 2-9: availability of numbers 2-9
        available_numbers = np.array(
            [1 if num in self.shared_pool else 0 for num in range(2, 10)],
            dtype=np.int32,
        )
        current_product = self.player_products[self.current_player]
        opponent = 2 if self.current_player == 1 else 1
        opponent_product = self.player_products[opponent]
        observation = np.array([current_product, opponent_product], dtype=np.int32)
        observation = np.concatenate([observation, available_numbers])
        return observation
