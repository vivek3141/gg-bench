import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (append '0'), 1 (append '1')
        self.action_space = spaces.Discrete(2)

        # Observation space: [current_modulo, current_player]
        # current_modulo: 0, 1, or 2
        # current_player: 0 or 1
        self.observation_space = spaces.MultiDiscrete(
            [3, 2]
        )  # Modulo (0,1,2), Player (0,1)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_modulo = 1  # Starts with binary '1' (Decimal 1)
        self.current_player = 0  # Player 0 starts
        self.done = False
        self.binary_str = "1"  # Initialize binary string
        return self._get_obs(), {}  # Observation, info

    def _get_obs(self):
        return np.array([self.current_modulo, self.current_player], dtype=int)

    def step(self, action):
        if action not in [0, 1]:
            raise ValueError("Invalid action. Action must be 0 or 1.")

        if self.done:
            return (
                self._get_obs(),
                0,
                self.done,
                False,
                {},
            )  # No reward after the game is done

        # Update binary string
        self.binary_str += str(action)

        # Compute new modulo
        new_modulo = (self.current_modulo * 2 + action) % 3

        if new_modulo == 0:
            # Invalid move - current player loses
            reward = -10
            self.done = True
            info = {"reason": "Invalid move - number is divisible by 3"}
        else:
            # Check if the next player has any valid moves
            next_modulo_0 = (new_modulo * 2 + 0) % 3
            next_modulo_1 = (new_modulo * 2 + 1) % 3

            if next_modulo_0 == 0 and next_modulo_1 == 0:
                # Next player has no valid moves - current player wins
                reward = 1
                self.done = True
                info = {"reason": "Opponent has no valid moves"}
            else:
                # Game continues
                reward = 0
                info = {}
                self.current_player = 1 - self.current_player  # Switch players

        # Update the current modulo
        self.current_modulo = new_modulo

        return (
            self._get_obs(),
            reward,
            self.done,
            False,
            info,
        )  # Observation, reward, done, truncated, info

    def render(self):
        # Display the current state of the game
        decimal_value = int(self.binary_str, 2)
        state_representation = (
            f"Current Binary Number: {self.binary_str} (Decimal: {decimal_value})\n"
            f"Current Player: Player {self.current_player + 1}\n"
            f"Current Modulo (mod 3): {self.current_modulo}\n"
        )
        print(state_representation)

    def valid_moves(self):
        # Return a list of valid actions (bits that can be appended without losing)
        valid = []
        for action in [0, 1]:
            new_modulo = (self.current_modulo * 2 + action) % 3
            if new_modulo != 0:
                valid.append(action)
        return valid
