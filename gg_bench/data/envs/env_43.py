import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Numbers from 2 to 50 inclusive, total of 49 numbers
        self.num_numbers = 49
        # Define action space: indices from 0 to 48, mapping to numbers from 2 to 50
        self.action_space = spaces.Discrete(self.num_numbers)
        # Observation space: array of size 49, values in {-1, 0, 1}
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.num_numbers,), dtype=np.int8
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the observation: all zeros (numbers are in the pool)
        self.observation = np.zeros(self.num_numbers, dtype=np.int8)
        # Current player: 1 or -1
        self.current_player = 1
        self.done = False
        return self.observation, {}  # observation, info

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}
        # Map action to number
        number = action + 2  # numbers from 2 to 50
        # Check if number is in the pool
        if self.observation[action] != 0:
            # Invalid move: number already taken
            self.done = True
            return self.observation, -10, True, False, {}
        # Get current player's captured numbers
        captured_numbers = [
            i + 2
            for i, val in enumerate(self.observation)
            if val == self.current_player
        ]
        # Check coprimality with captured numbers
        for n in captured_numbers:
            if math.gcd(number, n) != 1:
                # Invalid move: shares a common factor > 1 with captured number
                self.done = True
                return self.observation, -10, True, False, {}
        # Valid move
        # Update observation
        self.observation[action] = self.current_player
        # Check if opponent can make a valid move
        opponent_captured_numbers = [
            i + 2
            for i, val in enumerate(self.observation)
            if val == -self.current_player
        ]
        available_actions = [i for i, val in enumerate(self.observation) if val == 0]
        opponent_has_valid_move = False
        for opp_action in available_actions:
            opp_number = opp_action + 2
            # Check coprimality with opponent's captured numbers
            valid = True
            for n in opponent_captured_numbers:
                if math.gcd(opp_number, n) != 1:
                    valid = False
                    break
            if valid:
                opponent_has_valid_move = True
                break
        if not opponent_has_valid_move:
            # Opponent cannot move, current player wins
            self.done = True
            return self.observation, 1, True, False, {}
        else:
            # Switch player
            self.current_player *= -1
            return self.observation, 0, False, False, {}

    def render(self):
        s = "Current player: {}\n".format(
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        s += "Available numbers in pool: "
        pool_numbers = [i + 2 for i, val in enumerate(self.observation) if val == 0]
        s += ", ".join(str(n) for n in pool_numbers) + "\n"
        player1_numbers = [i + 2 for i, val in enumerate(self.observation) if val == 1]
        player2_numbers = [i + 2 for i, val in enumerate(self.observation) if val == -1]
        s += (
            "Player 1's captured numbers: "
            + ", ".join(str(n) for n in player1_numbers)
            + "\n"
        )
        s += (
            "Player 2's captured numbers: "
            + ", ".join(str(n) for n in player2_numbers)
            + "\n"
        )
        return s

    def valid_moves(self):
        # Get current player's captured numbers
        captured_numbers = [
            i + 2
            for i, val in enumerate(self.observation)
            if val == self.current_player
        ]
        # Get available actions
        available_actions = [i for i, val in enumerate(self.observation) if val == 0]
        valid_actions = []
        for action in available_actions:
            number = action + 2
            valid = True
            for n in captured_numbers:
                if math.gcd(number, n) != 1:
                    valid = False
                    break
            if valid:
                valid_actions.append(action)
        return valid_actions
