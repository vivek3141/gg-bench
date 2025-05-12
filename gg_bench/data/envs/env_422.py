import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_N=15):
        super(CustomEnv, self).__init__()

        # Actions correspond to digits 1-9, represented as actions 0-8
        self.action_space = spaces.Discrete(
            9
        )  # Actions 0 to 8 correspond to digits 1 to 9

        # Observation is the current value of N
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.int64
        )

        self.starting_N = starting_N

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.N = self.starting_N
        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.N]), {}

    def step(self, action):
        if self.done:
            # If game is already over, return the current state
            return np.array([self.N]), 0, True, False, {}

        # First, check if there are any valid moves for current player
        valid_actions = self.valid_moves()
        if not valid_actions:
            # No valid moves, current player loses
            self.done = True
            reward = -1  # Player loses
            return np.array([self.N]), reward, self.done, False, {}

        # Map the action to a digit
        digit = action + 1  # action 0 corresponds to digit 1

        # Check if the action is one of the valid actions
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid action
            return np.array([self.N]), reward, self.done, False, {}

        # Valid move
        self.N -= digit
        # Check if N has reached zero
        if self.N == 0:
            self.done = True
            reward = 1  # Current player wins
            return np.array([self.N]), reward, self.done, False, {}
        else:
            # Game continues
            reward = 0
            # Switch player
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            return np.array([self.N]), reward, self.done, False, {}

    def valid_moves(self):
        # Return list of valid action indices (0-8)
        digits_in_N = [int(d) for d in str(self.N) if d != "0"]
        valid_digits = set(digits_in_N)
        # Exclude zeros and digits that when subtracted make N negative
        valid_actions = []
        for digit in valid_digits:
            if self.N - digit >= 0:
                action = digit - 1  # Map digit back to action index
                valid_actions.append(action)
        return valid_actions

    def render(self):
        output = f"Player {self.current_player}'s turn. Current N: {self.N}\n"
        output += f"Valid digits to subtract: {sorted([action+1 for action in self.valid_moves()])}\n"
        return output
