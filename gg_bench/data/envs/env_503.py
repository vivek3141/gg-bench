import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Discrete(8) for multipliers 2 to 9
        self.action_space = spaces.Discrete(8)

        # Define observation space: Current Number (float), between 1 and a very large number
        self.observation_space = spaces.Box(
            low=np.array([1.0]), high=np.array([1e12]), shape=(1,), dtype=np.float64
        )

        # Initialize game state
        self.current_number = None
        self.current_player = None
        self.done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.float64),
            {},
        )  # Return observation and info

    def step(self, action):
        # Check if action is valid (action in 0..7)
        if not self.action_space.contains(action):
            # Invalid action: reward -10, terminate the game
            reward = -10
            self.done = True
            return (
                np.array([self.current_number], dtype=np.float64),
                reward,
                True,
                False,
                {},
            )

        # Convert action index to multiplier (action 0 corresponds to multiplier 2)
        multiplier = action + 2

        # Update Current Number
        self.current_number *= multiplier

        # Check game-ending condition
        if self.current_number > 1000:
            # Current player loses
            reward = -1  # Reward of -1 for losing
            self.done = True
        else:
            # Valid move, game continues
            reward = 0  # No reward for continuing
            self.done = False
            # Switch to the other player
            self.current_player = 2 if self.current_player == 1 else 1

        return (
            np.array([self.current_number], dtype=np.float64),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        # Return a string representing the current state
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        # Return a list of valid action indices (0 to 7 corresponding to multipliers 2 to 9)
        valid_actions = []
        for action in range(self.action_space.n):
            multiplier = action + 2
            if self.current_number * multiplier <= 1000:
                valid_actions.append(action)
        # If no safe moves, all moves are valid (player must choose one and lose)
        if not valid_actions:
            valid_actions = list(range(self.action_space.n))
        return valid_actions
