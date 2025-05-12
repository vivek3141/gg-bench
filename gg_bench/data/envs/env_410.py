import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to placing numbers 1 to 6 on the stack
        self.action_space = spaces.Discrete(
            6
        )  # Actions 0 to 5 correspond to numbers 1 to 6
        self.observation_space = spaces.Box(
            low=0, high=6, shape=(3,), dtype=np.int8
        )  # Top 3 numbers on the stack, zero-padded

        self.stack = []
        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.stack = []
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if self.done:
            # If the game is over, ignore further actions
            observation = self._get_observation()
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Reward for acting when game is over

        # Map action to token (number between 1 and 6)
        token = action + 1

        # Place the token on the stack
        self.stack.append(token)

        # Calculate the sum of top three numbers
        sum_top3 = sum(self.stack[-3:])

        if sum_top3 >= 12:
            # Current player loses
            self.done = True
            reward = -1  # Negative reward for losing
            observation = self._get_observation()
            return observation, reward, True, False, {}  # Game over
        else:
            # Game continues
            reward = 0  # No reward for valid move
            self._switch_player()
            observation = self._get_observation()
            return observation, reward, False, False, {}  # Continue game

    def render(self):
        # Return a string representation of the stack (top to bottom)
        stack_representation = (
            "Current Stack (top to bottom): ["
            + ", ".join(map(str, reversed(self.stack)))
            + "]"
        )
        return stack_representation

    def valid_moves(self):
        # All moves are valid (numbers 1 to 6)
        return [i for i in range(6)]  # Actions 0 to 5 correspond to numbers 1 to 6

    def _get_observation(self):
        # Get the top 3 numbers on the stack, zero-padded if fewer
        top_numbers = self.stack[-3:] if len(self.stack) >= 3 else self.stack
        padded_top_numbers = [0] * (3 - len(top_numbers)) + top_numbers
        observation = np.array(padded_top_numbers, dtype=np.int8)
        return observation

    def _switch_player(self):
        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1
