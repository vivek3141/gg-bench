import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (add 1), 1 (add 2)
        self.action_space = spaces.Discrete(2)
        # Observation: Current number
        self.observation_space = spaces.Box(
            low=1, high=1000, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            # Game is already over
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

        # Map action to addition (0->1, 1->2)
        addition = action + 1
        new_total = self.current_number + addition

        # Check if new total is prime
        if not is_prime(new_total):
            # Current player loses
            self.done = True
            reward = -10  # Loss penalty
            return (
                np.array([new_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

        # Check if opponent has any valid moves
        opponent_has_valid_move = False
        for opponent_action in [0, 1]:
            opponent_addition = opponent_action + 1
            opponent_total = new_total + opponent_addition
            if is_prime(opponent_total):
                opponent_has_valid_move = True
                break

        if not opponent_has_valid_move:
            # Current player wins
            self.done = True
            reward = 1  # Win reward
            self.current_number = new_total
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

        # Game continues
        self.current_number = new_total
        self.current_player *= -1  # Switch player
        reward = -10  # Valid move penalty
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def render(self):
        player = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current Number: {self.current_number}\nCurrent Player: {player}\n"

    def valid_moves(self):
        valid_actions = []
        for action in [0, 1]:
            addition = action + 1
            new_total = self.current_number + addition
            if is_prime(new_total):
                valid_actions.append(action)
        return valid_actions
