import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=60):
        super(CustomEnv, self).__init__()
        self.starting_number = starting_number
        self.current_number = self.starting_number
        self.current_player = 0  # 0 or 1

        # Define action and observation space
        # Action space: Discrete actions representing possible factors (from 0 up to starting_number)
        self.max_action = (
            self.starting_number + 1
        )  # Actions range from 0 to max_action-1
        self.action_space = spaces.Discrete(self.max_action)

        # Observation space: Current number and current player (0 or 1)
        self.observation_space = spaces.Box(
            low=np.array([2, 0]),
            high=np.array([self.starting_number, 1]),
            shape=(2,),
            dtype=np.int32,
        )

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 0  # Start with player 0
        self.done = False
        observation = np.array(
            [self.current_number, self.current_player], dtype=np.int32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Episode is over
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move
            self.done = True
            reward = -10
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.current_number = self.current_number // action

        # Check if the next player has any valid moves
        next_valid_moves = self.get_factors(self.current_number)
        if not next_valid_moves:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Switch player
            self.current_player = 1 - self.current_player
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            reward = 0
            return observation, reward, False, False, {}

    def render(self):
        board_str = f"Current player: Player {self.current_player}\n"
        board_str += f"Current number: {self.current_number}"
        return board_str

    def valid_moves(self):
        return self.get_factors(self.current_number)

    def get_factors(self, number):
        factors = []
        for i in range(2, number):  # Exclude 1 and the number itself
            if number % i == 0:
                factors.append(i)
        return factors
