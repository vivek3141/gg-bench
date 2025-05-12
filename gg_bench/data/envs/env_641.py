import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(8), actions correspond to multipliers from 2 to 9
        self.action_space = spaces.Discrete(8)

        # Observation space: cumulative total
        # Since cumulative total can grow large, set high to a large number
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([1e20]), dtype=np.float64
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # For generating random numbers
        self.np_random, _ = seeding.np_random(seed)
        self.cumulative_total = 1.0  # Starting cumulative total
        self.done = False
        return np.array([self.cumulative_total], dtype=np.float64), {}

    def step(self, action):
        if self.done:
            # Game is over
            return (
                np.array([self.cumulative_total], dtype=np.float64),
                0,
                True,
                False,
                {},
            )

        # Map action to multiplier
        multiplier = action + 2  # action 0 corresponds to multiplier 2

        # Apply multiplier to cumulative total
        self.cumulative_total *= multiplier

        # Check if agent loses
        if self.cumulative_total >= 100:
            # Agent loses
            self.done = True
            reward = -10  # Negative reward for losing
            return (
                np.array([self.cumulative_total], dtype=np.float64),
                reward,
                True,
                False,
                {},
            )

        # Agent's valid move, simulate opponent's move
        # Reward for valid move is -10
        reward = -10

        # Simulate opponent's move (randomly choose action)
        valid_actions = list(range(self.action_space.n))
        opponent_action = self.np_random.choice(valid_actions)
        opponent_multiplier = opponent_action + 2
        self.cumulative_total *= opponent_multiplier

        # Check if opponent loses
        if self.cumulative_total >= 100:
            # Opponent loses, agent wins
            self.done = True
            reward = 1  # Positive reward for winning
            return (
                np.array([self.cumulative_total], dtype=np.float64),
                reward,
                True,
                False,
                {},
            )

        # Game continues, next is agent's turn
        return (
            np.array([self.cumulative_total], dtype=np.float64),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        print(f"Current total: {self.cumulative_total}")

    def valid_moves(self):
        # Return list of valid actions (indices in action space)
        return list(range(self.action_space.n))  # All actions are valid
