import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        # Observation space: For each cell, store [number, owner_status]
        # number: 1 to 9
        # owner_status: -1 (Player 2), 0 (Unclaimed), 1 (Player 1)
        self.observation_space = spaces.Box(
            low=np.array([[1, -1]] * 9), high=np.array([[9, 1]] * 9), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign numbers 1-9 to the grid positions
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.numbers = np.arange(1, 10)
        self.np_random.shuffle(self.numbers)
        # Initialize owner status to 0 (unclaimed)
        self.owner = np.zeros(9, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Create the observation
        observation = np.stack((self.numbers, self.owner), axis=1)
        info = {}
        return observation, info

    def step(self, action):
        observation = np.stack((self.numbers, self.owner), axis=1)
        info = {}
        # Check for invalid action
        if action < 0 or action >= 9 or self.owner[action] != 0 or self.done:
            reward = -10
            terminated = True
            truncated = False
            self.done = True
            return observation, reward, terminated, truncated, info

        # Claim the cell
        self.owner[action] = self.current_player

        # Check if the game is over (all cells claimed)
        if np.all(self.owner != 0):
            self.done = True
            # Calculate the sums for each player
            player_sum = np.sum(self.numbers[self.owner == self.current_player])
            opponent_sum = np.sum(self.numbers[self.owner == -self.current_player])

            # Determine the winner
            if player_sum > opponent_sum:
                reward = 1  # Current player wins
            elif player_sum < opponent_sum:
                reward = 0  # Current player loses
            else:
                # Tie-breaker: current player loses if sums are equal
                reward = 0  # Current player loses
            terminated = True
            truncated = False
        else:
            # Game continues
            reward = 0
            terminated = False
            truncated = False
            # Switch to the other player
            self.current_player *= -1

        observation = np.stack((self.numbers, self.owner), axis=1)
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        # Build the grid representation
        grid = ""
        for i in range(3):
            grid += "+----+----+----+\n"
            grid += "|"
            for j in range(3):
                idx = i * 3 + j
                owner = self.owner[idx]
                if owner == 1:
                    cell = " X "
                elif owner == -1:
                    cell = " O "
                else:
                    cell = f" {self.numbers[idx]} "
                grid += cell + "|"
            grid += "\n"
        grid += "+----+----+----+\n"
        return grid

    def valid_moves(self):
        return [i for i in range(9) if self.owner[i] == 0]
