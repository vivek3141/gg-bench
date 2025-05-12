import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum possible number value in the pool
        self.max_number_value = 20

        # Define action and observation space
        # Actions correspond to numbers that can be merged
        self.action_space = spaces.Discrete(self.max_number_value + 1)

        # Observation is a count of numbers in the pool from 0 to max_number_value
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.max_number_value + 1,),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool with two 1s, two 2s, and two 3s
        self.observation = np.zeros(self.max_number_value + 1, dtype=np.int32)
        self.observation[1] = 2
        self.observation[2] = 2
        self.observation[3] = 2

        # Set the starting player to Player 1
        self.current_player = 1

        # Game is not over
        self.done = False

        return self.observation.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation.copy(), 0, True, False, {}

        # Check if the action is valid
        if self.observation[action] >= 2:
            # Perform the merge
            self.observation[action] -= 2
            new_number = action * 2

            # Ensure the new number doesn't exceed the maximum value
            if new_number <= self.max_number_value:
                self.observation[new_number] += 1
            else:
                # Handle cases where the new number exceeds the maximum allowed
                # For simplicity, we assume numbers won't exceed max_number_value
                pass

            # Switch to the next player
            self.current_player = 3 - self.current_player

            # Check if the next player can make a move
            if not self.can_player_move():
                # Opponent cannot move, current player wins
                self.done = True
                reward = 1  # Current player wins
            else:
                # Game continues
                reward = 0
        else:
            # Invalid move or no valid moves, current player loses
            self.done = True
            reward = -10  # Current player loses

        return self.observation.copy(), reward, self.done, False, {}

    def can_player_move(self):
        # Check if there is any number with a count of at least 2
        return any(count >= 2 for count in self.observation)

    def render(self):
        # Return a string representation of the current pool of numbers
        pool = []
        for number, count in enumerate(self.observation):
            pool.extend([number] * count)
        pool.sort()
        return f"Current pool: {pool}"

    def valid_moves(self):
        # Return a list of valid actions (numbers with count >= 2)
        return [i for i, count in enumerate(self.observation) if count >= 2]
