import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: choose a number from 1 to 10
        # Actions 0-9 correspond to numbers 1-10
        self.action_space = spaces.Discrete(10)

        # Observation space: cumulative sum (0-100), current_player (0 or 1)
        # MultiDiscrete([101, 2]) allows cumulative sums from 0-100 and current_player 0 or 1
        self.observation_space = spaces.MultiDiscrete([101, 2])

        # Primes greater than 50 and less than or equal to 100
        self.primes = {53, 59, 61, 67, 71, 73, 79, 83, 89, 97}

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cumulative_sums = {1: 0, 2: 0}
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Observation is [cumulative_sum, current_player - 1] (current_player - 1 to make it 0 or 1)
        observation = np.array(
            [self.cumulative_sums[self.current_player], self.current_player - 1],
            dtype=np.int32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            observation = np.array(
                [self.cumulative_sums[self.current_player], self.current_player - 1],
                dtype=np.int32,
            )
            reward = -10  # Penalty for taking action after game is over
            return observation, reward, True, False, {}

        number = action + 1  # Map actions 0-9 to numbers 1-10
        current_sum = self.cumulative_sums[self.current_player]

        # Check if any valid moves are available
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # No valid moves, player loses
            self.done = True
            reward = -10
            observation = np.array(
                [current_sum, self.current_player - 1], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Check if selected action is valid
        if action not in valid_actions:
            # Invalid move, player loses
            self.done = True
            reward = -10
            observation = np.array(
                [current_sum, self.current_player - 1], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Update cumulative sum
        current_sum += number
        self.cumulative_sums[self.current_player] = current_sum

        # Check for win condition
        if current_sum in self.primes:
            # Player wins
            self.done = True
            reward = 1
            observation = np.array(
                [current_sum, self.current_player - 1], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Check if cumulative sum exceeds 100 (should not happen due to valid_moves check)
        if current_sum > 100:
            # Player loses
            self.done = True
            reward = -10
            observation = np.array(
                [current_sum, self.current_player - 1], dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0  # No reward for normal move
        done = False

        # Next player's observation
        observation = np.array(
            [self.cumulative_sums[self.current_player], self.current_player - 1],
            dtype=np.int32,
        )
        return observation, reward, done, False, {}

    def render(self):
        # Return a string representation of the game state
        return f"Player {self.current_player}'s turn. Your cumulative sum is {self.cumulative_sums[self.current_player]}."

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        current_sum = self.cumulative_sums[self.current_player]
        return [a for a in range(10) if current_sum + (a + 1) <= 100]
