import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.MAX_NUMBER = 100  # Maximum starting number N

        # Define action space: Possible divisors from 0 to MAX_NUMBER
        self.action_space = spaces.Discrete(self.MAX_NUMBER + 1)  # Actions 0 to 100

        # Define observation space:
        #   obs[0]: Normalized current N (between 0 and 1)
        #   obs[1:]: Binary indicators for used divisors (1 if used, 0 otherwise)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.MAX_NUMBER + 1,), dtype=np.float32
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N_start = 100  # Starting number N
        self.N = self.N_start  # Current number N
        self.used_divisors = set()  # Set to keep track of used divisors
        self.done = False  # Game over flag
        self.current_player = 1  # Player 1 starts (1 or -1)
        obs = self._get_obs()
        return obs, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return current state
            obs = self._get_obs()
            return obs, 0.0, True, False, {}
        # Check if the action is a valid move
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move: game over and negative reward
            self.done = True
            reward = -10.0
            obs = self._get_obs()
            return obs, reward, True, False, {}
        else:
            # Valid move: update the game state
            self.N = self.N // action  # Update current number N
            self.used_divisors.add(action)  # Record the used divisor
            # Check if the game has been won
            if self.N == 1:
                self.done = True
                reward = 1.0  # Current player wins
            else:
                reward = -10.0  # Valid move made, game continues
                self.current_player *= -1  # Switch player
            obs = self._get_obs()
            return (
                obs,
                reward,
                self.done,
                False,
                {},
            )  # Return observation, reward, done, info

    def _get_obs(self):
        # Create an observation array
        obs = np.zeros(self.MAX_NUMBER + 1, dtype=np.float32)
        # Normalize current N and set as the first element
        obs[0] = self.N / float(self.N_start)
        # Mark used divisors in the observation
        for d in self.used_divisors:
            if d <= self.MAX_NUMBER:
                obs[d] = 1.0
        return obs

    def valid_moves(self):
        # Generate list of valid divisors for current N
        divisors = [
            i
            for i in range(2, self.N + 1)
            if self.N % i == 0 and i not in self.used_divisors
        ]
        return divisors  # Return list of valid moves

    def render(self):
        # Create a string representation of the current game state
        output = f"Current N: {self.N}\n"
        output += f"Used Divisors: {sorted(self.used_divisors)}\n"
        output += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += f"Valid Moves: {self.valid_moves()}\n"
        return output
