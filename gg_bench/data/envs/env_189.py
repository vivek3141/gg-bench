import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum value for N
        self.MAX_N = 100  # Adjust as needed

        # Define action space: Possible divisors from 0 to MAX_N
        self.action_space = spaces.Discrete(self.MAX_N + 1)

        # Define observation space: N (0 to MAX_N), used_divisors flags (0 or 1)
        low = np.zeros(self.MAX_N + 1, dtype=np.int32)  # N and used_divisors
        high = np.ones(self.MAX_N + 1, dtype=np.int32)
        high[0] = self.MAX_N  # N can be up to MAX_N
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Starting number N
        self.N = 30  # You can allow dynamic starting N if desired
        self.used_divisors = np.zeros(self.MAX_N + 1, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Return the initial observation and info
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}

        D = action  # Chosen divisor

        # Check if the action is valid
        if D <= 1 or D >= self.N:
            # Invalid move
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        if self.N % D != 0:
            # Invalid move
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        if self.used_divisors[D] == 1:
            # Divisor has been used already
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Valid move
        self.N -= D
        self.used_divisors[D] = 1  # Mark divisor as used

        # Check if the next player can make a move
        valid_moves_next = self._valid_moves(self.N)
        if not valid_moves_next:
            # Current player wins
            reward = 1
            self.done = True
        else:
            # Game continues
            reward = -10  # As per the instruction
            self.done = False
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1

        # Return observation, reward, done, truncated, info
        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        # Return a string representing the current state
        state_str = f"Current N: {self.N}\n"
        state_str += f"Used Divisors: {np.nonzero(self.used_divisors)[0].tolist()}\n"
        state_str += f"Current Player: Player {self.current_player}"
        return state_str

    def valid_moves(self):
        # Return a list of valid divisors (actions) based on current N
        return self._valid_moves(self.N)

    def _valid_moves(self, N):
        moves = []
        for D in range(2, N):
            if N % D == 0 and self.used_divisors[D] == 0:
                moves.append(D)
        return moves

    def _get_obs(self):
        # Construct the observation array
        observation = np.zeros(self.MAX_N + 1, dtype=np.int32)
        observation[0] = self.N
        observation[1:] = self.used_divisors[1:]
        return observation
