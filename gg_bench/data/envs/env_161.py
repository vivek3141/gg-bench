import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Set maximum N value
        self.N_max = 100  # Maximum value for N
        # Action space: For each possible k (split number), two choices (select k or N - k)
        self.action_space = spaces.Discrete(self.N_max * 2)
        # Observation space: Current N
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the starting number N
        if options is not None and "initial_N" in options:
            self.N = options["initial_N"]
        else:
            # Default initial N if not specified
            self.N = 16
        self.current_player = 1  # Player 1 starts (1 or -1 to represent players)
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game has already ended
            return np.array([self.N], dtype=np.int32), 0, True, False, {}
        # Decode action into split number k and choice s
        k = action // 2 + 1  # k ranges from 1 upwards
        s = action % 2  # s = 0 or 1, indicating which number to select as new N
        # Check if the action is valid
        if k >= self.N or k >= (self.N - k):
            # Invalid action: numbers are equal or k >= N - k
            reward = -10  # Penalty for invalid move
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        # Valid split
        split_numbers = (k, self.N - k)
        # Player chooses new N based on s
        if s == 0:
            new_N = split_numbers[0]
        else:
            new_N = split_numbers[1]
        self.N = new_N  # Update N for the next player
        # Switch player
        self.current_player *= -1
        # Check if the next player has any valid moves
        if not self.has_valid_moves(self.N):
            # Opponent cannot make a move, current player wins
            self.done = True
            reward = 1  # Reward for winning
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            # Game continues without reward
            reward = 0
            return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def has_valid_moves(self, N):
        # Returns True if there is at least one valid move for given N
        max_k = (N - 1) // 2
        return max_k >= 1  # At least one valid k exists

    def render(self):
        # Provide a simple string representation of the current game state
        return f"Current N: {self.N}, Player's Turn: {'1' if self.current_player == 1 else '2'}"

    def valid_moves(self):
        if self.done:
            return []
        N = self.N
        max_k = (N - 1) // 2
        valid_actions = []
        # Generate all valid actions based on current N
        for k in range(1, max_k + 1):
            for s in [0, 1]:
                action = (k - 1) * 2 + s
                valid_actions.append(action)
        return valid_actions
