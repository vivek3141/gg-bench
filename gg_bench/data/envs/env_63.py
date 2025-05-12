import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N  # Length of the sequence

        # Define action and observation space
        self.action_space = spaces.Discrete(self.N)

        # Observation space:
        # - First N entries: 0 or 1 indicating number availability
        # - Last entry: the last number selected (0 if no number has been selected yet)
        self.observation_space = spaces.Box(
            low=np.zeros(self.N + 1, dtype=np.int32),
            high=np.concatenate(
                [np.ones(self.N, dtype=np.int32), np.array([self.N], dtype=np.int32)]
            ),
            dtype=np.int32,
        )

        self.seed()

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.ones(self.N, dtype=np.int32)
        self.last_number = 0  # No number selected yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        if self.done:
            # Game is already over
            return self._get_obs(), 0, True, False, info

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            terminated = True
            self.done = True
            return self._get_obs(), reward, terminated, truncated, info

        # Valid move
        # Remove the selected number from sequence
        self.sequence[action] = 0
        number_selected = action + 1  # Number corresponds to index + 1
        self.last_number = number_selected

        # Check if the next player can make a move
        original_player = self.current_player
        self.current_player = 3 - self.current_player  # Switch player
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Next player cannot move; current player wins
            reward = 1
            terminated = True
            self.done = True
            # Switch back to winning player
            self.current_player = original_player
        else:
            # Game continues
            reward = 0
            # Keep current_player as next player

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        sequence_str = "Current sequence: ["
        for i in range(self.N):
            if self.sequence[i] == 1:
                sequence_str += f"{i+1}, "
            else:
                sequence_str += "_, "
        sequence_str = sequence_str[:-2] + "]"
        return (
            f"{sequence_str}\n"
            f"Current player: {self.current_player}\n"
            f"Last number selected: {self.last_number}"
        )

    def valid_moves(self):
        if self.last_number == 0:
            # First turn; any available number can be selected
            return [i for i in range(self.N) if self.sequence[i] == 1]
        else:
            prev_number = self.last_number
            valid_moves = []
            for i in range(self.N):
                if self.sequence[i] == 1:
                    number = i + 1
                    if prev_number % number == 0 or number % prev_number == 0:
                        valid_moves.append(i)
            return valid_moves

    def _get_obs(self):
        # Returns the observation
        obs = np.concatenate(
            [self.sequence, np.array([self.last_number], dtype=np.int32)]
        )
        return obs
