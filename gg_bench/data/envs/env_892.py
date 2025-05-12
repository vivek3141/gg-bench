import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define constants
        self.MAX_NUMBER = 1000  # Maximum number allowed in the game
        self.MAX_SEQ_LENGTH = 100  # Max length of the sequence

        # Define action and observation space
        # Actions are integers from 0 to MAX_NUMBER - 1, mapping to numbers from 1 to MAX_NUMBER
        self.action_space = spaces.Discrete(self.MAX_NUMBER)
        # Observation is the sequence of numbers selected so far, padded with zeros
        self.observation_space = spaces.Box(
            low=0, high=self.MAX_NUMBER, shape=(self.MAX_SEQ_LENGTH,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = [1]
        self.numbers_used = set(self.sequence)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.turn = 0  # Number of turns taken
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def _get_observation(self):
        obs = np.zeros(self.MAX_SEQ_LENGTH, dtype=np.int32)
        seq_length = len(self.sequence)
        obs[:seq_length] = self.sequence
        return obs

    def _get_valid_moves(self, N):
        min_M = N + 1
        max_M = min(2 * N, self.MAX_NUMBER)
        possible_M = set(range(min_M, max_M + 1))
        valid_moves = possible_M - self.numbers_used
        return valid_moves

    def valid_moves(self):
        if self.done:
            return []
        N = self.sequence[-1]
        valid_moves = self._get_valid_moves(N)
        # Map valid moves (M) to action indices (action = M - 1)
        valid_action_indices = [M - 1 for M in valid_moves]
        return valid_action_indices

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        M = action + 1  # Map action index to actual number
        N = self.sequence[-1]

        # Check if move is valid according to the rules
        is_valid_move = M > N and M <= 2 * N and M not in self.numbers_used

        if not is_valid_move:
            # Invalid move, player loses
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            return self._get_observation(), reward, terminated, truncated, {}

        # Valid move, update the game state
        self.sequence.append(M)
        self.numbers_used.add(M)

        # Check if opponent can make a valid move
        N_next = M
        valid_moves_opponent = self._get_valid_moves(N_next)

        if not valid_moves_opponent:
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1
            terminated = True
            truncated = False
            return self._get_observation(), reward, terminated, truncated, {}

        # Switch player
        self.current_player = 3 - self.current_player  # Switch player (1->2, 2->1)
        self.turn += 1

        reward = 0
        terminated = False
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        output = f"Current player: Player {self.current_player}\n"
        output += f"Sequence: {self.sequence}\n"
        output += f"Numbers Used: {sorted(self.numbers_used)}\n"
        if not self.done:
            valid_move_numbers = sorted(self._get_valid_moves(self.sequence[-1]))
            output += f"Valid Moves: {valid_move_numbers}\n"
        else:
            output += "Game over.\n"
        return output
