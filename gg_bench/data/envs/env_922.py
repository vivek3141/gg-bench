import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, M=20, N=4):
        super(CustomEnv, self).__init__()

        self.M = M  # Number range (1 to M)
        self.N = N  # Sequence length required to win

        # Define action and observation space
        self.action_space = spaces.Discrete(self.M)  # Actions are numbers from 1 to M
        self.observation_space = spaces.Dict(
            {
                "available_numbers": spaces.Box(
                    low=-1, high=1, shape=(self.M,), dtype=np.int8
                ),  # -1: used by Player -1, 0: available, 1: used by Player 1
                "player1_sequence": spaces.Box(
                    low=0, high=self.M, shape=(self.N,), dtype=np.int8
                ),  # Sequence of Player 1
                "player2_sequence": spaces.Box(
                    low=0, high=self.M, shape=(self.N,), dtype=np.int8
                ),  # Sequence of Player -1
                "current_player": spaces.Box(
                    low=-1, high=1, shape=(), dtype=np.int8
                ),  # -1 or 1
            }
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.available_numbers = np.zeros(
            self.M, dtype=np.int8
        )  # 0: available, 1/-1: used by player
        self.player_sequences = {
            1: np.zeros(self.N, dtype=np.int8),
            -1: np.zeros(self.N, dtype=np.int8),
        }
        self.sequence_lengths = {1: 0, -1: 0}
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action to number (1-based indexing)
        number = action + 1

        # Check if number is available
        if (
            action < 0
            or action >= self.M
            or self.available_numbers[action] != 0
            or action not in self.valid_moves()
        ):
            self.done = True
            reward = -10  # Invalid move
            return self._get_observation(), reward, True, False, {}

        # Update game state
        self.available_numbers[action] = self.current_player
        seq_length = self.sequence_lengths[self.current_player]
        self.player_sequences[self.current_player][seq_length] = number
        self.sequence_lengths[self.current_player] += 1

        # Check for winning condition
        if self.sequence_lengths[self.current_player] >= self.N:
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Check if the next player has valid moves
        self.current_player *= -1  # Switch player
        if not self.valid_moves():
            # Next player cannot move, current player wins
            self.done = True
            reward = 1  # Current player wins because opponent cannot move
            return self._get_observation(), reward, True, False, {}

        # If game not over, continue
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        available_numbers_status = "".join(
            [
                str(num + 1) if status == 0 else "X"
                for num, status in enumerate(self.available_numbers)
            ]
        )
        player1_seq = self.player_sequences[1][: self.sequence_lengths[1]]
        player2_seq = self.player_sequences[-1][: self.sequence_lengths[-1]]
        return (
            f"Available Numbers: {available_numbers_status}\n"
            f"Player 1 Sequence: {player1_seq}\n"
            f"Player 2 Sequence: {player2_seq}\n"
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )

    def valid_moves(self):
        valid_actions = []
        player_seq = self.player_sequences[self.current_player][
            : self.sequence_lengths[self.current_player]
        ]

        for action in range(self.M):
            if self.available_numbers[action] == 0:
                number = action + 1
                if self._is_valid_move(player_seq, number):
                    valid_actions.append(action)
        return valid_actions

    def _is_valid_move(self, seq, number):
        if len(seq) == 0:
            return True  # Any number is valid for the first move
        if len(seq) == 1:
            return True  # Any number is valid for the second move
        else:
            # Calculate previous difference
            prev_diff = abs(seq[-1] - seq[-2])
            # Calculate new difference
            new_diff = abs(number - seq[-1])
            if new_diff > prev_diff:
                return True
        return False

    def _get_observation(self):
        return {
            "available_numbers": self.available_numbers.copy(),
            "player1_sequence": self.player_sequences[1].copy(),
            "player2_sequence": self.player_sequences[-1].copy(),
            "current_player": self.current_player,
        }
