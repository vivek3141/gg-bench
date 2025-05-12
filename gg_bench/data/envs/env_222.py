import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is digits from 1 to 9
        self.action_space = spaces.Discrete(9)  # Actions 0-8 correspond to digits 1-9

        # Define the maximum sequence length
        self.max_sequence_length = 20  # You can adjust this value as needed

        # The observation space is the sequence of digits
        # We use zeros to represent empty positions
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_sequence_length,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.zeros(self.max_sequence_length, dtype=np.int32)
        self.sequence_length = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Return a copy of the current sequence
        return self.sequence.copy()

    def step(self, action):
        # Check if game is already done
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Validate the action
        if action < 0 or action >= 9:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {"error": "Invalid action"}

        digit = action + 1  # Map action 0-8 to digit 1-9

        # Check if sequence is already full
        if self.sequence_length >= self.max_sequence_length:
            # Cannot add more digits
            self.done = True
            return self._get_obs(), -10, True, False, {"error": "Sequence is full"}

        # Append the digit to the sequence
        self.sequence[self.sequence_length] = digit
        self.sequence_length += 1

        # Check for loss condition: any three consecutive digits sum to 15
        loss = False
        for i in range(self.sequence_length - 3 + 1):
            sum_of_three = (
                self.sequence[i] + self.sequence[i + 1] + self.sequence[i + 2]
            )
            if sum_of_three == 15:
                loss = True
                break

        if loss:
            # Current player loses
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}
        else:
            # Valid move, game continues
            reward = 1  # As per the prompt, we give a reward of 1 for a valid move
            # Switch player
            self.current_player = (
                3 - self.current_player
            )  # Switch between player 1 and 2
            return self._get_obs(), reward, False, False, {}

    def render(self):
        # Visual representation of sequence
        sequence_str = "Current sequence: " + " ".join(
            str(int(d)) for d in self.sequence[: self.sequence_length]
        )
        sequence_str += f"\nCurrent player: Player {self.current_player}"
        return sequence_str

    def valid_moves(self):
        # Returns list of valid actions (0-8)
        if self.done or self.sequence_length >= self.max_sequence_length:
            return []  # No valid moves
        else:
            return list(range(9))
