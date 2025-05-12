import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: integers from 0 to 9 correspond to numbers 1 to 10
        self.action_space = spaces.Discrete(10)

        # Set maximum sequence length
        self.max_sequence_length = 100

        # Observation space: sequence of numbers with padding
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(self.max_sequence_length,), dtype=np.int8
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.zeros(self.max_sequence_length, dtype=np.int8)
        self.sequence_length = 0
        self.current_player = 1  # Player 1: 1, Player 2: -1
        self.done = False
        return self.sequence.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.sequence.copy(), 0, True, False, {}

        # Map action to number (1 to 10)
        number = action + 1

        # Add number to sequence
        if self.sequence_length < self.max_sequence_length:
            self.sequence[self.sequence_length] = number
            self.sequence_length += 1

            # Check losing condition
            if self.sequence_length >= 3:
                sum_last_three = np.sum(
                    self.sequence[self.sequence_length - 3 : self.sequence_length]
                )
                if sum_last_three >= 15:
                    # Current player loses
                    self.done = True
                    reward = -1  # Penalize the player who lost
                    return self.sequence.copy(), reward, True, False, {}

            # Switch player
            self.current_player *= -1
            reward = 0
            return self.sequence.copy(), reward, False, False, {}
        else:
            # Sequence has reached max length
            self.done = True
            reward = 0
            return self.sequence.copy(), reward, True, False, {}

    def render(self):
        sequence_str = "Sequence: " + str(
            self.sequence[: self.sequence_length].tolist()
        )
        current_player_str = "Current Player: Player {}".format(
            1 if self.current_player == 1 else 2
        )
        return sequence_str + "\n" + current_player_str

    def valid_moves(self):
        if self.done:
            return []
        valid_actions = []
        for action in range(10):
            number = action + 1
            # Simulate adding the number to the sequence
            if self.sequence_length < self.max_sequence_length:
                temp_sequence = self.sequence.copy()
                temp_sequence[self.sequence_length] = number
                temp_sequence_length = self.sequence_length + 1
                if temp_sequence_length >= 3:
                    sum_last_three = np.sum(
                        temp_sequence[temp_sequence_length - 3 : temp_sequence_length]
                    )
                    if sum_last_three < 15:
                        valid_actions.append(action)
                else:
                    valid_actions.append(action)  # Less than 3 numbers, move is valid
        if not valid_actions:
            # No valid moves; all moves result in loss; must choose one
            valid_actions = list(range(10))
        return valid_actions
