import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: actions 0-8 correspond to numbers 1-9
        self.action_space = spaces.Discrete(9)
        # Observation space: sequence of up to 10 numbers (integers from 1 to 9), pad with zeros
        self.observation_space = spaces.Box(low=0, high=9, shape=(10,), dtype=np.int32)

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.zeros(10, dtype=np.int32)  # Sequence starts empty
        self.current_player = 1  # Player 1 starts
        self.turn = 0  # Number of turns played
        self.done = False
        return self.sequence.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.sequence.copy(), -10, True, False, {}  # Game is already over

        # Check if opponent lost on their previous move
        if self.turn >= 3:
            last_three = self.sequence[self.turn - 3 : self.turn]
            if np.sum(last_three) % 7 == 0:
                # Opponent (current_player * -1) lost; current player wins
                self.done = True
                return self.sequence.copy(), 1, True, False, {}

        # Map action to number (1 to 9)
        number = action + 1

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move; current player loses
            self.done = True
            return self.sequence.copy(), -10, True, False, {}

        # Append the number to the sequence
        self.sequence[self.turn] = number
        self.turn += 1

        # Check if current player loses after this move
        if self.turn >= 3:
            last_three = self.sequence[self.turn - 3 : self.turn]
            if np.sum(last_three) % 7 == 0:
                # Current player loses
                self.done = True
                return self.sequence.copy(), -10, True, False, {}

        # Check if sequence length limit is reached
        if self.turn >= 10:
            # Current player loses
            self.done = True
            return self.sequence.copy(), -10, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        # The game continues; return negative reward for a valid move
        return self.sequence.copy(), -10, False, False, {}

    def render(self):
        # Return a string representation of the current sequence
        seq_str = (
            "Current sequence: ["
            + ", ".join(str(num) for num in self.sequence[: self.turn])
            + "]"
        )
        return seq_str

    def valid_moves(self):
        # Return a list of valid actions (integers from 0 to 8)
        if self.done:
            return []
        valid_numbers = set(range(1, 10))  # Numbers 1 to 9 inclusive
        if self.turn > 0:
            last_number = self.sequence[self.turn - 1]
            valid_numbers.discard(last_number)
        valid_actions = [n - 1 for n in valid_numbers]  # Adjust for action mapping
        return valid_actions
