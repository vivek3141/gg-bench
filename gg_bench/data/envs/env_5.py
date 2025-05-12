import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(20) for numbers 1 to 20 (index 0 corresponds to number 1)
        self.action_space = spaces.Discrete(20)

        # Observation space: Box with shape (21,)
        # Indices 0-19: 1.0 if the number is available, 0.0 if removed
        # Index 20: Last number removed normalized between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(21,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the sequence: all numbers from 1 to 20 are available
        self.sequence = np.ones(20, dtype=np.float32)

        # No number has been removed yet
        self.last_number_removed = 0  # 0 indicates no number removed yet

        # Normalize last number removed for observation
        self.last_number_removed_normalized = 0.0

        # Player 1 starts (can be represented by 1, Player 2 by -1)
        self.current_player = 1

        self.done = False

        # Build the observation
        observation = np.concatenate(
            [self.sequence, [self.last_number_removed_normalized]]
        )

        return observation, {}

    def step(self, action):
        # Map action (0-19) to number (1-20)
        number_to_remove = action + 1

        if self.done:
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"error": "Game is already over."},
            )

        if self.sequence[action] == 0.0:
            # Invalid move: number already removed
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"error": "Invalid move: Number already removed."},
            )

        # Check if the move is valid according to the game rules
        valid_numbers = self.get_valid_numbers()

        if number_to_remove not in valid_numbers:
            # Invalid move according to the game rules
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"error": "Invalid move: Not a valid number to remove."},
            )

        # Remove the number from the sequence
        self.sequence[action] = 0.0

        # Update last number removed
        self.last_number_removed = number_to_remove
        self.last_number_removed_normalized = self.last_number_removed / 20.0

        # Check if the opponent can make a valid move
        self.current_player *= -1  # Switch to the other player

        valid_numbers_next_player = self.get_valid_numbers()

        if not valid_numbers_next_player:
            # Opponent has no valid moves; current player wins
            self.done = True
            return (
                self._get_obs(),
                1,
                True,
                False,
                {"message": "You win! Opponent has no valid moves."},
            )
        else:
            # Game continues
            return self._get_obs(), 0, False, False, {}

    def render(self):
        # Display the current sequence of available numbers
        remaining_numbers = [str(i + 1) for i in range(20) if self.sequence[i] == 1.0]
        sequence_str = "Remaining numbers: " + " ".join(remaining_numbers)
        last_num_str = (
            f"Last number removed: {self.last_number_removed}"
            if self.last_number_removed != 0
            else "No numbers have been removed yet."
        )
        return sequence_str + "\n" + last_num_str

    def valid_moves(self):
        # Return a list of valid actions (indices in the action space)
        valid_numbers = self.get_valid_numbers()
        valid_actions = [n - 1 for n in valid_numbers]  # Convert numbers to actions
        return valid_actions

    def get_valid_numbers(self):
        if self.last_number_removed == 0:
            # First turn: any available number can be removed
            valid_numbers = [i + 1 for i in range(20) if self.sequence[i] == 1.0]
        else:
            # Subsequent turns: numbers that are divisors or multiples of the last removed number
            valid_numbers = []
            for i in range(20):
                if self.sequence[i] == 1.0:
                    number = i + 1
                    if (
                        number % self.last_number_removed == 0
                        or self.last_number_removed % number == 0
                    ):
                        valid_numbers.append(number)
        return valid_numbers

    def _get_obs(self):
        # Build the observation
        observation = np.concatenate(
            [self.sequence, [self.last_number_removed_normalized]]
        )
        return observation
