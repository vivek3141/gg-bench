import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0-8 for adding number 1-9, 9-36 for making guesses
        self.action_space = spaces.Discrete(37)
        # Observations: sequences of both players (length 5 each)
        self.observation_space = spaces.Box(low=0, high=9, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_sequences = [[], []]  # Sequences for player 0 and player 1
        self.sequence_types = [None, None]  # Secret sequence types for both players
        self.sequence_params = [None, None]  # Parameters for the sequence types
        self.current_player = 0  # Player 0 starts
        self.done = False

        # Generate secret sequence types for both players
        self.sequence_types[0], self.sequence_params[0] = self.generate_sequence_type()
        self.sequence_types[1], self.sequence_params[1] = self.generate_sequence_type()
        return self.get_observation(), {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        reward = 0

        # Process action
        if action < 9:
            # Add number to sequence
            number = action + 1
            if len(self.player_sequences[self.current_player]) >= 5:
                # Cannot add more numbers, sequence is already complete
                self.done = True
                return self.get_observation(), -10, True, False, {}
            else:
                # Check if the number is valid for the player's sequence type
                is_valid = self.is_valid_number_for_sequence(
                    number,
                    self.sequence_types[self.current_player],
                    self.sequence_params[self.current_player],
                    self.player_sequences[self.current_player],
                )
                if is_valid:
                    self.player_sequences[self.current_player].append(number)
                    if len(self.player_sequences[self.current_player]) == 5:
                        # Player has completed their sequence and wins
                        self.done = True
                        return self.get_observation(), 1, True, False, {}
                else:
                    # Invalid move
                    self.done = True
                    return self.get_observation(), -10, True, False, {}
        else:
            # Make a guess about opponent's sequence type
            guess_type, guess_param = self.map_action_to_guess(action)
            opponent = 1 - self.current_player
            if (
                guess_type == self.sequence_types[opponent]
                and guess_param == self.sequence_params[opponent]
            ):
                # Correct guess
                self.player_sequences[opponent] = []
                # Opponent selects a new sequence type (cannot be the same as before)
                excluded_sequence = self.sequence_types[opponent]
                self.sequence_types[opponent], self.sequence_params[opponent] = (
                    self.generate_sequence_type(excluded_sequence)
                )
            else:
                # Incorrect guess, no reward or penalty
                pass  # Do nothing

        # Switch to the other player
        self.current_player = 1 - self.current_player

        return self.get_observation(), 0, False, False, {}

    def render(self):
        sequences = [
            f"Player {i} sequence: {self.player_sequences[i]}" for i in range(2)
        ]
        return "\n".join(sequences)

    def valid_moves(self):
        if len(self.player_sequences[self.current_player]) >= 5:
            # Cannot add more numbers
            valid_numbers = []
        else:
            valid_numbers = list(range(9))  # All numbers from 1 to 9

        valid_actions = valid_numbers + list(range(9, 37))  # Include guess actions
        return valid_actions

    def get_observation(self):
        # Return the sequences as observations
        obs = np.zeros(10, dtype=np.int8)
        # Current player's sequence in positions 0-4
        seq = self.player_sequences[self.current_player]
        obs[0 : len(seq)] = seq
        # Opponent's sequence in positions 5-9
        opp_seq = self.player_sequences[1 - self.current_player]
        obs[5 : 5 + len(opp_seq)] = opp_seq
        return obs

    def generate_sequence_type(self, excluded_type=None):
        sequence_types = [
            "arithmetic",
            "geometric",
            "fibonacci",
            "prime",
            "square",
            "cube",
            "multiple",
            "even",
            "odd",
        ]
        if excluded_type is not None:
            sequence_types.remove(excluded_type)

        seq_type = random.choice(sequence_types)
        param = None
        if seq_type == "arithmetic":
            # Common difference D (excluding 0)
            D_options = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
            param = random.choice(D_options)
        elif seq_type == "geometric":
            # Common ratio R (integers from 2 to 5)
            R_options = [2, 3, 4, 5]
            param = random.choice(R_options)
        elif seq_type == "multiple":
            # N from 2 to 9
            N_options = [2, 3, 4, 5, 6, 7, 8, 9]
            param = random.choice(N_options)
        else:
            param = None  # No parameter needed
        return seq_type, param

    def map_action_to_guess(self, action):
        if 9 <= action <= 18:
            # Arithmetic sequence guesses
            D_options = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
            idx = action - 9
            D = D_options[idx]
            return "arithmetic", D
        elif 19 <= action <= 22:
            # Geometric sequence guesses
            R_options = [2, 3, 4, 5]
            idx = action - 19
            R = R_options[idx]
            return "geometric", R
        elif 23 <= action <= 30:
            # Multiples of N guesses
            N_options = [2, 3, 4, 5, 6, 7, 8, 9]
            idx = action - 23
            N = N_options[idx]
            return "multiple", N
        elif 31 <= action <= 36:
            # Other sequence types without parameters
            types = [
                "fibonacci",
                "prime",
                "square",
                "cube",
                "even",
                "odd",
            ]
            idx = action - 31
            return types[idx], None
        else:
            raise ValueError("Invalid action for guessing.")

    def is_valid_number_for_sequence(self, number, seq_type, param, sequence):
        if seq_type == "arithmetic":
            D = param
            if not sequence:
                # Any starting number is valid
                return True
            else:
                # Check if the number continues the arithmetic sequence
                expected = sequence[-1] + D
                return number == expected
        elif seq_type == "geometric":
            R = param
            if not sequence:
                # Any starting number is valid (non-zero)
                return number != 0
            else:
                # Check if the number continues the geometric sequence
                expected = sequence[-1] * R
                return number == expected
        elif seq_type == "fibonacci":
            if len(sequence) < 2:
                # Any number is valid for the first two numbers
                return True
            else:
                # Check if the number is the sum of the previous two
                expected = sequence[-1] + sequence[-2]
                return number == expected
        elif seq_type == "prime":
            return self.is_prime(number)
        elif seq_type == "square":
            return self.is_square(number)
        elif seq_type == "cube":
            return self.is_cube(number)
        elif seq_type == "multiple":
            N = param
            return number % N == 0
        elif seq_type == "even":
            return number % 2 == 0
        elif seq_type == "odd":
            return number % 2 == 1
        else:
            return False

    def is_prime(self, n):
        if n <= 1:
            return False
        elif n <= 3:
            return True
        elif n % 2 == 0:
            return False
        else:
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True

    def is_square(self, n):
        return int(n**0.5) ** 2 == n

    def is_cube(self, n):
        return int(round(n ** (1 / 3))) ** 3 == n
