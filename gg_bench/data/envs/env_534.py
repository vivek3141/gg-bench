import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(9), corresponding to numbers 1 to 9
        self.action_space = spaces.Discrete(9)

        # The observation space consists of:
        # - Number Pool: 9 numbers (1 if available, 0 if taken)
        # - Player 1's Sequence: 5 numbers (0 if not yet chosen)
        # - Player 2's Sequence: 5 numbers (0 if not yet chosen)
        # - Current Player Indicator: 1 for Player 1, -1 for Player 2
        # Total length: 9 (Number Pool) + 5 (Player 1) + 5 (Player 2) + 1 (Current Player) = 20
        self.observation_space = spaces.Box(low=0, high=9, shape=(20,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool with numbers 1 to 9 (1 indicates available)
        self.number_pool = np.ones(9, dtype=np.int32)

        # Initialize player sequences
        self.player1_sequence = np.zeros(5, dtype=np.int32)
        self.player2_sequence = np.zeros(5, dtype=np.int32)

        # Set the starting player (1 for Player 1, -1 for Player 2)
        self.current_player = 1

        # Sequence lengths
        self.player1_seq_length = 0
        self.player2_seq_length = 0

        # Track when each player completes their sequence
        self.player1_seq_completed_turn = None
        self.player2_seq_completed_turn = None

        # Turn counter
        self.turn_counter = 1

        # Game over flag
        self.done = False

        return self._get_obs(), {}

    def _get_obs(self):
        # Concatenate the game state into a single observation array
        obs = np.concatenate(
            [
                self.number_pool,  # Number Pool (9,)
                self.player1_sequence,  # Player 1's Sequence (5,)
                self.player2_sequence,  # Player 2's Sequence (5,)
                np.array([self.current_player]),  # Current Player Indicator (1,)
            ]
        )
        return obs

    def step(self, action):
        if self.done:
            # If the game is over, no further actions are valid
            return self._get_obs(), 0, True, False, {}

        # Map action (0-8) to number (1-9)
        number_chosen = action + 1

        # Check if the chosen number is available in the Number Pool
        if self.number_pool[action] == 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Remove the chosen number from the Number Pool
        self.number_pool[action] = 0

        # Add the number to the current player's sequence
        if self.current_player == 1:
            self.player1_sequence[self.player1_seq_length] = number_chosen
            self.player1_seq_length += 1
            if self.player1_seq_length == 5 and self.player1_seq_completed_turn is None:
                self.player1_seq_completed_turn = self.turn_counter
        else:
            self.player2_sequence[self.player2_seq_length] = number_chosen
            self.player2_seq_length += 1
            if self.player2_seq_length == 5 and self.player2_seq_completed_turn is None:
                self.player2_seq_completed_turn = self.turn_counter

        # Check for a win condition
        winner = self._check_winner()

        if winner == self.current_player:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}
        elif winner != 0:
            # Opponent wins
            self.done = True
            return self._get_obs(), 0, True, False, {}
        else:
            # Increment turn counter
            self.turn_counter += 1

            # Check if all numbers have been taken
            if np.sum(self.number_pool) == 0:
                # Determine winner based on valid adjacent pairs
                winner = self._determine_winner()
                self.done = True
                if winner == self.current_player:
                    return self._get_obs(), 1, True, False, {}
                else:
                    return self._get_obs(), 0, True, False, {}
            else:
                # Switch to the other player
                self.current_player *= -1
                return self._get_obs(), 0, False, False, {}

    def _check_winner(self):
        # Check if the current player has a winning sequence
        if self.current_player == 1:
            sequence = self.player1_sequence[: self.player1_seq_length]
        else:
            sequence = self.player2_sequence[: self.player2_seq_length]

        if len(sequence) == 5:
            # Check if all adjacent pairs sum to even numbers
            all_even_sums = all(
                (sequence[i] + sequence[i + 1]) % 2 == 0 for i in range(4)
            )
            if all_even_sums:
                return self.current_player  # Current player wins

        return 0  # No winner yet

    def _determine_winner(self):
        # Determine winner based on the number of valid adjacent pairs
        player1_pairs = self._count_even_adjacent_pairs(self.player1_sequence)
        player2_pairs = self._count_even_adjacent_pairs(self.player2_sequence)

        if player1_pairs > player2_pairs:
            return 1
        elif player2_pairs > player1_pairs:
            return -1
        else:
            # Tie-breaker: Player who completed their sequence first wins
            if self.player1_seq_completed_turn < self.player2_seq_completed_turn:
                return 1
            else:
                return -1  # If equal, Player 2 wins by default

    def _count_even_adjacent_pairs(self, sequence):
        # Count the number of adjacent pairs that sum to an even number
        valid_pairs = 0
        for i in range(len(sequence) - 1):
            if sequence[i] == 0 or sequence[i + 1] == 0:
                break  # Sequence not fully built yet
            if (sequence[i] + sequence[i + 1]) % 2 == 0:
                valid_pairs += 1
        return valid_pairs

    def render(self):
        # Return a string representation of the current game state
        number_pool = [
            i + 1 for i, available in enumerate(self.number_pool) if available == 1
        ]
        player1_seq = self.player1_sequence[: self.player1_seq_length].tolist()
        player2_seq = self.player2_sequence[: self.player2_seq_length].tolist()
        current_player = "Player 1" if self.current_player == 1 else "Player 2"

        return (
            f"Current Number Pool: {number_pool}\n"
            f"Player 1's Sequence: {player1_seq}\n"
            f"Player 2's Sequence: {player2_seq}\n"
            f"Current Player: {current_player}\n"
        )

    def valid_moves(self):
        # Return a list of valid action indices (numbers still available)
        return [i for i in range(9) if self.number_pool[i] == 1]
