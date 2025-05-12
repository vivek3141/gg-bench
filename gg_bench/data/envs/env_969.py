import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # The numbers 1 through 9 correspond to action indices 0 through 8
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # - Shared pool availability: 9 elements (1 if available, 0 if not)
        # - Player 1 sequence: 9 elements (numbers picked, 0 if empty)
        # - Player 2 sequence: 9 elements (numbers picked, 0 if empty)
        # - Player 1 sum: 1 element
        # - Player 2 sum: 1 element
        # - Current player: 1 element (1 or 2)
        # Total: 9 + 9 + 9 + 1 + 1 + 1 = 30
        self.observation_space = spaces.Box(low=0, high=15, shape=(30,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Shared pool: True if the number is available, False if taken
        self.shared_pool = [True] * 9  # Indices 0-8 correspond to numbers 1-9

        # Player sequences: lists of numbers picked by each player
        self.player_sequences = {1: [], 2: []}

        # Player sums: total sum of each player's sequence
        self.player_sums = {1: 0, 2: 0}

        # Current player: 1 or 2
        self.current_player = 1

        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number = action + 1  # Map action index to number (0->1, ..., 8->9)

        # Check if the number is available in the shared pool
        if not self.shared_pool[action]:
            # Invalid move: number not available
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Get the player's last picked number
        player_sequence = self.player_sequences[self.current_player]
        if player_sequence:
            last_number = player_sequence[-1]
            if number <= last_number:
                # Invalid move: number not greater than last number in sequence
                self.done = True
                return self._get_observation(), -10, True, False, {}
        else:
            last_number = 0  # No previous number, allow any number

        # Check if adding the number would exceed a sum of 15
        player_sum = self.player_sums[self.current_player]
        if player_sum + number > 15:
            # Invalid move: sum would exceed 15
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: update the game state
        self.shared_pool[action] = False  # Remove number from shared pool
        player_sequence.append(number)
        self.player_sums[self.current_player] += number

        # Check for a win
        if self.player_sums[self.current_player] == 15:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player can make any valid moves
        if not self._has_valid_moves():
            # Next player cannot move, current player wins
            self.done = True
            # Since the current player wins due to opponent's inability to move
            return self._get_observation(), 1, True, False, {}

        # Game continues
        return self._get_observation(), 0, False, False, {}

    def render(self):
        shared_pool_numbers = [
            str(i + 1) for i, available in enumerate(self.shared_pool) if available
        ]
        shared_pool_str = "Shared Pool: [" + " ".join(shared_pool_numbers) + "]\n"

        player1_sequence_str = (
            "Player 1 Sequence: " + " ".join(map(str, self.player_sequences[1])) + "\n"
        )
        player1_sum_str = "Player 1 Sum: " + str(self.player_sums[1]) + "\n"

        player2_sequence_str = (
            "Player 2 Sequence: " + " ".join(map(str, self.player_sequences[2])) + "\n"
        )
        player2_sum_str = "Player 2 Sum: " + str(self.player_sums[2]) + "\n"

        current_player_str = f"Current Player: Player {self.current_player}\n"

        return (
            shared_pool_str
            + player1_sequence_str
            + player1_sum_str
            + player2_sequence_str
            + player2_sum_str
            + current_player_str
        )

    def valid_moves(self):
        valid_actions = []
        player_sequence = self.player_sequences[self.current_player]
        player_sum = self.player_sums[self.current_player]
        # Get the last number in the player's sequence
        if player_sequence:
            last_number = player_sequence[-1]
        else:
            last_number = 0  # No previous number, allow any number

        for i in range(9):
            if self.shared_pool[i]:
                number = i + 1
                if number > last_number and player_sum + number <= 15:
                    valid_actions.append(i)
        return valid_actions

    def _has_valid_moves(self):
        # Checks if the current player has any valid moves
        return len(self.valid_moves()) > 0

    def _get_observation(self):
        # Construct the observation array
        shared_pool_availability = np.array(
            [int(avail) for avail in self.shared_pool], dtype=np.int32
        )

        # Player sequences padded to length 9 with zeros
        player1_sequence_padded = np.zeros(9, dtype=np.int32)
        player1_sequence_padded[: len(self.player_sequences[1])] = (
            self.player_sequences[1]
        )

        player2_sequence_padded = np.zeros(9, dtype=np.int32)
        player2_sequence_padded[: len(self.player_sequences[2])] = (
            self.player_sequences[2]
        )

        # Player sums
        player1_sum = np.array([self.player_sums[1]], dtype=np.int32)
        player2_sum = np.array([self.player_sums[2]], dtype=np.int32)

        # Current player indicator
        current_player_indicator = np.array([self.current_player], dtype=np.int32)

        observation = np.concatenate(
            [
                shared_pool_availability,
                player1_sequence_padded,
                player2_sequence_padded,
                player1_sum,
                player2_sum,
                current_player_indicator,
            ]
        )

        return observation
