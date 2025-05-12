import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        # Observation space: shared_pool (9 elements) + player1_seq (9 elements) + player2_seq (9 elements)
        self.observation_space = spaces.Box(low=0, high=9, shape=(27,), dtype=np.int8)

        # Initialize the state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize shared pool: numbers 1 to 9 are available
        self.shared_pool = np.ones(
            9, dtype=np.int8
        )  # Indices 0-8 represent numbers 1-9

        # Initialize player sequences as empty lists
        self.player1_seq = []
        self.player2_seq = []

        # Current player: 1 or -1 (1 for Player 1, -1 for Player 2)
        self.current_player = 1

        self.done = False

        # Construct observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def _get_observation(self):
        # Shared pool: 1 if number is available, 0 if not
        shared_pool_obs = self.shared_pool.copy()  # Length 9

        # Sequences: pad sequences to length 9
        player1_seq_obs = np.zeros(9, dtype=np.int8)
        player2_seq_obs = np.zeros(9, dtype=np.int8)

        player1_seq_length = len(self.player1_seq)
        if player1_seq_length > 0:
            player1_seq_obs[:player1_seq_length] = self.player1_seq

        player2_seq_length = len(self.player2_seq)
        if player2_seq_length > 0:
            player2_seq_obs[:player2_seq_length] = self.player2_seq

        # Concatenate to get observation of length 27
        observation = np.concatenate(
            [shared_pool_obs, player1_seq_obs, player2_seq_obs]
        )

        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number = action + 1  # Map action index to number: 0->1, 1->2, ..., 8->9

        # Check if number is in shared pool
        if self.shared_pool[action] == 0:
            # Invalid move: number not available
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, done, truncated, info

        # Get current player's sequence
        if self.current_player == 1:
            player_seq = self.player1_seq
        else:
            player_seq = self.player2_seq

        # Check if number is valid for current player
        if len(player_seq) == 0:
            valid = True  # Any number is valid if sequence is empty
        else:
            if number > player_seq[-1]:
                valid = True
            else:
                valid = False

        if not valid:
            # Invalid move: number not strictly greater than last number in sequence
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, done, truncated, info

        # Valid move: update the state
        self.shared_pool[action] = 0  # Number is no longer available

        player_seq.append(number)  # Add number to current player's sequence

        # Now check if the next player can make any valid moves
        # Switch current player
        self.current_player *= -1

        # Get next player's sequence
        if self.current_player == 1:
            next_player_seq = self.player1_seq
        else:
            next_player_seq = self.player2_seq

        # Get possible numbers from shared pool
        available_numbers = (
            np.where(self.shared_pool == 1)[0] + 1
        )  # Numbers available in shared pool

        # Determine if next player has any valid moves
        if len(next_player_seq) == 0:
            # If sequence is empty, any available number is valid
            has_valid_moves = len(available_numbers) > 0
        else:
            # Number must be strictly greater than last number in next player's sequence
            has_valid_moves = np.any(available_numbers > next_player_seq[-1])

        if not has_valid_moves:
            # Next player cannot make a valid move. Current player wins.
            self.done = True
            reward = 1  # Reward for winning
            return (
                self._get_observation(),
                reward,
                True,
                False,
                {},
            )  # Observation, reward, done, truncated, info
        else:
            # Game continues
            reward = 0
            return (
                self._get_observation(),
                reward,
                False,
                False,
                {},
            )  # Observation, reward, done, truncated, info

    def render(self):
        output = (
            "Shared Pool: "
            + str([i + 1 for i in range(9) if self.shared_pool[i] == 1])
            + "\n"
        )
        output += "Player 1's Sequence: " + str(self.player1_seq) + "\n"
        output += "Player 2's Sequence: " + str(self.player2_seq) + "\n"
        output += (
            "Current Player: Player "
            + ("1" if self.current_player == 1 else "2")
            + "\n"
        )
        return output

    def valid_moves(self):
        # Returns list of valid action indices for the current player
        # Get current player's sequence
        if self.current_player == 1:
            player_seq = self.player1_seq
        else:
            player_seq = self.player2_seq

        # Get available numbers
        available_indices = np.where(self.shared_pool == 1)[
            0
        ]  # Indices of available numbers
        valid_action_indices = []

        if len(player_seq) == 0:
            # Any available number is valid
            valid_action_indices = list(available_indices)
        else:
            last_number = player_seq[-1]
            for idx in available_indices:
                number = idx + 1  # Map index to number
                if number > last_number:
                    valid_action_indices.append(idx)
        return valid_action_indices
