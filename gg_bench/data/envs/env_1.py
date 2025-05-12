import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space:
        # Actions 0-4: Flip bit at positions 1-5
        # Actions 5-36: Guess opponent's sequence (32 possible sequences)
        self.action_space = spaces.Discrete(37)

        # Define observation space:
        # Observation consists of:
        # - Player 1 public sequence (5 bits)
        # - Player 2 public sequence (5 bits)
        # - Last Match Count received (1 value)
        # - Current player indicator (1 or -1)
        self.observation_space = spaces.Box(low=0, high=5, shape=(12,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate random secret sequences for both players
        self.player1_secret = np.random.randint(0, 2, size=5, dtype=np.int8)
        self.player2_secret = np.random.randint(0, 2, size=5, dtype=np.int8)

        # Public sequences start as secret sequences
        self.player1_public = self.player1_secret.copy()
        self.player2_public = self.player2_secret.copy()

        # Initialize current player (1 or -1)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Initialize last Match Count received
        self.last_match_count = 0

        # Initialize last action taken
        self.last_action = -1  # No action taken yet

        # Prepare initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        if action < 0 or action >= 37:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        reward = -10  # Penalty for each valid move
        info = {}

        # Process action
        if action <= 4:
            # Flip action
            flip_position = action  # Positions 0-4 correspond to bits 1-5
            if self.current_player == 1:
                self.player1_public[flip_position] ^= 1  # Flip the bit
                # Compute Match Count
                match_count = np.sum(self.player1_public == self.player2_secret)
            else:
                self.player2_public[flip_position] ^= 1  # Flip the bit
                match_count = np.sum(self.player2_public == self.player1_secret)
            self.last_match_count = match_count
            done = False
        else:
            # Guess action
            guess_index = action - 5  # Map to 0-31
            guessed_sequence = self._index_to_sequence(guess_index)
            if self.current_player == 1:
                # Player 1 is guessing Player 2's secret sequence
                if np.array_equal(guessed_sequence, self.player2_secret):
                    # Correct guess
                    reward = 1
                    self.done = True
                    done = True
                else:
                    self.last_match_count = 0  # No Match Count on incorrect guess
                    done = False
            else:
                # Player 2 is guessing Player 1's secret sequence
                if np.array_equal(guessed_sequence, self.player1_secret):
                    reward = 1
                    self.done = True
                    done = True
                else:
                    self.last_match_count = 0
                    done = False

        # Update last action
        self.last_action = action

        # Switch current player
        self.current_player *= -1

        observation = self._get_observation()
        return observation, reward, self.done, False, info

    def render(self):
        current_player_str = "Player 1" if self.current_player == 1 else "Player 2"
        output = "Current Player: {}\n".format(current_player_str)
        output += "Player 1 Public Sequence: {}\n".format(
            " ".join(map(str, self.player1_public))
        )
        output += "Player 2 Public Sequence: {}\n".format(
            " ".join(map(str, self.player2_public))
        )
        output += "Last Match Count: {}\n".format(self.last_match_count)
        output += "Last Action Taken: {}\n".format(self.last_action)
        return output

    def valid_moves(self):
        # All actions from 0 to 36 are valid
        return list(range(37))

    def _get_observation(self):
        # Observation consists of:
        # - Player 1 public sequence (5 bits)
        # - Player 2 public sequence (5 bits)
        # - Last Match Count received (1 value)
        # - Current player indicator (1 or -1)
        observation = np.concatenate(
            (
                self.player1_public,
                self.player2_public,
                np.array([self.last_match_count], dtype=np.int8),
                np.array([self.current_player], dtype=np.int8),
            )
        )
        return observation

    def _index_to_sequence(self, index):
        # Converts an index from 0 to 31 to a 5-bit binary sequence
        return np.array([(index >> i) & 1 for i in reversed(range(5))], dtype=np.int8)
