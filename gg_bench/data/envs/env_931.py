import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, forbidden_sequence=[1, 2, 3], max_sequence_length=20):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 possible actions (digits 1-9)
        self.action_space = spaces.Discrete(9)
        # Observation space is the shared sequence up to a maximum length
        self.max_sequence_length = max_sequence_length
        self.observation_space = spaces.Box(
            low=0,
            high=9,
            shape=(self.max_sequence_length,),
            dtype=np.int32,
        )

        # Initialize the forbidden sequence
        self.forbidden_sequence = forbidden_sequence

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with an empty shared sequence
        self.shared_sequence = []
        # Player 1 starts first
        self.current_player = 1
        # Game is not done
        self.done = False
        # Prepare the observation
        observation = self._get_observation()
        return observation, {}  # observation, info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}  # Game already over

        # Convert action (0-8) to digit (1-9)
        digit = action + 1

        # Add digit to shared sequence
        self.shared_sequence.append(digit)

        # Check for forbidden sequence
        if self._check_forbidden_sequence():
            self.done = True
            reward = -10  # Current player loses
            return self._get_observation(), reward, True, False, {}
        else:
            # Valid move, switch player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = -10  # Penalty for valid move
            return self._get_observation(), reward, False, False, {}

    def render(self):
        sequence_str = " ".join(map(str, self.shared_sequence))
        return f"Current sequence: {sequence_str}"

    def valid_moves(self):
        # All digits from 1 to 9 are always valid
        return list(range(9))  # Actions are 0-8 corresponding to digits 1-9

    def _check_forbidden_sequence(self):
        seq_length = len(self.shared_sequence)
        forbidden_length = len(self.forbidden_sequence)
        # Only check if there are enough elements
        if seq_length >= forbidden_length:
            # Check for forbidden sequence at all possible positions
            for i in range(seq_length - forbidden_length + 1):
                if (
                    self.shared_sequence[i : i + forbidden_length]
                    == self.forbidden_sequence
                ):
                    return True  # Forbidden sequence found
        return False

    def _get_observation(self):
        # Create an observation array of fixed size
        observation = np.zeros(self.max_sequence_length, dtype=np.int32)
        # Copy the shared sequence into the observation
        sequence_length = len(self.shared_sequence)
        if sequence_length > 0:
            observation[:sequence_length] = self.shared_sequence
        return observation
