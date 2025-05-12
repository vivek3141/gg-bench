import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Multiplicity: A Two-Player Turn-Based Game environment.
    Action space:
        - spaces.Discrete(50): Actions 0 to 49 correspond to choosing numbers 1 to 50.
    Observation space:
        - spaces.Box(low=low_obs, high=high_obs, dtype=np.int32):
            - First 50 entries: 0 (unchosen) or 1 (chosen).
            - Last entry: Current N (from 1 to 50).
    Rewards:
        - +1 if the current player wins.
        - -10 if the current player plays an invalid move.
        - 0 otherwise.
    The environment internally manages player turns for self-play reinforcement learning.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        self.action_space = spaces.Discrete(50)  # Actions correspond to numbers 1 to 50

        # Define observation space
        # Observation consists of:
        # - First 50 elements: 0 (unchosen) or 1 (chosen)
        # - Last element: current N (from 1 to 50)
        low_obs = np.zeros(51, dtype=np.int32)
        high_obs = np.ones(51, dtype=np.int32)
        high_obs[:50] = 1  # Chosen numbers indicators
        high_obs[50] = 50  # Current N can be from 1 to 50
        low_obs[50] = 1  # Current N starts from 1
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize game state
        self.chosen_numbers = np.zeros(50, dtype=np.int32)  # 0: unchosen, 1: chosen
        self.current_N = 1  # Starting number N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Observation, info

    def _get_obs(self):
        # Combine chosen numbers and current N into observation
        obs = np.zeros(51, dtype=np.int32)
        obs[:50] = self.chosen_numbers
        obs[50] = self.current_N
        return obs

    def _get_valid_moves(self):
        # Compute valid moves based on the current state
        valid_moves = []
        for i in range(50):
            if self.chosen_numbers[i] == 0:
                number = i + 1
                if number == self.current_N:
                    continue
                is_proper_multiple = (
                    number > self.current_N and number % self.current_N == 0
                )
                is_proper_divisor = (
                    number < self.current_N and self.current_N % number == 0
                )
                if is_proper_multiple or is_proper_divisor:
                    valid_moves.append(i)
        return valid_moves

    def valid_moves(self):
        # Public method to get valid moves
        return self._get_valid_moves()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action to chosen number
        chosen_number = action + 1  # Action 0 corresponds to number 1

        # Check if the number has already been chosen
        if self.chosen_numbers[action] == 1:
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"invalid_move": "Number already chosen"},
            )

        # Check if the chosen number is equal to current N
        if chosen_number == self.current_N:
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"invalid_move": "Chosen number equals current N"},
            )

        # Check if the chosen number is a proper multiple or proper divisor of current N
        is_proper_multiple = (
            chosen_number > self.current_N and chosen_number % self.current_N == 0
        )
        is_proper_divisor = (
            chosen_number < self.current_N and self.current_N % chosen_number == 0
        )

        if not (is_proper_multiple or is_proper_divisor):
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"invalid_move": "Number is not a proper multiple or proper divisor"},
            )

        # Valid move, update the state
        self.chosen_numbers[action] = 1
        self.current_N = chosen_number

        # Check if opponent has any valid moves
        valid_moves_opponent = self._get_valid_moves()
        if not valid_moves_opponent:
            # Opponent cannot move; current player wins
            self.done = True
            return (
                self._get_obs(),
                1,
                True,
                False,
                {"winner": self.current_player},
            )
        else:
            # Switch to the next player
            self.current_player = 2 if self.current_player == 1 else 1
            return self._get_obs(), 0, False, False, {}

    def render(self):
        output = f"Current N: {self.current_N}\n"
        output += "Chosen Numbers: "
        chosen_numbers_list = [
            str(i + 1) for i in range(50) if self.chosen_numbers[i] == 1
        ]
        output += ", ".join(chosen_numbers_list) + "\n"
        output += f"Current Player: Player {self.current_player}\n"
        return output
