import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers 1 through 20 are actions 0 through 19
        self.action_space = spaces.Discrete(20)

        # Observation space:
        # First 20 elements: 0 if number is available, 1 if used
        # 21st element: last number used (0 if none)
        # 22nd element: current player (1 or 2)
        self.observation_space = spaces.Box(
            low=np.array([0] * 20 + [0] + [1]),
            high=np.array([1] * 20 + [20] + [2]),
            shape=(22,),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers are initially available
        self.used_numbers = np.zeros(20, dtype=np.int32)
        # Sequence is empty, so last number is 0
        self.last_number = 0
        # Player 1 starts
        self.current_player = 1
        self.done = False
        # Build initial observation
        observation = np.concatenate(
            (self.used_numbers, np.array([self.last_number, self.current_player]))
        )
        return observation, {}

    def step(self, action):
        if self.done:
            # Game is over
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )

        number = action + 1  # Map action to number (0-19 to 1-20)

        # Check if the action is valid
        if not self._is_valid_action(number):
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )

        # Action is valid, update the game state
        self.used_numbers[action] = 1
        self.last_number = number

        # Check if the opponent has any valid moves
        self.current_player = 3 - self.current_player  # Switch player
        opponent_valid_moves = self.valid_moves()

        if not opponent_valid_moves:
            # Opponent cannot make a move, current player wins
            self.done = True
            reward = 1
            # Switch back to the winning player in observation
            self.current_player = 3 - self.current_player
            return (
                self._get_observation(),
                reward,
                True,
                False,
                {},
            )

        # Game continues
        reward = 0
        return (
            self._get_observation(),
            reward,
            False,
            False,
            {},
        )

    def _is_valid_action(self, number):
        action = number - 1
        if action < 0 or action >= 20:
            return False  # Invalid action
        if self.used_numbers[action] == 1:
            return False  # Number has already been used

        if self.last_number == 0:
            # First move, any available number is valid
            return True

        # Check if the number is a valid factor or multiple of last_number
        if number % self.last_number == 0 or self.last_number % number == 0:
            if number == 1 and self._has_other_factors():
                # Exclude 1 unless no other options are available
                return False
            return True

        return False

    def _has_other_factors(self):
        # Check if there are other unused factors excluding 1
        factors = []
        for i in range(2, self.last_number + 1):
            if (
                self.last_number % i == 0
                and self.used_numbers[i - 1] == 0
                and i != self.last_number
            ):
                factors.append(i)
        return bool(factors)

    def valid_moves(self):
        valid_actions = []

        for action in range(20):
            if self.used_numbers[action] == 0:
                number = action + 1
                if self.last_number == 0:
                    # First move
                    valid_actions.append(action)
                else:
                    if number % self.last_number == 0 or self.last_number % number == 0:
                        if number == 1 and self._has_other_factors():
                            continue  # Exclude 1 unless no other options
                        valid_actions.append(action)

        return valid_actions

    def render(self):
        used_numbers_list = [i + 1 for i in range(20) if self.used_numbers[i] == 1]
        available_numbers_list = [i + 1 for i in range(20) if self.used_numbers[i] == 0]
        sequence = [i + 1 for i in range(20) if self.used_numbers[i] == 1]
        state_str = f"Available Numbers: {available_numbers_list}\n"
        state_str += f"Used Numbers: {used_numbers_list}\n"
        state_str += f"Sequence: {sequence}\n"
        state_str += f"Last Number: {self.last_number}\n"
        state_str += f"Current Player: Player {self.current_player}\n"
        return state_str

    def _get_observation(self):
        # Build the observation array
        observation = np.concatenate(
            (
                self.used_numbers,
                np.array([self.last_number, self.current_player], dtype=np.int32),
            )
        )
        return observation
