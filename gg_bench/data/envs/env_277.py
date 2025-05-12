import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            10
        )  # Actions correspond to indices 0-9 for numbers 2-20
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )  # 10 for available numbers, 1 for last number removed

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            10, dtype=np.float32
        )  # Numbers 2-20 are available
        self.last_number_removed = -1  # No number removed yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )  # Game already over

        # Map action index to actual number
        number = 2 + action * 2  # Numbers from 2 to 20, even numbers
        number_index = action  # Indices from 0 to 9

        # Check if the number is available
        if self.available_numbers[number_index] == 0:
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid move: number not available

        # Check if the move is valid according to the game rules
        if self.last_number_removed == -1:
            # First move, any number can be removed
            valid_move = True
        else:
            # Subsequent moves
            if (
                number % self.last_number_removed == 0
                or self.last_number_removed % number == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid move: does not satisfy divisor/multiple condition

        # Valid move: update the game state
        self.available_numbers[number_index] = 0  # Remove the number
        self.last_number_removed = number

        # Check if the opponent has any valid moves
        if not self._has_valid_moves():
            # Opponent cannot move; current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}  # Reward, done, info

        # Switch to the next player (since the agent plays both players, we continue)
        self.current_player *= -1

        return self._get_observation(), -10, False, False, {}  # Reward per valid move

    def render(self):
        # Create a string representation of the game state
        numbers = [2 + i * 2 for i in range(10)]
        available_numbers = [
            str(numbers[i]) if self.available_numbers[i] == 1 else "X"
            for i in range(10)
        ]
        available_numbers_str = ", ".join(available_numbers)
        last_removed_str = (
            str(self.last_number_removed) if self.last_number_removed != -1 else "None"
        )
        player_str = f"Player {1 if self.current_player == 1 else 2}"

        state_str = (
            f"Available Numbers: {available_numbers_str}\n"
            f"Last Number Removed: {last_removed_str}\n"
            f"Current Turn: {player_str}\n"
        )
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        numbers = [2 + i * 2 for i in range(10)]
        for i in range(10):
            if self.available_numbers[i] == 1:
                number = numbers[i]
                if self.last_number_removed == -1:
                    # First move, any available number is valid
                    valid_actions.append(i)
                else:
                    if (
                        number % self.last_number_removed == 0
                        or self.last_number_removed % number == 0
                    ):
                        valid_actions.append(i)
        return valid_actions

    def _get_observation(self):
        # Create the observation array
        observation = np.zeros(11, dtype=np.float32)
        observation[:10] = self.available_numbers  # Available numbers
        # Last number removed (normalized between 0 and 1)
        observation[10] = (
            (self.last_number_removed - 2) / 18
            if self.last_number_removed != -1
            else -1
        )
        return observation

    def _has_valid_moves(self):
        # Check if the next player has any valid moves
        numbers = [2 + i * 2 for i in range(10)]
        for i in range(10):
            if self.available_numbers[i] == 1:
                number = numbers[i]
                if (
                    number % self.last_number_removed == 0
                    or self.last_number_removed % number == 0
                ):
                    return True  # Valid move is available
        return False  # No valid moves left
