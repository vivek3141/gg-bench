import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 49 possible actions (numbers 2 to 50)
        self.action_space = spaces.Discrete(49)
        # Observation space:
        # - First 49 elements: Availability of numbers 2 to 50 (1.0 if available, 0.0 if not)
        # - Last element: Current number selected by the previous player, normalized to [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(50,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool: 1.0 means available, 0.0 means selected
        self.number_pool = np.ones(49, dtype=np.float32)
        self.current_number = None  # No current number yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Return initial observation and info
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # If the game is over, no further moves are allowed
            return self._get_observation(), 0, True, False, {}

        # Map action to the selected number (2 to 50)
        selected_number = action + 2

        # Check if the selected number is available
        if self.number_pool[action] != 1.0:
            # Invalid move: number already selected
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if the move is valid
        if self.current_number is None:
            # First move, any number is valid
            valid_move = True
        else:
            # Move is valid if the selected number is a divisor or multiple of the current number
            if (
                self.current_number % selected_number == 0
                or selected_number % self.current_number == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: update the game state
        self.number_pool[action] = 0.0  # Mark the number as selected
        self.current_number = selected_number

        # Check if the next player has any valid moves
        if not self._has_valid_moves():
            # Next player cannot move; current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch player
        self.current_player *= -1
        return self._get_observation(), 0, False, False, {}

    def render(self):
        numbers_available = [
            str(i + 2) for i in range(49) if self.number_pool[i] == 1.0
        ]
        numbers_selected = [str(i + 2) for i in range(49) if self.number_pool[i] == 0.0]
        current_number_str = (
            str(self.current_number) if self.current_number is not None else "None"
        )
        render_str = ""
        render_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        render_str += f"Current Number: {current_number_str}\n"
        render_str += f"Numbers Available: {', '.join(numbers_available)}\n"
        render_str += f"Numbers Selected: {', '.join(numbers_selected)}\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid actions (indices) for the current player
        valid_actions = []
        for i in range(49):
            if self.number_pool[i] == 1.0:
                candidate_number = i + 2
                if self.current_number is None:
                    # First move, any number is valid
                    valid_actions.append(i)
                else:
                    if (
                        self.current_number % candidate_number == 0
                        or candidate_number % self.current_number == 0
                    ):
                        valid_actions.append(i)
        return valid_actions

    def _get_observation(self):
        obs = np.zeros(50, dtype=np.float32)
        obs[:49] = self.number_pool  # Numbers availability
        # Normalize current_number to [0,1]
        obs[49] = self.current_number / 50.0 if self.current_number is not None else 0.0
        return obs

    def _has_valid_moves(self):
        # Check if the next player has any valid moves
        for i in range(49):
            if self.number_pool[i] == 1.0:
                candidate_number = i + 2
                if (
                    self.current_number % candidate_number == 0
                    or candidate_number % self.current_number == 0
                ):
                    return True
        return False
