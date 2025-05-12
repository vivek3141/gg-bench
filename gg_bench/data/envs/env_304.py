import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Odd-Even Elimination Environment for Gymnasium
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(20): actions 0 to 19 correspond to numbers 1 to 20
        self.action_space = spaces.Discrete(20)

        # The observation space is a Box of shape (21,)
        # First 20 elements represent the availability of numbers 1-20 (1=available, 0=removed)
        # The 21st element represents the parity required for the next move:
        # 0 = even, 1 = odd, 2 = no requirement (at the start of the game)
        self.observation_space = spaces.Box(low=0, high=2, shape=(21,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False

        # Initialize the observation
        self.observation = np.ones(21, dtype=np.int8)

        # All numbers from 1 to 20 are available at the start
        self.observation[0:20] = 1

        # No parity requirement at the start
        self.observation[20] = 2  # 2 represents 'no requirement'

        return self.observation, {}  # Return observation and info as per Gym API

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}  # Game is already over

        number = action + 1  # Actions 0-19 correspond to numbers 1-20

        # Check if the selected number is available
        if self.observation[number - 1] == 0:
            # Number already removed; invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        parity_required = self.observation[20]

        # Check if the selected number meets the parity requirement
        if parity_required == 2:
            # No parity requirement; any number is acceptable
            valid_parity = True
        else:
            valid_parity = (number % 2) == parity_required

        if not valid_parity:
            # Parity does not match; invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        # Valid move; update the game state
        self.observation[number - 1] = 0  # Remove the selected number

        # Update parity requirement for the next move
        parity_required_next = (number % 2) ^ 1  # Switch parity: even <-> odd
        self.observation[20] = parity_required_next

        # Check if there are valid moves left for the next player
        available_numbers = (
            np.where(self.observation[0:20] == 1)[0] + 1
        )  # Numbers still available

        # Check for available numbers that meet the required parity
        valid_moves_exist = any(
            (num % 2) == parity_required_next for num in available_numbers
        )

        if not valid_moves_exist:
            # No valid moves for the next player; current player wins
            self.done = True
            return self.observation, 1, True, False, {}
        else:
            # Game continues to the next player
            return self.observation, 0, False, False, {}

    def render(self):
        # Generate a string representation of the game state
        available_numbers = (
            np.where(self.observation[0:20] == 1)[0] + 1
        )  # Numbers still available
        parity_required = self.observation[20]

        if parity_required == 2:
            parity_info = "No parity requirement. Any number can be selected."
        elif parity_required == 0:
            parity_info = "Next move must be an EVEN number."
        elif parity_required == 1:
            parity_info = "Next move must be an ODD number."
        else:
            parity_info = "Unknown parity requirement."

        game_state = (
            f"Available Numbers: {' '.join(map(str, available_numbers))}\n{parity_info}"
        )
        return game_state

    def valid_moves(self):
        parity_required = self.observation[20]
        valid_actions = []
        for action in range(20):
            number = action + 1
            if self.observation[number - 1] == 1:
                if parity_required == 2:
                    # No parity requirement; any available number is valid
                    valid_actions.append(action)
                else:
                    if (number % 2) == parity_required:
                        # Number meets the parity requirement
                        valid_actions.append(action)
        return valid_actions
