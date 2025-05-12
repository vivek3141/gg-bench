import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space corresponds to possible factors from 2 to 50 (inclusive)
        # Mapped to indices from 0 to 48
        self.action_space = spaces.Discrete(
            49
        )  # Actions 0 to 48 correspond to factors 2 to 50

        # The observation space is the current number, an integer between 1 and 50
        # Using Box with shape (1,) to represent the current number as an array
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([50]), dtype=np.int32
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "starting_number" in options:
            starting_number = options["starting_number"]
            if starting_number < 10 or starting_number > 50:
                raise ValueError("Starting number must be between 10 and 50 inclusive.")
            self.current_number = starting_number
        else:
            # Random starting number between 10 and 50 inclusive
            self.current_number = np.random.randint(10, 51)
        self.done = False
        self.current_player = 1  # Player indicator for potential future use
        return np.array([self.current_number], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Map action index to selected factor (actions 0-48 correspond to factors 2-50)
        selected_factor = action + 2

        # Get list of proper factors of the current number
        proper_factors = self.get_proper_factors(self.current_number)

        # Check if selected_factor is a valid proper factor
        if selected_factor not in proper_factors:
            # Invalid move: end game with penalty
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Valid move: subtract the selected factor from the current number
            self.current_number -= selected_factor

            # Check for win condition (current number is reduced to 1)
            if self.current_number == 1:
                # Current player wins
                self.done = True
                reward = 1
                return (
                    np.array([self.current_number], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            elif self.current_number > 1:
                # Check if the next player has valid moves
                next_proper_factors = self.get_proper_factors(self.current_number)
                if not next_proper_factors:
                    # Next player cannot move: current player wins
                    self.done = True
                    reward = 1
                    return (
                        np.array([self.current_number], dtype=np.int32),
                        reward,
                        True,
                        False,
                        {},
                    )
                else:
                    # Game continues
                    self.current_player *= -1  # Switch player
                    reward = 0
                    return (
                        np.array([self.current_number], dtype=np.int32),
                        reward,
                        False,
                        False,
                        {},
                    )
            else:
                # Unexpected scenario: current number less than 1
                self.done = True
                reward = 0
                return (
                    np.array([self.current_number], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )

    def render(self):
        # Return a string representation of the current game state
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        # Return a list of valid action indices based on current proper factors
        proper_factors = self.get_proper_factors(self.current_number)
        # Map factors back to action indices (factor - 2)
        valid_actions = [f - 2 for f in proper_factors]
        return valid_actions

    def get_proper_factors(self, n):
        # Return a list of proper factors of n (greater than 1 and less than n)
        factors = [i for i in range(2, n) if n % i == 0]
        return factors
