import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define maximum number for the game
        self.max_number = 100

        # Define action and observation space
        # The action is a subtraction of a positive divisor up to max_number
        self.action_space = spaces.Discrete(
            self.max_number + 1
        )  # Actions from 0 to max_number

        # The observation includes:
        # - Current Number (N): integer from 0 to max_number
        # - Parity Indicator: integer in {-1, 0, 1}
        #   - -1: Odd divisor required
        #   - 0: No parity restriction
        #   - 1: Even divisor required
        self.observation_space = spaces.Box(
            low=np.array([0, -1]),
            high=np.array([self.max_number, 1]),
            dtype=np.int32,
        )

        # Initialize the game state
        self.current_number = None
        self.parity_required = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting number N can be set to 15 or any other positive integer up to max_number
        self.current_number = 15
        self.parity_required = 0  # No parity restriction on the first turn
        self.done = False
        observation = np.array(
            [self.current_number, self.parity_required], dtype=np.int32
        )
        info = {}
        return observation, info

    def step(self, action):
        reward = 0
        info = {}

        # Check if the game has already ended
        if self.done:
            return (
                np.array([self.current_number, self.parity_required], dtype=np.int32),
                reward,
                self.done,
                False,
                info,
            )

        # Check if action is within the valid range
        if action <= 0 or action > self.max_number:
            # Invalid action: not a positive integer within the valid range
            reward = -10
            self.done = True
            return (
                np.array([self.current_number, self.parity_required], dtype=np.int32),
                reward,
                self.done,
                False,
                info,
            )

        # Check if action is a valid move
        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move: action not in valid moves
            reward = -10
            self.done = True
            return (
                np.array([self.current_number, self.parity_required], dtype=np.int32),
                reward,
                self.done,
                False,
                info,
            )

        # Apply the action
        self.current_number -= action

        # Check if the game is won
        if self.current_number == 0:
            reward = 1
            self.done = True
            observation = np.array(
                [self.current_number, self.parity_required], dtype=np.int32
            )
            return observation, reward, self.done, False, info

        # Update parity_required for the next move
        if action % 2 == 0:
            # Subtracted an even number, next required is odd
            self.parity_required = -1
        else:
            # Subtracted an odd number, next required is even
            self.parity_required = 1

        # Check if there are any valid moves left
        if len(self.valid_moves()) == 0:
            # No valid moves left for the next turn, current player wins
            reward = 1
            self.done = True
            observation = np.array(
                [self.current_number, self.parity_required], dtype=np.int32
            )
            return observation, reward, self.done, False, info

        # Continue the game
        observation = np.array(
            [self.current_number, self.parity_required], dtype=np.int32
        )
        return observation, reward, self.done, False, info

    def render(self):
        parity_str = (
            "No parity restriction"
            if self.parity_required == 0
            else (
                "Even divisor required"
                if self.parity_required == 1
                else "Odd divisor required"
            )
        )
        render_str = (
            f"Current Number: {self.current_number}\nParity Requirement: {parity_str}"
        )
        return render_str

    def valid_moves(self):
        # Finds all positive integer divisors of current_number matching the required parity
        if self.current_number <= 0:
            return []

        divisors = [
            i for i in range(1, self.current_number + 1) if self.current_number % i == 0
        ]

        if self.parity_required == 0:
            # No parity restriction
            valid_moves = divisors
        elif self.parity_required == 1:
            # Even divisors required
            valid_moves = [d for d in divisors if d % 2 == 0]
        elif self.parity_required == -1:
            # Odd divisors required
            valid_moves = [d for d in divisors if d % 2 == 1]
        else:
            # Invalid parity requirement
            valid_moves = []

        return valid_moves
