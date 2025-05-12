import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N  # Size of the Number Set (default is 10)
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.N + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.used_numbers = set()
        self.current_number = None
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Initialize observation
        # observation[0:N]: 0 if number is available, 1 if used
        # observation[N]: -1 indicates no current number at the start
        self.observation = np.zeros(self.N + 1, dtype=np.float32)
        self.observation[self.N] = -1  # No current number at the start

        return self.observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}  # Game is over

        number = action + 1  # Map action index to number in Number Set (1 to N)

        # Check if the number has already been used
        if number in self.used_numbers:
            self.done = True
            return self.observation, -1, True, False, {}  # Invalid move, game over

        # Check if the action is valid
        if self.current_number is None:
            valid = True  # First move can be any number
        else:
            # Check if number is a factor or multiple of current_number
            if number % self.current_number == 0 or self.current_number % number == 0:
                valid = True
            else:
                valid = False

        if not valid:
            self.done = True
            return self.observation, -1, True, False, {}  # Invalid move, game over

        # Valid move, update the game state
        self.used_numbers.add(number)
        self.observation[number - 1] = 1  # Mark number as used
        self.current_number = number

        # Update observation[N] to represent current number index normalized between -1 and 1
        if self.N > 1:
            self.observation[self.N] = (
                2 * (number - 1) / (self.N - 1) - 1
            )  # Normalize to [-1, 1]
        else:
            self.observation[self.N] = 0  # If N is 1, set to 0

        # Check if the next player has any valid moves
        valid_moves_remaining = self.valid_moves()
        if not valid_moves_remaining:
            # Opponent cannot make valid moves, current player wins
            self.done = True
            reward = 1  # Current player wins
            return self.observation, reward, True, False, {}

        # Game continues, switch to the next player
        reward = -10  # As per the prompt
        self.current_player *= -1  # Switch player

        return (
            self.observation,
            reward,
            False,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def valid_moves(self):
        # Returns a list of valid action indices given the current state
        valid_actions = []
        for action in range(self.N):
            number = action + 1  # Map action index to number
            if number in self.used_numbers:
                continue
            if self.current_number is None:
                # First move, any number is valid
                valid = True
            else:
                # Check if number is a factor or multiple of current_number
                if (
                    number % self.current_number == 0
                    or self.current_number % number == 0
                ):
                    valid = True
                else:
                    valid = False
            if valid:
                valid_actions.append(action)
        return valid_actions

    def render(self):
        s = "Used Numbers: {}\n".format(sorted(self.used_numbers))
        s += "Available Numbers: {}\n".format(
            [n for n in range(1, self.N + 1) if n not in self.used_numbers]
        )
        s += "Current Number: {}\n".format(self.current_number)
        s += "Current Player: {}\n".format(
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        return s
