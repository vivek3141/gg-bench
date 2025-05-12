import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum starting number for the game
        self.max_starting_number = 100

        # Initialize starting number and current number
        self.starting_number = self.max_starting_number
        self.current_number = self.starting_number

        # Define action space
        # Actions correspond to divisors from 2 up to max_starting_number
        # Action indices: 0 corresponds to divisor 2, 1 to divisor 3, etc.
        self.action_space = spaces.Discrete(
            self.max_starting_number - 1
        )  # Actions: 0 to 98

        # Define observation space
        # Observation is the current number, ranging from 1 to max_starting_number
        self.observation_space = spaces.Box(
            low=1, high=self.max_starting_number, shape=(1,), dtype=np.int32
        )

        # Internal variables
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game over flag

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set starting number from options or use default
        if options and "starting_number" in options:
            self.starting_number = options["starting_number"]
            if self.starting_number > self.max_starting_number:
                self.starting_number = self.max_starting_number
        else:
            self.starting_number = self.max_starting_number

        self.current_number = self.starting_number

        # Reset internal variables
        self.current_player = 1
        self.done = False

        # Return initial observation and info
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is over, return current state
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Get valid actions for the current player
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player cannot make a move, player loses
            self.done = True
            reward = -10  # Penalty for losing
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if action not in valid_actions:
            # Invalid move, player loses
            self.done = True
            reward = -10  # Penalty for invalid move
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Map action index to divisor
        divisor = action + 2

        # Perform the division
        self.current_number = int(self.current_number / divisor)

        # Check for win condition
        if self.current_number == 1:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if the next player has any valid moves
        next_valid_moves = []
        for next_divisor in range(2, self.current_number + 1):
            if self.current_number % next_divisor == 0:
                next_action_index = next_divisor - 2
                next_valid_moves.append(next_action_index)

        if not next_valid_moves:
            # Next player cannot move, current player wins
            self.done = True
            reward = 1  # Reward for winning
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to next player
        self.current_player = 3 - self.current_player  # Alternate between 1 and 2

        # Continue the game
        return np.array([self.current_number], dtype=np.int32), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        valid_actions = self.valid_moves()
        valid_divisors = [action + 2 for action in valid_actions]
        output = (
            f"Current number: {self.current_number}\n"
            f"Current player: Player {self.current_player}\n"
            f"Available divisors: {valid_divisors}\n"
        )
        return output

    def valid_moves(self):
        if self.done:
            return []
        valid_actions = []
        for divisor in range(2, self.current_number + 1):
            if self.current_number % divisor == 0:
                action_index = divisor - 2
                valid_actions.append(action_index)
        return valid_actions
