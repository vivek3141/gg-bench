import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: actions are integers from 1 to 100 (possible divisors)
        self.action_space = spaces.Discrete(101)  # Actions: 1 to 100 inclusive

        # Define observation space: the Current Number
        self.observation_space = spaces.Box(low=1, high=100, shape=(1,), dtype=np.int32)

        # Initialize variables
        self.current_number = None
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.last_change_turn = (
            None  # To keep track of the last turn when number changed
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with a random number between 10 and 100 inclusive
        self.current_number = self.np_random.integers(10, 101)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.last_change_turn = None  # Reset last change turn
        observation = np.array([self.current_number], dtype=np.int32)
        return observation, {}

    def step(self, action):
        info = {}
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            observation = np.array([self.current_number], dtype=np.int32)
            return observation, reward, self.done, False, info

        # Perform the action
        chosen_divisor = action
        prev_number = self.current_number
        self.current_number = self.current_number // chosen_divisor

        # Update last change turn if the number changed
        if self.current_number != prev_number:
            self.last_change_turn = self.current_player

        # Check for win condition
        if self.current_number == 1:
            # Current player wins
            self.done = True
            reward = 1
            observation = np.array([self.current_number], dtype=np.int32)
            return observation, reward, self.done, False, info

        # Check if next player has valid moves
        next_valid_moves = self._get_proper_divisors(self.current_number)
        if len(next_valid_moves) == 1 and next_valid_moves[0] == 1:
            # Next player cannot make a valid move, current player wins
            self.done = True
            reward = 1
            observation = np.array([self.current_number], dtype=np.int32)
            return observation, reward, self.done, False, info

        # Check for stagnation (number doesn't change after both players' turns)
        if (
            self.last_change_turn is not None
            and self.last_change_turn != self.current_player
        ):
            if prev_number == self.current_number:
                # The number didn't change during both players' turns, game ends
                self.done = True
                if self.last_change_turn == self.current_player:
                    # Opponent made the last valid change, current player loses
                    reward = -1
                else:
                    # Current player made the last valid change, current player wins
                    reward = 1
                observation = np.array([self.current_number], dtype=np.int32)
                return observation, reward, self.done, False, info

        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0  # No immediate reward
        observation = np.array([self.current_number], dtype=np.int32)
        return observation, reward, self.done, False, info

    def render(self):
        return f"Current Number: {self.current_number}, Current Player: {self.current_player}"

    def valid_moves(self):
        return self._get_proper_divisors(self.current_number)

    def _get_proper_divisors(self, n):
        # Returns a list of proper divisors of n (includes 1 and excludes n)
        divisors = [i for i in range(1, n) if n % i == 0]
        return divisors
