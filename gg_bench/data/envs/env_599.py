import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0 - Discard, 1 - Keep
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # Indices 0-8 represent numbers 1-9 with the following values:
        #  0: Not yet drawn
        #  1: In current player's collection
        # -1: In opponent's collection
        #  2: Discarded
        # Index 9: Drawn number (1-9), 0 if no number is drawn
        self.observation_space = spaces.Box(low=-1, high=9, shape=(10,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.game_state = np.zeros(9, dtype=np.int8)  # Numbers 1-9 status
        self.stack = np.arange(1, 10)
        np.random.shuffle(self.stack)
        self.stack = list(self.stack)
        self.done = False
        self.drawn_number = self.stack.pop(0) if self.stack else 0
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if action not in [0, 1]:
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )

        # Check if drawn number is already taken (should not happen)
        if self.game_state[self.drawn_number - 1] != 0:
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )

        # Apply action
        if action == 1:
            # Keep the drawn number
            self.game_state[self.drawn_number - 1] = self.current_player
        else:
            # Discard the drawn number
            self.game_state[self.drawn_number - 1] = 2  # Discarded

        # Check for win condition
        collected_numbers = [
            idx + 1
            for idx, val in enumerate(self.game_state)
            if val == self.current_player
        ]
        if self._check_win(collected_numbers):
            self.done = True
            return (
                self._get_observation(),
                1,  # Current player wins
                True,
                False,
                {},
            )

        # Check if the stack is exhausted
        if not self.stack:
            # Determine the winner based on sums
            opponent = -self.current_player
            opponent_numbers = [
                idx + 1 for idx, val in enumerate(self.game_state) if val == opponent
            ]
            current_sum = self._best_sum(collected_numbers)
            opponent_sum = self._best_sum(opponent_numbers)

            if current_sum > opponent_sum:
                self.done = True
                return (
                    self._get_observation(),
                    1,  # Current player wins
                    True,
                    False,
                    {},
                )
            else:
                self.done = True
                return (
                    self._get_observation(),
                    -1,  # Current player loses
                    True,
                    False,
                    {},
                )

        # Switch player
        self.current_player *= -1

        # Draw the next number
        self.drawn_number = self.stack.pop(0) if self.stack else 0

        return (
            self._get_observation(),
            0,  # No immediate reward
            False,  # Game continues
            False,
            {},
        )

    def _check_win(self, numbers):
        # Check all combinations of collected numbers for a sum of exactly 15
        for r in range(1, len(numbers) + 1):
            for combo in combinations(numbers, r):
                if sum(combo) == 15:
                    return True
        return False

    def _best_sum(self, numbers):
        # Find the highest sum <= 15 from any combination of collected numbers
        best = -1
        for r in range(1, len(numbers) + 1):
            for combo in combinations(numbers, r):
                s = sum(combo)
                if s <= 15 and s > best:
                    best = s
        return best

    def _get_observation(self):
        # Build the observation array
        observation = np.zeros(10, dtype=np.int8)
        observation[:9] = self.game_state
        observation[9] = self.drawn_number if self.drawn_number else 0
        return observation

    def render(self):
        # Create a string representation of the game state
        game_str = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        game_str += "Game State:\n"
        status_map = {0: "Not drawn", 1: "Player 1", -1: "Player 2", 2: "Discarded"}
        for i in range(9):
            status = status_map[self.game_state[i]]
            game_str += f"Number {i+1}: {status}\n"
        game_str += f"Drawn Number: {self.drawn_number}\n"
        return game_str

    def valid_moves(self):
        # Always returns [0, 1] as valid moves unless the game is over
        if self.done:
            return []
        return [0, 1]
