import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action_space is Discrete(9), representing numbers 1 to 9
        self.action_space = spaces.Discrete(9)
        # The observation_space is a Box space with values -1, 0, or 1 for each number
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers = np.zeros(
            9, dtype=np.int8
        )  # 0 for available numbers, 1 for Player 1, -1 for Player 2
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        self.last_captured = {
            1: None,
            -1: None,
        }  # Store the last captured number for each player
        return self.numbers.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.numbers.copy(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player has no valid moves, skip their turn
            self.current_player *= -1  # Switch player
            valid_actions = self.valid_moves()
            if not valid_actions:
                # Neither player has valid moves, game over
                self.done = True
                return self.numbers.copy(), 0, True, False, {}

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self.numbers.copy(), -10, True, False, {}

        # Valid move, capture the number
        self.numbers[action] = self.current_player
        N = action + 1  # Number corresponding to action index (1-9)
        self.last_captured[self.current_player] = N

        # Check if game is over (all numbers captured)
        if np.all(self.numbers != 0):
            # Current player wins by capturing the last number
            self.done = True
            return self.numbers.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        return self.numbers.copy(), 0, False, False, {}

    def render(self):
        shared_numbers = [i + 1 for i in range(9) if self.numbers[i] == 0]
        player1_captures = [i + 1 for i in range(9) if self.numbers[i] == 1]
        player2_captures = [i + 1 for i in range(9) if self.numbers[i] == -1]

        output = "Shared Number List: {}\n".format(shared_numbers)
        output += "Player 1's Captured Numbers: {}\n".format(player1_captures)
        output += "Player 2's Captured Numbers: {}\n".format(player2_captures)
        output += "Current Player: Player {}\n".format(
            1 if self.current_player == 1 else 2
        )
        return output

    def valid_moves(self):
        if self.done:
            return []

        unavailable_numbers = set()
        # Add numbers adjacent to the last captured numbers by each player
        for player in [1, -1]:
            last_num = self.last_captured[player]
            if last_num is not None:
                adjacent_numbers = [last_num - 1, last_num + 1]
                for num in adjacent_numbers:
                    if 1 <= num <= 9:
                        unavailable_numbers.add(num)

        # Valid moves are indices where numbers are available and not adjacent to last captured numbers
        valid_actions = []
        for i in range(9):
            if self.numbers[i] == 0 and (i + 1) not in unavailable_numbers:
                valid_actions.append(i)
        return valid_actions
