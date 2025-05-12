import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Numbers from 2 to 30
        self.numbers_range = list(range(2, 31))  # numbers from 2 to 30 inclusive
        self.num_numbers = len(self.numbers_range)  # Should be 29

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_numbers)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.num_numbers,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers = np.zeros(
            self.num_numbers, dtype=np.float32
        )  # 0: unselected, 1: Player 1, -1: Player 2
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        self.player_has_won = {1: False, -1: False}
        return self.numbers.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.numbers.copy(), 0, True, False, {}  # Game is over

        # Check if current player can declare victory
        if self.player_has_won[self.current_player]:
            self.done = True
            return self.numbers.copy(), 1, True, False, {}

        # Validate action
        if self.numbers[action] != 0:
            # Invalid move
            self.done = True
            return self.numbers.copy(), -10, True, False, {}

        # Select the number
        self.numbers[action] = self.current_player

        # Check if the current player has achieved the winning condition
        player_numbers = [
            self.numbers_range[i]
            for i in range(self.num_numbers)
            if self.numbers[i] == self.current_player
        ]
        if len(player_numbers) >= 3:
            # Check all combinations of three numbers
            for combo in itertools.combinations(player_numbers, 3):
                if self.check_sequence(combo):
                    # Player has formed a valid sequence
                    self.player_has_won[self.current_player] = True
                    break  # No need to check further

        # Check if there are no more valid moves
        if np.all(self.numbers != 0):
            self.done = True
            return self.numbers.copy(), 0, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return (
            self.numbers.copy(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def check_sequence(self, numbers):
        # Check if a sequence can be formed where each number divides the next
        for perm in itertools.permutations(numbers, 3):
            if perm[1] % perm[0] == 0 and perm[2] % perm[1] == 0:
                return True
        return False

    def render(self):
        # Return a string representation of the game state
        output = "Numbers selected:\n"
        for i in range(self.num_numbers):
            num = self.numbers_range[i]
            owner = self.numbers[i]
            if owner == 0:
                output += f"{num}: Unselected\n"
            elif owner == 1:
                output += f"{num}: Player 1\n"
            else:
                output += f"{num}: Player 2\n"
        return output

    def valid_moves(self):
        # Return a list of valid moves (indices of unselected numbers)
        if self.done:
            return []
        return [i for i in range(self.num_numbers) if self.numbers[i] == 0]
