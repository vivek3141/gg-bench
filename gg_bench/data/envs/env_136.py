import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(20)  # Numbers from 1 to 20

        # Observation space: Array of 20 numbers
        # 0: available, 1: owned by current player, -1: owned by opponent
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = set(range(1, 21))  # Numbers from 1 to 20
        self.player1_numbers = []
        self.player2_numbers = []
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Observation: Array of zeros (all numbers available)
        self.state = np.zeros(20, dtype=np.int8)
        return self.state, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.state, 0, True, False, {}  # Game already over

        if action < 0 or action >= 20:
            return self.state, -10, True, False, {}

        number = action + 1  # Map action to number (1 to 20)

        if number not in self.available_numbers:
            return self.state, -10, True, False, {}

        # Valid move
        self.available_numbers.remove(number)
        if self.current_player == 1:
            self.player1_numbers.append(number)
            self.state[action] = 1
        else:
            self.player2_numbers.append(number)
            self.state[action] = -1

        # Check for victory
        if self.check_victory():
            self.done = True
            return self.state, 1, True, False, {}

        # Check if all numbers are exhausted
        if not self.available_numbers:
            # Secondary victory condition
            player1_sum = sum(self.player1_numbers)
            player2_sum = sum(self.player2_numbers)
            if player1_sum > player2_sum:
                reward = 1 if self.current_player == 1 else -1
            elif player2_sum > player1_sum:
                reward = 1 if self.current_player == 2 else -1
            else:
                reward = 0  # Tie
            self.done = True
            return self.state, reward, True, False, {}

        # Switch current player
        self.current_player = 1 if self.current_player == 2 else 2
        return self.state, 0, False, False, {}

    def check_victory(self):
        player_numbers = (
            self.player1_numbers if self.current_player == 1 else self.player2_numbers
        )

        if len(player_numbers) < 3:
            return False

        # Check all combinations of 3 numbers
        for combo in combinations(player_numbers, 3):
            sorted_combo = sorted(combo)
            if sorted_combo[1] - sorted_combo[0] == sorted_combo[2] - sorted_combo[1]:
                return True  # Found arithmetic sequence

        return False

    def render(self):
        available_numbers_str = ", ".join(
            str(num) for num in sorted(self.available_numbers)
        )
        player1_numbers_str = ", ".join(
            str(num) for num in sorted(self.player1_numbers)
        )
        player2_numbers_str = ", ".join(
            str(num) for num in sorted(self.player2_numbers)
        )

        render_str = f"Available Numbers:\n{available_numbers_str}\n\n"
        render_str += f"Player 1's Numbers: {player1_numbers_str}\n"
        render_str += f"Player 2's Numbers: {player2_numbers_str}\n"
        return render_str

    def valid_moves(self):
        return [num - 1 for num in self.available_numbers]
