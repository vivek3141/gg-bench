import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space (9 digits * 2 sides: left or right)
        self.action_space = spaces.Discrete(18)

        # Define observation space
        # Elements 0-8: Digit pool availability (0 or 1) for digits 1-9
        # Elements 9-17: Current player's number digits, left-padded with zeros
        # Elements 18-26: Opponent player's number digits, left-padded with zeros
        # Element 27: Current player (1 or 2)
        self.observation_space = spaces.Box(low=0, high=9, shape=(28,), dtype=np.uint8)

        # Target number for divisibility (e.g., 7)
        self.target_number = 7

        # Map action indices to (digit, side)
        self.action_map = [
            (digit, side) for digit in range(1, 10) for side in ["L", "R"]
        ]

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the digit pool (digits 1-9 available)
        self.digit_pool = {digit: True for digit in range(1, 10)}
        # Initialize player numbers as empty strings
        self.player_numbers = {1: "", 2: ""}
        # Player 1 starts
        self.current_player = 1
        self.done = False

        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, False, {}

        # Map action index to (digit, side)
        digit, side = self.action_map[action]

        # Check if the selected digit is available
        if not self.digit_pool.get(digit, False):
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Remove the digit from the pool
        self.digit_pool[digit] = False

        # Place the digit on the selected side of current player's number
        if side == "L":
            self.player_numbers[self.current_player] = (
                str(digit) + self.player_numbers[self.current_player]
            )
        else:
            self.player_numbers[self.current_player] = self.player_numbers[
                self.current_player
            ] + str(digit)

        # Check if the current player's number is divisible by the target number
        player_number_int = int(self.player_numbers[self.current_player])
        if player_number_int % self.target_number == 0:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, self.done, False, {}

        # Check if all digits have been used
        if not any(self.digit_pool.values()):
            # Proceed to tie-breaker
            winner = self._tie_breaker()
            self.done = True
            if winner == 0:
                # Draw
                reward = 0
            elif winner == self.current_player:
                # Current player wins tie-breaker
                reward = 1
            else:
                # Current player loses tie-breaker
                reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Valid move; apply negative reward as per the prompt
        reward = -10

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        # Visual representation of the game state
        digit_pool_str = "Available Digits: " + " ".join(
            [str(digit) for digit in range(1, 10) if self.digit_pool[digit]]
        )
        player_numbers_str = f"Player 1's Number: {self.player_numbers[1]} \nPlayer 2's Number: {self.player_numbers[2]}"
        current_player_str = f"Current Player: Player {self.current_player}"
        return "\n".join([digit_pool_str, player_numbers_str, current_player_str])

    def valid_moves(self):
        # List of valid action indices based on available digits
        moves = []
        for action_index in range(18):
            digit, side = self.action_map[action_index]
            if self.digit_pool.get(digit, False):
                moves.append(action_index)
        return moves

    def _get_observation(self):
        # Construct the observation array
        obs = np.zeros(28, dtype=np.uint8)

        # Digit pool availability
        for digit in range(1, 10):
            obs[digit - 1] = int(self.digit_pool[digit])

        # Current player's number digits, left-padded with zeros
        curr_num_digits = [int(d) for d in self.player_numbers[self.current_player]]
        curr_num_padded = [0] * (9 - len(curr_num_digits)) + curr_num_digits
        obs[9:18] = curr_num_padded

        # Opponent player's number digits, left-padded with zeros
        opponent = 2 if self.current_player == 1 else 1
        opp_num_digits = [int(d) for d in self.player_numbers[opponent]]
        opp_num_padded = [0] * (9 - len(opp_num_digits)) + opp_num_digits
        obs[18:27] = opp_num_padded

        # Current player indicator
        obs[27] = self.current_player

        return obs

    def _tie_breaker(self):
        # Calculate the absolute differences for both players
        num1 = int(self.player_numbers[1]) if self.player_numbers[1] else 0
        num2 = int(self.player_numbers[2]) if self.player_numbers[2] else 0

        # Difference to the nearest lower multiple of the target number
        diff1 = num1 % self.target_number
        diff2 = num2 % self.target_number

        # Determine the winner based on smaller difference
        if diff1 < diff2:
            return 1
        elif diff2 < diff1:
            return 2
        else:
            return 0  # Draw
