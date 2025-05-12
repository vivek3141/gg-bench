import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 20 possible actions (selecting numbers 1 to 20)
        self.action_space = spaces.Discrete(20)
        # Observation space is a vector of size 20 with values -1, 0, or 1
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(20,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 0: number is available, 1: selected by current player, -1: selected by opponent
        self.number_pool = np.zeros(20, dtype=np.float32)
        # Current player: 1 or -1
        self.current_player = 1
        self.terminated = False
        observation = self.number_pool.copy()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.terminated:
            observation = self.number_pool.copy() * self.current_player
            return observation, 0, True, False, {}
        if action < 0 or action >= 20 or self.number_pool[action] != 0:
            # Invalid move
            self.terminated = True
            observation = self.number_pool.copy() * self.current_player
            return observation, -10, True, False, {}
        else:
            # Valid move
            self.number_pool[action] = self.current_player
            # Check for victory for current player
            player_numbers = (
                np.where(self.number_pool == self.current_player)[0] + 1
            )  # Numbers from 1 to 20
            if self.check_victory(player_numbers):
                self.terminated = True
                observation = self.number_pool.copy() * self.current_player
                return observation, 1, True, False, {}
            else:
                # Check if all numbers are selected
                if np.all(self.number_pool != 0):
                    # Game End Due to Number Exhaustion
                    player_numbers = (
                        np.where(self.number_pool == self.current_player)[0] + 1
                    )
                    opponent_numbers = (
                        np.where(self.number_pool == -self.current_player)[0] + 1
                    )
                    player_longest_trail = self.longest_consecutive_trail(
                        player_numbers
                    )
                    opponent_longest_trail = self.longest_consecutive_trail(
                        opponent_numbers
                    )
                    if player_longest_trail > opponent_longest_trail:
                        winner = self.current_player
                    elif opponent_longest_trail > player_longest_trail:
                        winner = -self.current_player
                    else:
                        # Tie-breaker: higher numbers in trail wins
                        player_trail_value = self.highest_trail_value(player_numbers)
                        opponent_trail_value = self.highest_trail_value(
                            opponent_numbers
                        )
                        if player_trail_value > opponent_trail_value:
                            winner = self.current_player
                        else:
                            winner = -self.current_player
                    self.terminated = True
                    observation = self.number_pool.copy() * self.current_player
                    if winner == self.current_player:
                        return observation, 1, True, False, {}
                    else:
                        return observation, 0, True, False, {}
                else:
                    # Switch current player
                    self.current_player *= -1
                    observation = self.number_pool.copy() * self.current_player
                    return observation, 0, False, False, {}

    def render(self):
        number_pool_list = [i + 1 for i in range(20) if self.number_pool[i] == 0]
        player1_numbers = [i + 1 for i in range(20) if self.number_pool[i] == 1]
        player2_numbers = [i + 1 for i in range(20) if self.number_pool[i] == -1]
        render_str = f"Number Pool: {number_pool_list}\n"
        render_str += f"Player 1 Collection: {sorted(player1_numbers)}\n"
        render_str += f"Player 2 Collection: {sorted(player2_numbers)}\n"
        return render_str

    def valid_moves(self):
        return [i for i in range(20) if self.number_pool[i] == 0]

    def check_victory(self, player_numbers):
        # Check if player_numbers contain a trail of 7 consecutive numbers
        if len(player_numbers) < 7:
            return False
        player_numbers_sorted = np.sort(player_numbers)
        # Check for sequences of 7 consecutive numbers
        for i in range(len(player_numbers_sorted) - 6):
            if np.array_equal(
                player_numbers_sorted[i : i + 7],
                np.arange(player_numbers_sorted[i], player_numbers_sorted[i] + 7),
            ):
                return True
        return False

    def longest_consecutive_trail(self, player_numbers):
        if len(player_numbers) == 0:
            return 0
        player_numbers_sorted = np.sort(player_numbers)
        longest = current = 1
        for i in range(1, len(player_numbers_sorted)):
            if player_numbers_sorted[i] == player_numbers_sorted[i - 1] + 1:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 1
        return longest

    def highest_trail_value(self, player_numbers):
        # Return the maximum value of the longest trail
        if len(player_numbers) == 0:
            return 0
        player_numbers_sorted = np.sort(player_numbers)
        longest = current = 1
        max_value = player_numbers_sorted[0]
        temp_max = player_numbers_sorted[0]
        for i in range(1, len(player_numbers_sorted)):
            if player_numbers_sorted[i] == player_numbers_sorted[i - 1] + 1:
                current += 1
                temp_max = player_numbers_sorted[i]
                if current >= longest:
                    longest = current
                    if temp_max > max_value:
                        max_value = temp_max
            else:
                current = 1
                temp_max = player_numbers_sorted[i]
        return max_value
