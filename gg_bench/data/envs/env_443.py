import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 9 discrete actions (select numbers 1-9)
        self.action_space = spaces.Discrete(9)

        # Observation space: array of 11 elements
        # Elements 0-8: status of numbers 1-9
        #   - 0: available
        #   - 1: taken by current player
        #   - -1: taken by opponent
        # Elements 9-10: [current player's total sum, opponent's total sum]
        self.observation_space = spaces.Box(
            low=np.array([-1] * 9 + [0, 0]),
            high=np.array([1] * 9 + [45, 45]),
            dtype=np.int32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Initialize numbers as available (0)
        self.numbers = np.zeros(9, dtype=np.int32)
        # Players' total sums
        self.player_sums = {1: 0, -1: 0}
        # Randomly choose starting player: 1 or -1
        self.current_player = self.np_random.choice([1, -1])
        # Game done flag
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        # From the current player's perspective
        observation_numbers = np.zeros(9, dtype=np.int32)
        for i in range(9):
            if self.numbers[i] == self.current_player:
                observation_numbers[i] = 1
            elif self.numbers[i] == -self.current_player:
                observation_numbers[i] = -1
            else:
                observation_numbers[i] = 0
        totals = [
            self.player_sums[self.current_player],
            self.player_sums[-self.current_player],
        ]
        observation = np.concatenate([observation_numbers, totals])
        return observation

    def step(self, action):
        if self.done:
            # If game is over, return current observation
            observation = self._get_observation()
            return observation, 0, True, False, {}

        if not self.action_space.contains(action):
            # Invalid action
            observation = self._get_observation()
            return observation, -10, True, False, {}

        # Map action to number (1-9)
        number = action + 1

        if self.numbers[action] != 0:
            # Number has already been taken
            observation = self._get_observation()
            return observation, -10, True, False, {}

        # Valid move
        # Assign number to current player
        self.numbers[action] = self.current_player
        self.player_sums[self.current_player] += number

        # Check for win condition
        perfect_squares = [4, 9, 16, 25]
        player_total = self.player_sums[self.current_player]

        if player_total in perfect_squares:
            # Current player wins
            self.done = True
            observation = self._get_observation()
            return observation, 1, True, False, {}

        # Check if no numbers remain
        if not np.any(self.numbers == 0):
            # Game over, determine winner
            self.done = True
            opponent_total = self.player_sums[-self.current_player]

            # Find the next perfect square after the highest total
            possible_squares = [4, 9, 16, 25]
            next_perfect_square = None
            for square in possible_squares:
                if square > max(player_total, opponent_total):
                    next_perfect_square = square
                    break
            else:
                # No next perfect square, use the last one
                next_perfect_square = possible_squares[-1]

            player_diff = next_perfect_square - player_total
            opponent_diff = next_perfect_square - opponent_total

            if player_diff < opponent_diff:
                # Current player wins
                reward = 1
            elif player_diff > opponent_diff:
                # Current player loses
                reward = -1
            else:
                # Tie (according to game rules, no draws; handle as a loss for current player)
                reward = -1

            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Switch to other player
        self.current_player *= -1

        observation = self._get_observation()
        return observation, 0, False, False, {}

    def render(self):
        # Build a string representing the current game state
        output = "Numbers available: "
        for i in range(9):
            if self.numbers[i] == 0:
                output += f"{i+1} "
        output += "\n"
        output += f"Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += f"Player 1's total sum: {self.player_sums[1]}\n"
        output += f"Player 2's total sum: {self.player_sums[-1]}\n"
        return output

    def valid_moves(self):
        # Return list of valid actions
        return [i for i in range(9) if self.numbers[i] == 0]
