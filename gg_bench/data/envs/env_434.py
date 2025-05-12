import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # The action space is digits 0-9 for picking digits, and 10 for declaring victory
        self.action_space = spaces.Discrete(11)

        # Observation space:
        # Indices 0-9: Pool digit availability (1 if available, 0 if picked)
        # Indices 10-12: Current player's digits (-1 if not yet picked)
        # Indices 13-15: Opponent's digits (-1 if not yet picked)
        # Index 16: Current player (0 or 1)
        self.observation_space = spaces.Box(low=-1, high=10, shape=(17,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared pool - all digits are available (1)
        self.pool_digits = np.ones(10, dtype=np.int8)
        # Initialize player hands - all digits are -1 (not yet picked)
        self.player_digits = {
            0: np.full(3, -1, dtype=np.int8),
            1: np.full(3, -1, dtype=np.int8),
        }
        # Current player (0 or 1)
        self.current_player = 0
        self.done = False
        self.info = {}
        # Form the initial observation
        observation = self._get_observation()
        return observation, self.info  # Return observation and info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # Check if the game is already over
        if self.done:
            return self._get_observation(), reward, terminated, truncated, self.info

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            reward = -10
            self.done = True
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self.info

        if action == 10:
            # Declare victory
            player_hand = self.player_digits[self.current_player]
            if -1 in player_hand:
                # Not enough digits to declare victory
                reward = -10
                self.done = True
                terminated = True
            else:
                # Check if any permutation of the player's digits forms a multiple of 7
                if self._can_form_multiple_of_seven(player_hand):
                    reward = 1
                    self.done = True
                    terminated = True
                else:
                    # Miscalculation results in opponent's immediate win
                    reward = -10
                    self.done = True
                    terminated = True
        else:
            # Picking a digit
            if self.pool_digits[action] == 0:
                # Digit not available
                reward = -10
                self.done = True
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Remove digit from pool
            self.pool_digits[action] = 0
            # Add digit to player's hand
            player_hand = self.player_digits[self.current_player]
            for i in range(3):
                if player_hand[i] == -1:
                    player_hand[i] = action
                    break

            self.player_digits[self.current_player] = player_hand

            # Check if player now has 3 digits
            if -1 not in player_hand:
                # If both players have 3 digits, check for end-of-game conditions
                other_player = 1 - self.current_player
                other_player_hand = self.player_digits[other_player]
                if -1 not in other_player_hand:
                    # Both players have 3 digits, resolve the game
                    self.done = True
                    terminated = True
                    winner = self._determine_winner()
                    if winner == self.current_player:
                        reward = 1
                    else:
                        reward = 0  # Other player won or it's a tie
            # No immediate victory or end game, continue to next player
        # Switch current player if game is not over
        if not self.done:
            self.current_player = 1 - self.current_player

        return self._get_observation(), reward, terminated, truncated, self.info

    def render(self):
        pool_str = "Shared Pool Digits (Available: 1, Picked: 0):\n"
        pool_str += " ".join(f"{i}:{self.pool_digits[i]}" for i in range(10))
        player_0_hand = " ".join(
            str(d) if d != -1 else "_" for d in self.player_digits[0]
        )
        player_1_hand = " ".join(
            str(d) if d != -1 else "_" for d in self.player_digits[1]
        )
        current_player_str = f"Current Player: {self.current_player}"
        hands_str = (
            f"Player 0's Hand: {player_0_hand}\nPlayer 1's Hand: {player_1_hand}"
        )
        return f"{pool_str}\n{hands_str}\n{current_player_str}"

    def valid_moves(self):
        valid_actions = []
        player_hand = self.player_digits[self.current_player]
        if -1 in player_hand:
            # Player can pick digits
            available_digits = np.nonzero(self.pool_digits)[0].tolist()
            valid_actions.extend(available_digits)
        else:
            # Player has picked 3 digits, can declare victory
            valid_actions.append(10)
        return valid_actions

    def _get_observation(self):
        # Create observation array
        observation = np.zeros(17, dtype=np.int8)
        observation[0:10] = self.pool_digits
        observation[10:13] = self.player_digits[self.current_player]
        observation[13:16] = self.player_digits[1 - self.current_player]
        observation[16] = self.current_player
        return observation

    def _can_form_multiple_of_seven(self, digits):
        from itertools import permutations

        # Generate all permutations of the digits
        perms = permutations(digits)
        for perm in perms:
            number = perm[0] * 100 + perm[1] * 10 + perm[2]
            if number % 7 == 0:
                return True
        return False

    def _determine_winner(self):
        # Both players have 3 digits
        player_0_numbers = self._get_possible_numbers(self.player_digits[0])
        player_1_numbers = self._get_possible_numbers(self.player_digits[1])

        player_0_diffs = [
            abs(n - self._nearest_multiple_of_seven(n)) for n in player_0_numbers
        ]
        player_1_diffs = [
            abs(n - self._nearest_multiple_of_seven(n)) for n in player_1_numbers
        ]

        min_diff_0 = min(player_0_diffs)
        min_diff_1 = min(player_1_diffs)

        if min_diff_0 < min_diff_1:
            winner = 0
        elif min_diff_1 < min_diff_0:
            winner = 1
        else:
            # Tiebreaker is higher number
            max_number_0 = max(player_0_numbers)
            max_number_1 = max(player_1_numbers)
            if max_number_0 > max_number_1:
                winner = 0
            elif max_number_1 > max_number_0:
                winner = 1
            else:
                # It's a tie (unlikely with 3-digit numbers)
                winner = None
        return winner

    def _get_possible_numbers(self, digits):
        from itertools import permutations

        perms = permutations(digits)
        numbers = set()
        for perm in perms:
            number = perm[0] * 100 + perm[1] * 10 + perm[2]
            numbers.add(number)
        return list(numbers)

    def _nearest_multiple_of_seven(self, n):
        return round(n / 7) * 7
