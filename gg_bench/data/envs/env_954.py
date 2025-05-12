import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(9) representing numbers 1 to 9
        self.action_space = spaces.Discrete(9)

        # Observation space: A vector of length 16
        # [current_player, shared_pool(9), player1_hand(3), player2_hand(3)]
        self.observation_space = spaces.Box(low=0, high=9, shape=(16,), dtype=np.int8)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the shared pool (numbers 1 to 9, all available)
        self.shared_pool = [
            1
        ] * 9  # Index 0 represents number 1, index 8 represents number 9

        # Initialize player hands
        self.player_hands = {1: [], -1: []}  # Player 1: 1, Player 2: -1

        # Set the current player (Player 1 starts)
        self.current_player = 1

        # Game over flag
        self.done = False

        # Used for tie-breaker rules
        self.player_completed_hand = {1: False, -1: False}

        # Return observation and info
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number_selected = action + 1  # Numbers are 1-indexed from action_space 0-8

        # Check if action is valid (number is available)
        if self.shared_pool[action] == 0:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Valid move
        # Remove the number from shared pool
        self.shared_pool[action] = 0

        # Add the number to current player's hand
        self.player_hands[self.current_player].append(number_selected)

        # Check for win condition
        reward = 0
        done = False

        current_hand = self.player_hands[self.current_player]
        if len(current_hand) == 3:
            # Check if the hand is in strictly ascending order
            if current_hand[0] < current_hand[1] < current_hand[2]:
                # Current player wins
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, {}
            else:
                self.player_completed_hand[self.current_player] = True

        # If both players have completed their hands or shared pool is empty
        if (len(self.player_hands[1]) == 3 and len(self.player_hands[-1]) == 3) or sum(
            self.shared_pool
        ) == 0:
            # Apply tie-breaker rules
            winner = self._determine_winner()
            if winner == self.current_player:
                reward = 1
            elif winner == 0:
                reward = 0  # Tie
            else:
                reward = -1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1

        return self._get_observation(), reward, False, False, {}

    def _get_observation(self):
        obs = np.zeros((16,), dtype=np.int8)
        obs[0] = self.current_player
        obs[1:10] = self.shared_pool  # Indicate availability of numbers 1 to 9

        # Player 1's hand
        player1_hand = self.player_hands[1]
        obs[10:13] = player1_hand + [0] * (3 - len(player1_hand))

        # Player 2's hand
        player2_hand = self.player_hands[-1]
        obs[13:16] = player2_hand + [0] * (3 - len(player2_hand))

        return obs

    def _determine_winner(self):
        # Determine the winner according to the tie-breaker rules

        # First, check if either player has ascending sequence
        p1_hand = self.player_hands[1]
        p2_hand = self.player_hands[-1]

        p1_ascending = len(p1_hand) == 3 and p1_hand[0] < p1_hand[1] < p1_hand[2]
        p2_ascending = len(p2_hand) == 3 and p2_hand[0] < p2_hand[1] < p2_hand[2]

        if p1_ascending and p2_ascending:
            # Both players have ascending sequences
            # The player who moved second wins
            return -1  # Player 2 wins
        elif p1_ascending:
            return 1
        elif p2_ascending:
            return -1
        else:
            # Neither player has an ascending sequence
            # Tie-breaker rules:

            # 1. Player with the longest ascending subsequence in their hand wins
            p1_longest_seq = self._longest_ascending_subsequence(p1_hand)
            p2_longest_seq = self._longest_ascending_subsequence(p2_hand)

            if p1_longest_seq > p2_longest_seq:
                return 1
            elif p1_longest_seq < p2_longest_seq:
                return -1
            else:
                # 2. Player with the lowest sum of their hand's numbers wins
                p1_sum = sum(p1_hand)
                p2_sum = sum(p2_hand)

                if p1_sum < p2_sum:
                    return 1
                elif p1_sum > p2_sum:
                    return -1
                else:
                    # 3. Player who took the last turn loses
                    # Since current player just played, they took the last turn and lose
                    return -self.current_player  # Current player loses

    def _longest_ascending_subsequence(self, hand):
        n = len(hand)
        if n == 0:
            return 0
        dp = [1] * n  # dp[i] = length of LIS ending at hand[i]
        for i in range(n):
            for j in range(i):
                if hand[j] < hand[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def render(self):
        shared_numbers = [
            idx + 1 for idx, val in enumerate(self.shared_pool) if val == 1
        ]
        shared_pool_str = f"Shared Pool: {shared_numbers}\n"

        player1_hand = self.player_hands[1]
        player2_hand = self.player_hands[-1]

        player1_hand_str = f"Player 1's Hand: {player1_hand}\n"
        player2_hand_str = f"Player 2's Hand: {player2_hand}\n"

        current_player_str = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"

        return (
            shared_pool_str + player1_hand_str + player2_hand_str + current_player_str
        )

    def valid_moves(self):
        return [idx for idx, val in enumerate(self.shared_pool) if val == 1]
