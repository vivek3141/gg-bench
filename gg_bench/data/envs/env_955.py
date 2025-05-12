import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, Nmax=20, sequence_length=5):
        super(CustomEnv, self).__init__()
        self.Nmax = Nmax  # Maximum number in the pool
        self.sequence_length = sequence_length  # Required sequence length to win

        # Define action and observation space
        # The actions are numbers from 1 to Nmax (0 to Nmax-1 in indices)
        self.action_space = spaces.Discrete(self.Nmax)
        # The observation is an array of size Nmax, with values:
        # 0: number is available
        # 1: number taken by player 1
        # -1: number taken by player 2
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.Nmax,), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = list(range(1, self.Nmax + 1))  # Numbers from 1 to Nmax
        self.player_hands = {1: [], -1: []}  # Hands for player 1 and player 2
        self.board = np.zeros(self.Nmax, dtype=np.int8)  # State of numbers
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        number = action + 1  # Map action to number (1-indexed)
        if action < 0 or action >= self.Nmax or self.board[action] != 0 or self.done:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        self.board[action] = self.current_player
        self.number_pool.remove(number)
        self.player_hands[self.current_player].append(number)

        # Check for winning condition
        if self.has_won(self.player_hands[self.current_player]):
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check if no more valid moves can be made
        if len(self.number_pool) == 0:
            self.done = True
            winner = self.determine_winner()
            if winner == self.current_player:
                return self.board.copy(), 1, True, False, {}
            else:
                return self.board.copy(), -1, True, False, {}
        else:
            # Switch to the other player
            self.current_player *= -1
            return self.board.copy(), 0, False, False, {}

    def has_won(self, hand):
        if len(hand) < self.sequence_length:
            return False
        sorted_hand = sorted(hand)
        consecutive_count = 1
        for i in range(1, len(sorted_hand)):
            if sorted_hand[i] == sorted_hand[i - 1] + 1:
                consecutive_count += 1
                if consecutive_count >= self.sequence_length:
                    return True
            else:
                consecutive_count = 1
        return False

    def determine_winner(self):
        # Determine the player with the longest consecutive sequence
        seq_player1 = self.longest_consecutive_sequence(self.player_hands[1])
        seq_player2 = self.longest_consecutive_sequence(self.player_hands[-1])
        if len(seq_player1) > len(seq_player2):
            return 1
        elif len(seq_player2) > len(seq_player1):
            return -1
        else:
            # Tie-breaker: highest numerical sequence
            if seq_player1 and seq_player2:
                if seq_player1[-1] > seq_player2[-1]:
                    return 1
                elif seq_player2[-1] > seq_player1[-1]:
                    return -1
            # If still tied, it's a draw
            return 0

    def longest_consecutive_sequence(self, hand):
        if not hand:
            return []
        sorted_hand = sorted(set(hand))
        longest_seq = []
        current_seq = [sorted_hand[0]]

        for i in range(1, len(sorted_hand)):
            if sorted_hand[i] == sorted_hand[i - 1] + 1:
                current_seq.append(sorted_hand[i])
            else:
                if len(current_seq) > len(longest_seq):
                    longest_seq = current_seq
                current_seq = [sorted_hand[i]]

        if len(current_seq) > len(longest_seq):
            longest_seq = current_seq
        return longest_seq

    def render(self):
        pool_numbers = [str(num) for num in self.number_pool]
        player1_hand = sorted(self.player_hands[1])
        player2_hand = sorted(self.player_hands[-1])
        return (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
            f"Number Pool: {', '.join(pool_numbers)}\n"
            f"Player 1 Hand: {player1_hand}\n"
            f"Player 2 Hand: {player2_hand}\n"
        )

    def valid_moves(self):
        return [num - 1 for num in self.number_pool]
