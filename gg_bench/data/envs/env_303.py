import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(26)  # 26 letters in the alphabet
        self.observation_space = spaces.Box(low=-1, high=1, shape=(26,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            26, dtype=np.int8
        )  # 0: unclaimed, 1: player 1, -1: player 2
        self.current_player = 1  # Player 1 starts (can be 1 or -1)
        self.done = False
        return self.board, {}  # Return observation and info

    def step(self, action):
        # Check for invalid moves
        if self.done or self.board[action] != 0:
            return self.board, -10, True, False, {}  # Invalid move

        # Claim the letter
        self.board[action] = self.current_player

        # Check for win
        claimed_indices = np.where(self.board == self.current_player)[0]
        claimed_indices.sort()
        win = False
        for i in range(len(claimed_indices) - 2):
            if (
                claimed_indices[i + 1] == claimed_indices[i] + 1
                and claimed_indices[i + 2] == claimed_indices[i] + 2
            ):
                win = True
                break

        if win:
            self.done = True
            return self.board, 1, True, False, {}  # Win

        # Check if all letters are claimed (game over with no winner)
        if np.all(self.board != 0):
            self.done = True
            return self.board, 0, True, False, {}  # Game over, no winner

        # Switch player
        self.current_player *= -1

        return self.board, 0, False, False, {}  # Continue game

    def render(self):
        # Build visual representation of the game state
        letters = [chr(ord("A") + i) for i in range(26)]
        representation = "Available Letters:\n"
        for i in range(26):
            if self.board[i] == 0:
                representation += letters[i] + " "
        representation += "\n\n"

        representation += "Player 1's Collection:\n"
        player1_letters = [letters[i] for i in range(26) if self.board[i] == 1]
        representation += ", ".join(player1_letters) + "\n\n"

        representation += "Player 2's Collection:\n"
        player2_letters = [letters[i] for i in range(26) if self.board[i] == -1]
        representation += ", ".join(player2_letters) + "\n"

        return representation

    def valid_moves(self):
        return [i for i in range(26) if self.board[i] == 0]
