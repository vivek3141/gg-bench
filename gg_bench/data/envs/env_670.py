import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 5 territories (A=0, B=1, C=2, D=3, E=4)
        self.action_space = spaces.Discrete(5)

        # Observation space:
        # For each territory, we have counts of tokens for both players
        # Shape: (10,) => [P1_A, P2_A, P1_B, P2_B, ..., P1_E, P2_E]
        self.observation_space = spaces.Box(low=0, high=5, shape=(10,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board: counts of tokens in each territory for both players
        self.board = np.zeros(10, dtype=np.int32)
        # Current player: 1 for Player 1 (X), -1 for Player 2 (O)
        self.current_player = 1
        self.done = False
        # Tokens used by each player
        self.tokens_used = {1: 0, -1: 0}
        # Control of territories
        self.controlled_territories = {1: [], -1: []}
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Place a token in the selected territory
        index = action * 2  # Territory index for current player
        # Check if current player has tokens remaining
        if self.tokens_used[self.current_player] >= 5:
            # Invalid move: no tokens left
            self.done = True
            return self.board.copy(), -10, True, False, {}

        self.board[index] += 1
        self.tokens_used[self.current_player] += 1

        # Update control
        self.update_control()

        # Check for win condition
        if len(self.controlled_territories[self.current_player]) >= 3:
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check if no valid moves remain (both players have used all tokens)
        if self.tokens_used[1] >= 5 and self.tokens_used[-1] >= 5:
            self.done = True
            # Determine winner based on control
            if len(self.controlled_territories[1]) > len(
                self.controlled_territories[-1]
            ):
                winner = 1
            elif len(self.controlled_territories[-1]) > len(
                self.controlled_territories[1]
            ):
                winner = -1
            else:
                # Since a draw is impossible, we can declare the last player as the winner
                winner = self.current_player
            if winner == self.current_player:
                return self.board.copy(), 1, True, False, {}
            else:
                return self.board.copy(), -1, True, False, {}

        # Switch current player
        self.current_player *= -1
        return (
            self.board.copy(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def update_control(self):
        self.controlled_territories = {1: [], -1: []}
        for t in range(5):  # Territories A-E
            p1_tokens = self.board[t * 2]
            p2_tokens = self.board[t * 2 + 1]
            if p1_tokens > p2_tokens:
                self.controlled_territories[1].append(t)
            elif p2_tokens > p1_tokens:
                self.controlled_territories[-1].append(t)
            # else contested, controlled by neither

    def render(self):
        territories = ["A", "B", "C", "D", "E"]
        board_str = "Current Player: {}\n".format(
            "X" if self.current_player == 1 else "O"
        )
        board_str += "Territories:\n"
        for t in range(5):
            p1_tokens = self.board[t * 2]
            p2_tokens = self.board[t * 2 + 1]
            territory_control = "Contested"
            if p1_tokens > p2_tokens:
                territory_control = "Controlled by X"
            elif p2_tokens > p1_tokens:
                territory_control = "Controlled by O"
            board_str += "Territory {}: X={} O={} - {}\n".format(
                territories[t], p1_tokens, p2_tokens, territory_control
            )
        return board_str

    def valid_moves(self):
        # A move is valid if the player has tokens left
        if self.tokens_used[self.current_player] >= 5:
            return []
        else:
            return list(range(5))  # Can choose any territory
