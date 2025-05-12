import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # The action space corresponds to selecting numbers from 2 to 20 (indices 0 to 18)
        self.action_space = spaces.Discrete(
            19
        )  # Actions: 0-18 representing numbers 2-20
        # Observation space represents the state of numbers 2-20
        # -1: selected by opponent, 0: in shared pool, 1: selected by current player
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19,), dtype=np.int8)
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared pool with numbers from 2 to 20
        self.state = np.zeros(19, dtype=np.int8)  # 0: in shared pool
        self.current_player = 1  # Player 1 starts (can be 1 or -1)
        self.done = False
        # Initialize player selections
        self.player_numbers = {1: [], -1: []}  # Numbers selected by each player
        return self.state.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.state.copy(), 0, True, False, {}
        # Check if current player has any valid moves
        valid_moves = self.get_valid_moves(self.current_player)
        if not valid_moves:
            # Current player cannot make a move; they lose
            self.done = True
            return self.state.copy(), -1, True, False, {}
        # Map action to the actual number
        number = action + 2  # Numbers from 2 to 20
        if self.state[action] != 0:
            # Invalid move: Number has already been selected
            self.done = True
            return self.state.copy(), -10, True, False, {}
        # Check if the action is a valid move
        if action not in valid_moves:
            # Invalid move: Not co-prime with player's numbers
            self.done = True
            return self.state.copy(), -10, True, False, {}
        # Valid move: Update the state
        self.state[action] = self.current_player  # Mark the number as selected
        self.player_numbers[self.current_player].append(number)
        # Check if the opponent can make a valid move
        opponent = -self.current_player
        opponent_valid_moves = self.get_valid_moves(opponent)
        if not opponent_valid_moves:
            # Opponent cannot make a move; current player wins
            self.done = True
            return self.state.copy(), 1, True, False, {}
        else:
            # Switch to the opponent and continue the game
            self.current_player = opponent
            return self.state.copy(), 0, False, False, {}

    def get_valid_moves(self, player):
        """Return a list of valid actions for the given player."""
        valid_moves = []
        player_numbers = self.player_numbers[player]
        for action in range(19):
            if self.state[action] != 0:
                continue  # Number already selected
            number = action + 2  # Map action to number
            if all(np.gcd(number, n) == 1 for n in player_numbers):
                valid_moves.append(action)
        return valid_moves

    def valid_moves(self):
        """Return a list of valid actions for the current player."""
        return self.get_valid_moves(self.current_player)

    def render(self):
        """Return a visual representation of the environment's state."""
        shared_pool = [i + 2 for i, s in enumerate(self.state) if s == 0]
        player1_numbers = self.player_numbers[1]
        player2_numbers = self.player_numbers[-1]
        current_player_str = "Player 1" if self.current_player == 1 else "Player 2"
        output = f"Current player: {current_player_str}\n"
        output += f"Shared Pool: {shared_pool}\n"
        output += f"Player 1 Numbers: {player1_numbers}\n"
        output += f"Player 2 Numbers: {player2_numbers}\n"
        return output
