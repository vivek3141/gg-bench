import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 20 possible numbers (1 to 20)
        self.action_space = spaces.Discrete(20)

        # Observation space:
        # - First 20 elements: -1 (selected by Player 2), 0 (available), 1 (selected by Player 1)
        # - Last element: current number (0 if none, else 1-20)
        self.observation_space = spaces.Box(
            low=-1, high=20, shape=(21,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(20, dtype=np.int32)
        self.current_number = 0  # No number selected yet
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_obs()
        return observation, {}  # Return observation and empty info

    def _get_obs(self):
        # Observation is the board state and the current number
        return np.concatenate((self.board.copy(), [self.current_number]))

    def step(self, action):
        if self.done:
            # Game is over; no more moves allowed
            return self._get_obs(), 0, True, False, {}

        number_chosen = action + 1  # Map action to number (1-20)

        # Check if the number is already selected
        if self.board[action] != 0:
            # Invalid move: number already selected
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # First turn: any number is valid
        if self.current_number == 0:
            valid_move = True
        else:
            # Check Number Selection Rule
            if (self.current_number % number_chosen == 0) or (
                number_chosen % self.current_number == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move: does not follow the rules
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Valid move: update the game state
        self.board[action] = self.current_player
        self.current_number = number_chosen

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.get_valid_moves()

        if len(opponent_valid_moves) == 0:
            # Opponent cannot make a valid move; current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}
        else:
            # Switch to the next player
            self.current_player *= -1
            reward = 0
            return self._get_obs(), reward, False, False, {}

    def get_valid_moves(self):
        # Get a list of valid moves for the current player
        available_actions = [i for i in range(20) if self.board[i] == 0]
        if self.current_number == 0:
            # First move: all available numbers are valid
            valid_moves = available_actions
        else:
            # Subsequent moves: numbers that are factors or multiples of the current number
            valid_moves = []
            for action in available_actions:
                number = action + 1
                if (self.current_number % number == 0) or (
                    number % self.current_number == 0
                ):
                    valid_moves.append(action)
        return valid_moves

    def valid_moves(self):
        # Public method to get valid moves; used externally if needed
        return self.get_valid_moves()

    def render(self):
        # Generate a string representation of the current game state
        player1_numbers = [i + 1 for i in range(20) if self.board[i] == 1]
        player2_numbers = [i + 1 for i in range(20) if self.board[i] == -1]
        available_numbers = [i + 1 for i in range(20) if self.board[i] == 0]

        output = "\nSequence Capture Game State:\n"
        output += "---------------------------\n"
        output += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += f"Available Numbers: {available_numbers}\n"
        output += f"Player 1's Numbers: {player1_numbers}\n"
        output += f"Player 2's Numbers: {player2_numbers}\n"
        output += f"Current Number: {self.current_number if self.current_number != 0 else 'None'}\n"
        output += "---------------------------\n"
        return output
