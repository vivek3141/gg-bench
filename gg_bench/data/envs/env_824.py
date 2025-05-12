import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting numbers 1 to 20 (indices 0 to 19)
        self.action_space = spaces.Discrete(20)

        # Observation is a 20-element vector:
        # - 0: Number is available
        # - 1: Number selected by current player
        # - -1: Number selected by opponent
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = np.zeros(20, dtype=np.int8)
        self.current_player = 1  # Players are 1 and -1
        self.done = False
        return self.observation, {}  # Return observation and info

    def valid_moves(self):
        if self.done:
            return []

        # If current player has no numbers selected yet (first turn)
        if not (self.observation == self.current_player).any():
            # Can select any available number
            return [i for i in range(20) if self.observation[i] == 0]
        else:
            # Last number selected by current player
            player_indices = np.where(self.observation == self.current_player)[0]
            last_number_index = player_indices[-1]
            last_number = last_number_index + 1  # Numbers are from 1 to 20

            # Available numbers
            available_indices = [i for i in range(20) if self.observation[i] == 0]

            # Valid moves: numbers that are factors or multiples of last_number
            valid_moves = []
            for index in available_indices:
                number = index + 1
                if last_number % number == 0 or number % last_number == 0:
                    valid_moves.append(index)
            return valid_moves

    def step(self, action):
        if self.done:
            return self.observation, -10, True, False, {}

        # Check if action is within bounds
        if action < 0 or action >= 20:
            # Invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        # Check if action is valid
        if self.observation[action] != 0 or action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        # Valid move
        self.observation[action] = self.current_player

        # Check for win by completing sequence of five numbers
        player_sequence_length = np.sum(self.observation == self.current_player)
        if player_sequence_length == 5:
            self.done = True
            return self.observation, 1, True, False, {}

        # Check if opponent can make a valid move
        opponent = -self.current_player
        self.current_player = (
            opponent  # Temporarily switch to opponent for valid moves check
        )
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot make a valid move; current player wins
            self.done = True
            self.current_player = -self.current_player  # Switch back to current player
            return self.observation, 1, True, False, {}

        # Switch back to current player for next turn
        self.current_player = -self.current_player

        # Switch to opponent for next turn
        self.current_player = -self.current_player

        return self.observation, 0, False, False, {}

    def render(self):
        # Build a string representing the game state
        s = ""
        s += "Available numbers:\n"
        available_numbers = [str(i + 1) for i in range(20) if self.observation[i] == 0]
        s += " ".join(available_numbers) + "\n"

        s += "Player 1 sequence:\n"
        player1_numbers = [str(i + 1) for i in range(20) if self.observation[i] == 1]
        s += " ".join(player1_numbers) + "\n"

        s += "Player 2 sequence:\n"
        player2_numbers = [str(i + 1) for i in range(20) if self.observation[i] == -1]
        s += " ".join(player2_numbers) + "\n"

        s += f"Current player: {1 if self.current_player == 1 else 2}\n"

        return s
