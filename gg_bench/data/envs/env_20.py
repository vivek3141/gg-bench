import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(50)

        # Observation space is an array of length 51
        # First 50 entries: -1 (opponent's picks), 0 (available), 1 (agent's picks)
        # Last entry: last opponent's move (range -1 to 49)
        self.observation_space = spaces.Box(
            low=-1, high=49, shape=(51,), dtype=np.int32
        )

        # Precompute factors for numbers 1 to 50
        self.factors = []
        for n in range(1, 51):
            self.factors.append(self.get_factors(n))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            50, dtype=np.int32
        )  # 0: available, 1: player 1, -1: player 2
        self.current_player = 1  # 1 or -1
        self.last_opponent_action = -1  # -1 indicates no previous action
        self.done = False
        return self.get_observation(), {}  # Observation and info

    def get_observation(self):
        # Return the observation adjusted for the current player
        # Multiply board by current_player so that agent's own moves are 1, opponent's -1
        observation = np.append(
            self.board * self.current_player, self.last_opponent_action
        )
        return observation

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}
        if self.board[action] != 0 or action not in self.valid_moves():
            self.done = True
            return self.get_observation(), -10, True, False, {}  # Invalid move

        self.board[action] = self.current_player
        self.last_opponent_action = action

        # Determine next player
        next_player = -self.current_player
        # Temporarily switch to next player to get their valid moves
        self.current_player = next_player

        if len(self.valid_moves()) == 0:
            self.done = True
            # Switch back to current player for correct observation
            self.current_player = -next_player
            return self.get_observation(), 1, True, False, {}  # Current player wins

        # Switch to next player
        self.current_player = next_player
        return self.get_observation(), 0, False, False, {}  # Continue game

    def render(self):
        # Build a string representation of the game state
        output = "--- Prime Connection ---\n\n"
        output += "Available Numbers:\n"
        available_numbers = [str(i + 1) for i in range(50) if self.board[i] == 0]
        output += ", ".join(available_numbers) + "\n\n"
        output += f"Current Player's Turn: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += (
            f"Last number selected by opponent: "
            f"{self.last_opponent_action +1 if self.last_opponent_action != -1 else 'None'}\n"
        )
        return output

    def valid_moves(self):
        if self.last_opponent_action == -1:
            # First player's move, can select any available number
            valid_moves = np.where(self.board == 0)[0].tolist()
            return valid_moves
        else:
            last_number = self.last_opponent_action + 1
            last_factors = self.factors[last_number - 1]
            valid_moves = []
            for idx in np.where(self.board == 0)[0]:
                number = idx + 1
                if len(last_factors.intersection(self.factors[number - 1])) > 0:
                    valid_moves.append(idx)
            return valid_moves

    @staticmethod
    def get_factors(n):
        factors = set()
        for i in range(2, n // 2 + 1):
            if n % i == 0:
                factors.add(i)
        factors.add(n)
        return factors
