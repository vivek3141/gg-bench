import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: Subtract numbers 1, 2, 3, or 4 (indices 0-3)
        self.action_space = spaces.Discrete(4)

        # Observations: [shared_number, opponent_last_move]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([25, 4]), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 25  # Starting shared number
        self.current_player = 1  # Player 1 starts
        self.opponent_last_move = 0  # No forbidden number initially
        self.done = False  # Game is not over
        # Return initial observation and info
        observation = np.array(
            [self.shared_number, self.opponent_last_move], dtype=np.float32
        )
        return observation, {}

    def step(self, action):
        if self.done:
            # If the game is over, return the current state
            observation = np.array(
                [self.shared_number, self.opponent_last_move], dtype=np.float32
            )
            return observation, 0, True, False, {}

        selected_number = action + 1  # Map action indices to numbers 1-4

        # Check for invalid move
        invalid_move = False

        # Cannot choose the forbidden number (opponent's last move)
        if selected_number == self.opponent_last_move:
            invalid_move = True

        # Cannot reduce shared number below zero
        if self.shared_number - selected_number < 0:
            invalid_move = True

        if invalid_move:
            # Invalid move; current player loses
            reward = -10
            self.done = True
            observation = np.array(
                [self.shared_number, self.opponent_last_move], dtype=np.float32
            )
            return observation, reward, True, False, {}

        # Valid move; apply the subtraction
        self.shared_number -= selected_number

        # Update opponent's last move (forbidden number for next player)
        self.opponent_last_move = selected_number

        # Check if the game is won
        if self.shared_number == 0:
            # Current player wins
            reward = 1
            self.done = True
            observation = np.array(
                [self.shared_number, self.opponent_last_move], dtype=np.float32
            )
            return observation, reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        # Check if the next player has any valid moves
        opponent_valid_moves = []
        for i in range(4):
            number = i + 1
            if number != self.opponent_last_move and self.shared_number - number >= 0:
                opponent_valid_moves.append(i)

        if not opponent_valid_moves:
            # Opponent has no valid moves; current player wins
            reward = 1
            self.done = True
            # No need to switch player as the game ends here
            observation = np.array(
                [self.shared_number, self.opponent_last_move], dtype=np.float32
            )
            return observation, reward, True, False, {}

        # Game continues
        reward = 0
        observation = np.array(
            [self.shared_number, self.opponent_last_move], dtype=np.float32
        )
        return observation, reward, False, False, {}

    def render(self):
        return (
            f"Current Shared Number: {self.shared_number}\n"
            f"Opponent's Last Move (Forbidden Number): {self.opponent_last_move}\n"
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        )

    def valid_moves(self):
        valid_moves = []
        for i in range(4):
            number = i + 1
            if number != self.opponent_last_move and self.shared_number - number >= 0:
                valid_moves.append(i)
        return valid_moves
