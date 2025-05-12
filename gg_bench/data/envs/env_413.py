import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            9
        )  # Positions 0 to 8 (mapped to positions 1 to 9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            9, dtype=np.int8
        )  # Positions 0 to 8 represent positions 1 to 9
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        return self.board.copy(), self.info  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, self.info  # Game already over

        if self.board[action] != 0:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                self.info,
            )  # Invalid move

        # Place the weight
        self.board[action] = self.current_player

        # Calculate torques after the move
        left_torque = self.calculate_torque(side="left")
        right_torque = self.calculate_torque(side="right")

        # Check if beam is unbalanced
        beam_unbalanced = self.is_unbalanced(left_torque, right_torque)

        if beam_unbalanced:
            # Current player loses
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, self.info
        else:
            # Switch to the other player
            self.current_player *= -1
            reward = -10  # As per instruction: -10 for valid move
            return self.board.copy(), reward, False, False, self.info

    def calculate_torque(self, side):
        pivot_position = 4  # Index 4 corresponds to position 5 (pivot)
        torque = 0
        if side == "left":
            # Positions 0 to 3 (positions 1 to 4)
            for i in range(0, 4):
                if self.board[i] != 0:
                    distance = pivot_position - i  # (5 - position number)
                    torque += distance * 1  # Weight is always 1
        elif side == "right":
            # Positions 5 to 8 (positions 6 to 9)
            for i in range(5, 9):
                if self.board[i] != 0:
                    distance = i - pivot_position  # (position number - 5)
                    torque += distance * 1  # Weight is always 1
        return torque

    def is_unbalanced(self, left_torque, right_torque):
        left_weights = any(self.board[0:4] != 0)
        right_weights = any(self.board[5:9] != 0)

        # Beam is unbalanced if torques are unequal and there are weights on both sides
        if left_weights and right_weights and left_torque != right_torque:
            return True
        else:
            return False

    def render(self):
        beam_str = "Beam State:\n"
        beam_str += "Positions: 1  2  3  4  [5]  6  7  8  9\n"
        beam_str += "Weights:   "
        for i in range(9):
            if i == 4:
                continue  # Skip pivot position
            if self.board[i] == 1:
                beam_str += " X "
            elif self.board[i] == -1:
                beam_str += " O "
            else:
                beam_str += " . "
        beam_str = beam_str[:19] + "[·]" + beam_str[19:]
        beam_str += "\n"
        return beam_str

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]
