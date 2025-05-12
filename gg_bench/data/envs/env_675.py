import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to moving forward by 1, 2, or 3 nodes
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0 (move 1), 1 (move 2), 2 (move 3)
        # Observation space is an array representing the 11 nodes
        # Each node can be -1 (Player 2), 0 (empty), or 1 (Player 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(11, dtype=np.int8)
        self.board[0] = 1  # Player 1 starts at Node 0
        self.board[10] = -1  # Player 2 starts at Node 10
        self.player_positions = {1: 0, -1: 10}
        self.player_move_turns = {1: 0, -1: 0}
        self.current_player = 1  # Player 1 starts
        self.turn_number = 0
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}

        valid_actions = self.valid_moves()
        if not valid_actions:
            # No valid moves available; the game ends in loss for current player
            self.done = True
            return self.board.copy(), -10, True, False, {}

        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Get current player's position
        old_position = self.player_positions[self.current_player]

        # Determine move distance
        move_distance = (
            action + 1
        )  # Actions are 0, 1, 2 => Moves are 1, 2, 3 nodes forward

        # Calculate new position based on the player
        if self.current_player == 1:
            new_position = old_position + move_distance
        else:
            new_position = old_position - move_distance

        # Check for capture before moving
        if self.board[new_position] == -self.current_player:
            # Capture occurs
            self.done = True
            reward = 1  # Current player wins
            return self.board.copy(), reward, True, False, {}

        # Move the player
        self.board[old_position] = 0
        self.board[new_position] = self.current_player
        self.player_positions[self.current_player] = new_position
        self.player_move_turns[self.current_player] = self.turn_number

        # Increment turn number
        self.turn_number += 1

        # Switch to the next player
        self.current_player *= -1

        return (
            self.board.copy(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def render(self):
        board_str = ""
        for i in range(11):
            if self.board[i] == 1:
                board_str += "[P1]"
            elif self.board[i] == -1:
                board_str += "[P2]"
            else:
                board_str += "[   ]"
        board_str += "\nNodes: "
        for i in range(11):
            board_str += f"{i:^5}"
        return board_str

    def valid_moves(self):
        valid_actions = []
        current_position = self.player_positions[self.current_player]
        for action in range(3):
            move_distance = action + 1
            if self.current_player == 1:
                new_position = current_position + move_distance
                if new_position <= 10:
                    valid_actions.append(action)
            else:
                new_position = current_position - move_distance
                if new_position >= 0:
                    valid_actions.append(action)
        return valid_actions
