import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Move Forward, 1 - Push Back Opponent
        self.action_space = spaces.Discrete(2)

        # Define observation space: An array representing the 9 nodes
        # Values: 0 - empty, 1 - Player 1's token, -1 - Player 2's token
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board: Player 1 on Node 1 (index 0), Player 2 on Node 9 (index 8)
        self.board = np.zeros(9, dtype=np.int8)
        self.p1_pos = 0
        self.p2_pos = 8
        self.board[self.p1_pos] = 1
        self.board[self.p2_pos] = -1

        # Player 1 starts first
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, no more moves can be taken
            return self.board.copy(), 0, True, False, {}

        reward = 0
        info = {}

        # Get the list of valid actions for the current player
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action attempted
            self.done = True
            reward = -10
            return self.board.copy(), reward, True, False, info

        # Process the action for the current player
        if self.current_player == 1:
            # Player 1's turn
            if action == 0:
                # Move forward towards Node 9
                new_pos = self.p1_pos + 1
                if new_pos > 8 or self.board[new_pos] != 0:
                    # Cannot move off the board or onto an occupied node
                    self.done = True
                    reward = -10
                    return self.board.copy(), reward, True, False, info
                # Move Player 1's token forward
                self.board[self.p1_pos] = 0
                self.p1_pos = new_pos
                self.board[self.p1_pos] = 1
            elif action == 1:
                # Push back Player 2
                if abs(self.p2_pos - self.p1_pos) != 1 or self.p2_pos == 8:
                    # Not adjacent or cannot push opponent off the board
                    self.done = True
                    reward = -10
                    return self.board.copy(), reward, True, False, info
                # Push Player 2's token back
                self.board[self.p2_pos] = 0
                self.p2_pos += 1
                self.board[self.p2_pos] = -1
        else:
            # Player 2's turn
            if action == 0:
                # Move forward towards Node 1
                new_pos = self.p2_pos - 1
                if new_pos < 0 or self.board[new_pos] != 0:
                    # Cannot move off the board or onto an occupied node
                    self.done = True
                    reward = -10
                    return self.board.copy(), reward, True, False, info
                # Move Player 2's token forward
                self.board[self.p2_pos] = 0
                self.p2_pos = new_pos
                self.board[self.p2_pos] = -1
            elif action == 1:
                # Push back Player 1
                if abs(self.p2_pos - self.p1_pos) != 1 or self.p1_pos == 0:
                    # Not adjacent or cannot push opponent off the board
                    self.done = True
                    reward = -10
                    return self.board.copy(), reward, True, False, info
                # Push Player 1's token back
                self.board[self.p1_pos] = 0
                self.p1_pos -= 1
                self.board[self.p1_pos] = 1

        # Check for victory condition
        if self.current_player == 1 and self.p1_pos == 8:
            # Player 1 wins
            self.done = True
            reward = 1
        elif self.current_player == -1 and self.p2_pos == 0:
            # Player 2 wins
            self.done = True
            reward = 1
        else:
            # Switch turns to the other player
            self.current_player *= -1

        return self.board.copy(), reward, self.done, False, info

    def render(self):
        # Return a string representation of the board
        board_str = "Board State:\n"
        for i in range(9):
            token = self.board[i]
            if token == 1:
                board_str += " P1 "
            elif token == -1:
                board_str += " P2 "
            else:
                board_str += f" {i+1} "
            if i < 8:
                board_str += "-"
        board_str += "\n"
        return board_str

    def valid_moves(self):
        # Return a list of valid actions for the current player
        valid_actions = []

        if self.done:
            return valid_actions

        if self.current_player == 1:
            # Player 1's possible actions
            # Move forward if the next node is within bounds and unoccupied
            new_pos = self.p1_pos + 1
            if new_pos <= 8 and self.board[new_pos] == 0:
                valid_actions.append(0)
            # Push back opponent if adjacent and opponent can be pushed back
            if abs(self.p2_pos - self.p1_pos) == 1 and self.p2_pos < 8:
                valid_actions.append(1)
        else:
            # Player 2's possible actions
            # Move forward if the next node is within bounds and unoccupied
            new_pos = self.p2_pos - 1
            if new_pos >= 0 and self.board[new_pos] == 0:
                valid_actions.append(0)
            # Push back opponent if adjacent and opponent can be pushed back
            if abs(self.p2_pos - self.p1_pos) == 1 and self.p1_pos > 0:
                valid_actions.append(1)

        return valid_actions
