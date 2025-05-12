import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 to 5, representing moves of 1 to 6 nodes
        self.action_space = spaces.Discrete(6)

        # Observation space: 11 nodes, -1: opponent's token, 0: empty, 1: current player's token
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False

        # Initialize the board
        self.board = np.zeros(11, dtype=np.int8)

        # Starting positions
        # Player 1 starts on Node 1 (index 0)
        # Player 2 starts on Node 11 (index 10)
        self.positions = {1: 0, -1: 10}  # Player 1's position  # Player 2's position

        self.board[self.positions[1]] = 1
        self.board[self.positions[-1]] = -1

        # Player 1 starts the game
        self.current_player = 1

        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Map action to move distance (actions are 0-5, representing moves of 1-6)
        move_distance = action + 1

        curr_pos = self.positions[self.current_player]
        opp_pos = self.positions[-self.current_player]

        # Calculate maximum move distance based on the rules
        if self.current_player == 1:
            delta_pos = opp_pos - curr_pos - 1
        else:
            delta_pos = curr_pos - opp_pos - 1

        max_distance = min(6, delta_pos) if delta_pos > 0 else 0

        # Check if the player is blocked and must pass
        if max_distance == 0:
            # Player must pass their turn
            self.current_player *= -1
            return self.board.copy(), 0, False, False, {}

        # Validate the action
        if move_distance < 1 or move_distance > max_distance:
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Move the token
        self.board[curr_pos] = 0  # Remove token from current position

        if self.current_player == 1:
            new_pos = curr_pos + move_distance
        else:
            new_pos = curr_pos - move_distance

        # Check for capturing the opponent's token
        if new_pos == opp_pos:
            self.board[new_pos] = self.current_player
            self.positions[self.current_player] = new_pos
            self.board[opp_pos] = 0  # Remove opponent's token
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check for reaching the opponent's home node
        if (self.current_player == 1 and new_pos == 10) or (
            self.current_player == -1 and new_pos == 0
        ):
            self.board[new_pos] = self.current_player
            self.positions[self.current_player] = new_pos
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Normal movement
        self.board[new_pos] = self.current_player
        self.positions[self.current_player] = new_pos

        # Switch to the next player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = "Board State:\n"
        for i in range(11):
            node = self.board[i]
            if node == 1:
                board_str += " P "  # Current player's token
            elif node == -1:
                board_str += " O "  # Opponent's token
            else:
                board_str += " - "
        board_str += "\n"
        return board_str

    def valid_moves(self):
        curr_pos = self.positions[self.current_player]
        opp_pos = self.positions[-self.current_player]

        # Calculate maximum move distance
        if self.current_player == 1:
            delta_pos = opp_pos - curr_pos - 1
        else:
            delta_pos = curr_pos - opp_pos - 1

        max_distance = min(6, delta_pos) if delta_pos > 0 else 0

        # Return list of valid action indices
        valid_actions = [i for i in range(max_distance)]
        return valid_actions
