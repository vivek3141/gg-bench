import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium.utils.seeding


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # The observation space is a 5x5 grid
        # Each cell can be:
        # -1: Removed cell ('X')
        #  0: Empty cell ('.')
        #  1: Player 1's position ('P1')
        #  2: Player 2's position ('P2')
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        # The action space is a discrete space of all possible move and remove combinations
        # Action encoding:
        # action = move_to_index * 25 + remove_cell_index
        # move_to_index and remove_cell_index are from 0 to 24 (since 5x5 grid)
        self.action_space = spaces.Discrete(625)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.done = False

        # Players decide who goes first (Player 1)
        self.current_player = 1  # 1 for Player 1, 2 for Player 2

        # Starting positions: randomly assign positions to P1 and P2
        available_positions = [(i, j) for i in range(5) for j in range(5)]
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        pos_p1 = self.np_random.choice(len(available_positions))
        self.p1_pos = available_positions.pop(pos_p1)

        pos_p2 = self.np_random.choice(len(available_positions))
        self.p2_pos = available_positions.pop(pos_p2)

        # Update the board with initial positions
        self.board[self.p1_pos] = 1
        self.board[self.p2_pos] = 2

        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Decode action into move and remove positions
        move_and_remove = self._decode_action(action)
        if move_and_remove is None:
            # Invalid action (out of bounds)
            reward = -10
            terminated = True
            return self.board.copy(), reward, terminated, truncated, info

        move_to, remove_cell = move_and_remove

        # Check if move is valid
        valid_moves = self._get_valid_moves(self.current_player)
        if move_to not in valid_moves:
            # Invalid move
            reward = -10
            terminated = True
            return self.board.copy(), reward, terminated, truncated, info

        # Check if removal is valid
        valid_removals = self._get_valid_removals(self.current_player)
        if remove_cell not in valid_removals:
            # Invalid removal
            reward = -10
            terminated = True
            return self.board.copy(), reward, terminated, truncated, info

        # Perform the move
        prev_pos = self.p1_pos if self.current_player == 1 else self.p2_pos
        self.board[prev_pos] = 0
        self.board[move_to] = self.current_player

        # Update player's position
        if self.current_player == 1:
            self.p1_pos = move_to
        else:
            self.p2_pos = move_to

        # Remove the cell
        self.board[remove_cell] = -1  # Mark as removed

        # Check if opponent has any valid moves
        opponent = 2 if self.current_player == 1 else 1
        opponent_valid_moves = self._get_valid_moves(opponent)
        if not opponent_valid_moves:
            # Current player wins
            reward = 1
            terminated = True
            return self.board.copy(), reward, terminated, truncated, info

        # Switch to the next player
        self.current_player = opponent

        return self.board.copy(), reward, terminated, truncated, info

    def render(self):
        board_str = ""
        for i in range(5):
            for j in range(5):
                cell = self.board[i, j]
                if cell == 0:
                    board_str += ". "
                elif cell == -1:
                    board_str += "X "
                elif cell == 1:
                    board_str += "P1 "
                elif cell == 2:
                    board_str += "P2 "
                else:
                    board_str += "? "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        # Return a list of valid action indices
        actions = []
        valid_moves = self._get_valid_moves(self.current_player)
        valid_removals = self._get_valid_removals(self.current_player)
        for move in valid_moves:
            for remove in valid_removals:
                action = self._encode_action(move, remove)
                actions.append(action)
        return actions

    def _get_valid_moves(self, player):
        # Returns a list of valid move positions (as tuples) for the current player
        pos = self.p1_pos if player == 1 else self.p2_pos
        possible_moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < 5 and 0 <= ny < 5:
                    if self.board[nx, ny] == 0:
                        possible_moves.append((nx, ny))
        return possible_moves

    def _get_valid_removals(self, player):
        # Returns a list of valid cells to remove (as tuples) based on the game rules
        opponent = 2 if player == 1 else 1
        opponent_pos = self.p1_pos if opponent == 1 else self.p2_pos

        removals = []
        for i in range(5):
            for j in range(5):
                if self.board[i, j] != 0:
                    continue  # Cannot remove non-empty cell
                # Cannot remove a cell adjacent to the opponent
                if abs(i - opponent_pos[0]) <= 1 and abs(j - opponent_pos[1]) <= 1:
                    continue
                removals.append((i, j))
        return removals

    def _encode_action(self, move_pos, remove_pos):
        # Encodes the move and removal positions into a single action index
        move_index = move_pos[0] * 5 + move_pos[1]
        remove_index = remove_pos[0] * 5 + remove_pos[1]
        action = move_index * 25 + remove_index
        return action

    def _decode_action(self, action):
        # Decodes the action index back into move and removal positions
        if action < 0 or action >= 625:
            return None  # Invalid action
        move_index = action // 25
        remove_index = action % 25
        move_row, move_col = divmod(move_index, 5)
        remove_row, remove_col = divmod(remove_index, 5)
        move_pos = (move_row, move_col)
        remove_pos = (remove_row, remove_col)
        return move_pos, remove_pos
