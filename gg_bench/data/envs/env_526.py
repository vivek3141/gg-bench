import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 6 possible moves + 1 pass action
        self.action_space = spaces.Discrete(7)

        # Observation space: positions of the 6 pieces
        # Positions range from -1 (captured) to 10
        self.observation_space = spaces.Box(low=-1, high=10, shape=(6,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Positions of pieces for Player 1 (indices 0-2): P1-A, P1-B, P1-C
        # Positions of pieces for Player 2 (indices 3-5): P2-A, P2-B, P2-C
        # Starting positions: Player 1 at positions 0,1,2; Player 2 at positions 10,9,8
        self.piece_positions = np.array([0, 1, 2, 10, 9, 8], dtype=np.int32)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return self.piece_positions.copy()

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        if action == 6:
            # Pass action
            if len(valid_actions) > 1:
                # Player has valid moves but chose to pass (invalid)
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                # Pass is valid as no valid moves are available
                # Switch the player
                self.current_player *= -1
                return self._get_obs(), 0, False, False, {}

        # Map action to piece and movement
        piece_idx = action // 2
        move_amount = 1 + (action % 2)  # 1 or 2 steps

        if self.current_player == 1:
            piece_indices = [0, 1, 2]
            direction = 1  # Move towards higher positions
        else:
            piece_indices = [3, 4, 5]
            direction = -1  # Move towards lower positions

        piece_global_idx = piece_indices[piece_idx]
        piece_position = self.piece_positions[piece_global_idx]

        if piece_position == -1:
            # Piece is captured
            self.done = True
            return self._get_obs(), -10, True, False, {}

        new_position = piece_position + direction * move_amount

        # Check board limits
        if new_position < 0 or new_position > 10:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check for collision with own pieces
        own_piece_positions = self.piece_positions[piece_indices]
        if new_position in own_piece_positions:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        opponent_piece_indices = [i for i in range(6) if i not in piece_indices]
        opponent_piece_positions = self.piece_positions[opponent_piece_indices]

        reward = 0
        terminated = False

        if new_position in opponent_piece_positions:
            # Capture opponent's piece
            opp_index = np.where(opponent_piece_positions == new_position)[0][0]
            opp_piece_idx = opponent_piece_indices[opp_index]
            self.piece_positions[opp_piece_idx] = -1  # Remove opponent's piece

            # Check for victory by capturing all opponent's pieces
            if all(self.piece_positions[opponent_piece_indices] == -1):
                reward = 1
                self.done = True
                terminated = True

        # Move the piece
        self.piece_positions[piece_global_idx] = new_position

        # Check for victory by crossing over
        if self.current_player == 1 and new_position >= 8:
            reward = 1
            self.done = True
            terminated = True
        elif self.current_player == -1 and new_position <= 2:
            reward = 1
            self.done = True
            terminated = True

        if not self.done:
            # Switch the player
            self.current_player *= -1

        return self._get_obs(), reward, self.done, False, {}

    def valid_moves(self):
        if self.done:
            return []

        valid_actions = []

        if self.current_player == 1:
            piece_indices = [0, 1, 2]
            direction = 1
        else:
            piece_indices = [3, 4, 5]
            direction = -1

        own_piece_positions = self.piece_positions[piece_indices]

        for action in range(6):
            piece_idx = action // 2
            move_amount = 1 + (action % 2)
            piece_global_idx = piece_indices[piece_idx]
            piece_position = self.piece_positions[piece_global_idx]

            if piece_position == -1:
                continue

            new_position = piece_position + direction * move_amount

            if new_position < 0 or new_position > 10:
                continue

            if new_position in own_piece_positions:
                continue

            # Action is valid
            valid_actions.append(action)

        if not valid_actions:
            valid_actions.append(6)  # Pass action

        return valid_actions

    def render(self):
        board = ["." for _ in range(11)]

        for idx, pos in enumerate(self.piece_positions):
            if pos == -1:
                continue  # Captured piece
            if idx <= 2:
                piece_label = f'P1-{"ABC"[idx]}'
            else:
                piece_label = f'P2-{"ABC"[idx - 3]}'
            board[pos] = piece_label

        board_str = ""
        for idx, cell in enumerate(board):
            board_str += f"{idx}: {cell}  "
            if idx == 10:
                board_str += "\n"
        return board_str
