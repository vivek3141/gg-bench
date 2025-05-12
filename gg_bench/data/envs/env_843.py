import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - move left (-1), 1 - move right (+1)
        self.action_space = spaces.Discrete(2)

        # Define observation space
        # Observation consists of:
        # - Positions 0-10: board cells, values -1 (opponent), 0 (empty), 1 (current player)
        # - Position 11: Inversion state (0 or 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board: positions 0-10
        self.board = np.zeros(11, dtype=np.int32)

        # Initialize players' positions
        self.positions = {1: 5, -1: 5}  # Player 1 and Player -1 start at position 5
        self.current_player = 1  # Start with Player 1

        # Place the players' markers on the board
        self.board[self.positions[1]] = 1
        self.board[self.positions[-1]] = -1

        # Initialize inversion state and game over flag
        self.inversion = False
        self.done = False

        # Return the initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Map action to movement
        move = -1 if action == 0 else 1

        # Update the board
        prev_position = self.positions[self.current_player]
        new_position = prev_position + move

        # Remove current player's marker from previous position
        self.board[prev_position] = 0

        # Place marker at the new position
        self.positions[self.current_player] = new_position
        self.board[new_position] = self.current_player

        # Check for inversion
        if not self.inversion:
            opponent_position = self.positions[-self.current_player]
            if (prev_position - opponent_position) * (
                new_position - opponent_position
            ) < 0:
                self.inversion = True

        # Check for win
        goal = 10 if self.inversion else 0
        if self.positions[self.current_player] == goal:
            self.done = True
            reward = 1
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Switch current player
        self.current_player *= -1

        # No reward for a valid move that doesn't win the game
        reward = 0

        observation = self._get_observation()
        return observation, reward, False, False, {}

    def render(self):
        board_str = ""
        for i in range(11):
            cell = self.board[i]
            if cell == 1:
                board_str += " A "
            elif cell == -1:
                board_str += " B "
            else:
                board_str += " . "
        inversion_state = "Yes" if self.inversion else "No"
        goal = "10" if self.inversion else "0"
        render_str = (
            f"Cells:   0  1  2  3  4  5  6  7  8  9 10\n"
            f"Markers:{board_str}\n"
            f"Goal: Cell {goal}\n"
            f"Inversion: {inversion_state}\n"
            f"Current Player: {'A' if self.current_player == 1 else 'B'}\n"
        )
        return render_str

    def valid_moves(self):
        moves = []
        current_pos = self.positions[self.current_player]
        opponent_pos = self.positions[-self.current_player]

        # Check move left (-1)
        left_pos = current_pos - 1
        if left_pos >= 0 and left_pos != opponent_pos:
            moves.append(0)  # Action 0 corresponds to move left

        # Check move right (+1)
        right_pos = current_pos + 1
        if right_pos <= 10 and right_pos != opponent_pos:
            moves.append(1)  # Action 1 corresponds to move right

        return moves

    def _get_observation(self):
        # Create a view of the board from the current player's perspective
        observation = np.zeros(12, dtype=np.int32)
        for i in range(11):
            if self.board[i] == self.current_player:
                observation[i] = 1  # Current player's marker
            elif self.board[i] == -self.current_player:
                observation[i] = -1  # Opponent's marker
            else:
                observation[i] = 0  # Empty cell
        observation[11] = int(self.inversion)  # Inversion state
        return observation
