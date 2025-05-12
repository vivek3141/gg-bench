import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (move 1 position forward), 1 (move 2 positions forward)
        self.action_space = spaces.Discrete(2)

        # Observations: [Player 1 position, Player 2 position, Current player]
        # Positions are integers from 1 to 7
        # Current player: 1 or 2
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 1]), high=np.array([7, 7, 2]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set initial positions
        self.p1_position = 1
        self.p2_position = 7

        # Player 1 starts
        self.current_player = 1

        # Game not done
        self.done = False

        # Return initial observation and info
        observation = np.array(
            [self.p1_position, self.p2_position, self.current_player], dtype=np.int32
        )
        return observation, {}

    def step(self, action):
        # Check if game is already done
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Map action to move distance (0 -> 1, 1 -> 2)
        move_distance = action + 1

        # Get current positions
        current_pos = self.p1_position if self.current_player == 1 else self.p2_position
        opponent_pos = (
            self.p2_position if self.current_player == 1 else self.p1_position
        )

        # Determine movement direction
        direction = 1 if self.current_player == 1 else -1

        # Calculate new position
        new_position = current_pos + move_distance * direction

        # Check for invalid move (move off the board)
        if new_position < 1 or new_position > 7:
            self.done = True
            reward = -10.0
            return self._get_obs(), reward, True, False, {}

        # Check if move skips over opponent
        positions_between = range(
            min(current_pos, new_position) + 1, max(current_pos, new_position)
        )
        if opponent_pos in positions_between:
            self.done = True
            reward = -10.0
            return self._get_obs(), reward, True, False, {}

        # Check for capture
        if new_position == opponent_pos:
            # Current player wins
            self.done = True
            if self.current_player == 1:
                self.p1_position = new_position
            else:
                self.p2_position = new_position
            reward = 1.0
            return self._get_obs(), reward, True, False, {}

        # Valid move; update position
        if self.current_player == 1:
            self.p1_position = new_position
        else:
            self.p2_position = new_position

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Continue game
        reward = 0.0
        return self._get_obs(), reward, False, False, {}

    def render(self):
        board = ["[ ]"] * 7
        board_positions = [" (1)", " (2)", " (3)", " (4)", " (5)", " (6)", " (7)"]

        # Place player tokens
        board[self.p1_position - 1] = "[P1]"
        board[self.p2_position - 1] = "[P2]"

        # Build the board string
        board_str = " ".join(board) + "\n" + " ".join(board_positions)
        return board_str

    def valid_moves(self):
        # Get current positions
        current_pos = self.p1_position if self.current_player == 1 else self.p2_position
        opponent_pos = (
            self.p2_position if self.current_player == 1 else self.p1_position
        )

        # Determine movement direction
        direction = 1 if self.current_player == 1 else -1

        valid_actions = []
        for action in [0, 1]:  # Actions 0 (move 1), 1 (move 2)
            move_distance = action + 1
            new_position = current_pos + move_distance * direction

            # Check for move off the board
            if new_position < 1 or new_position > 7:
                continue

            # Check if move skips over opponent
            positions_between = range(
                min(current_pos, new_position) + 1, max(current_pos, new_position)
            )
            if opponent_pos in positions_between:
                continue

            valid_actions.append(action)

        return valid_actions

    def _get_obs(self):
        observation = np.array(
            [self.p1_position, self.p2_position, self.current_player], dtype=np.int32
        )
        return observation
