import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Movement options corresponding to powers of two
        self.move_values = [1, 2, 4, 8, 16]
        self.action_space = spaces.Discrete(len(self.move_values))

        # Observation space: positions of both players
        self.observation_space = spaces.Box(low=0, high=31, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = [0, 0]  # Positions for Player 1 and Player 2
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return np.array(self.positions, dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.array(self.positions, dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if action not in [0, 1, 2, 3, 4]:
            reward = -10
            self.done = True
            return (
                np.array(self.positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        move = self.move_values[action]
        current_position = self.positions[self.current_player]
        new_position = current_position + move

        # Check if move is valid
        if not self.is_valid_move(current_position, move):
            reward = -10
            self.done = True
            return (
                np.array(self.positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Update position
        self.positions[self.current_player] = new_position

        # Check for win condition
        if new_position >= 31:
            reward = 1
            self.done = True
            return (
                np.array(self.positions, dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # No reward for regular move
        reward = 0

        # Switch to the other player
        self.current_player = 1 - self.current_player

        return np.array(self.positions, dtype=np.int32), reward, False, False, {}

    def is_valid_move(self, current_position, move):
        new_position = current_position + move

        # Legal Moves: You cannot move in a way that would bypass position 31 without landing on or exceeding it exactly.
        if current_position < 31 and new_position > 31:
            return False

        return True

    def render(self):
        render_str = f"Player 1 Position: {self.positions[0]}\n"
        render_str += f"Player 2 Position: {self.positions[1]}\n"
        render_str += f"Current Player: {'Player 1' if self.current_player == 0 else 'Player 2'}\n"
        return render_str

    def valid_moves(self):
        current_position = self.positions[self.current_player]
        valid_actions = []
        for idx, move in enumerate(self.move_values):
            if self.is_valid_move(current_position, move):
                valid_actions.append(idx)
        return valid_actions
