import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0, 1, 2 correspond to move distances 1, 2, 3
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0], dtype=np.float32
            ),  # current_marker_position, opponent_last_move
            high=np.array([15, 3], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_marker_position = 8
        self.current_player = 1  # Player 1 starts first
        self.terminated = False
        self.opponent_last_move = 0  # No move has been made yet
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [self.current_marker_position, self.opponent_last_move], dtype=np.float32
        )

    def valid_moves(self):
        valid_moves = []
        opponent_last_move = self.opponent_last_move  # 0 if first turn

        if self.current_player == 1:
            own_goal_position = 0
            remaining_distance = self.current_marker_position - own_goal_position
        else:
            own_goal_position = 15
            remaining_distance = own_goal_position - self.current_marker_position

        for action in range(3):  # Actions 0, 1, 2 correspond to move distances 1, 2, 3
            move_distance = action + 1
            if opponent_last_move != 0 and move_distance == opponent_last_move:
                continue  # Forbidden move
            if move_distance > remaining_distance:
                continue  # Cannot overshoot the goal
            valid_moves.append(action)

        return valid_moves

    def step(self, action):
        if self.terminated:
            return self._get_obs(), 0, True, False, {}

        # Map action to move distance
        move_distance = action + 1

        # Check if action is valid
        if action not in self.valid_moves():
            self.terminated = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        # Apply the move
        if self.current_player == 1:
            self.current_marker_position -= move_distance  # Move left towards 0
            self.current_marker_position = max(self.current_marker_position, 0)
            own_goal_position = 0
        else:
            self.current_marker_position += move_distance  # Move right towards 15
            self.current_marker_position = min(self.current_marker_position, 15)
            own_goal_position = 15

        # Check for win condition
        if self.current_marker_position == own_goal_position:
            self.terminated = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Update opponent's last move
        self.opponent_last_move = move_distance

        # Switch players
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_obs(), 0, False, False, {}

    def render(self):
        # Create a visual representation of the number line and marker
        positions = list(range(16))
        line = "|" + "".join(f"{i:2}|" for i in positions) + "\n"
        marker_line = "|"
        for i in positions:
            if i == int(self.current_marker_position):
                marker_line += " M|"
            else:
                marker_line += "  |"
        info = (
            f"Current Marker Position: {int(self.current_marker_position)}\n"
            f"Current Player: Player {self.current_player}\n"
            f"Opponent's Last Move: {int(self.opponent_last_move)}\n"
        )
        return line + marker_line + "\n" + info
