import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space is Discrete(6)
        # Actions: 0-5
        # 0: Move forward by 1
        # 1: Move backward by 1
        # 2: Move forward by 2
        # 3: Move backward by 2
        # 4: Move forward by 3
        # 5: Move backward by 3
        self.action_space = spaces.Discrete(6)

        # Observation space: positions of Player 1 and Player 2
        # Positions are integers from 1 to 20
        self.observation_space = spaces.Box(low=1, high=20, shape=(2,), dtype=np.int64)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = [1, 20]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return np.array(self.player_positions, dtype=np.int64), {}

    def _action_to_movement(self, action):
        if action == 0:
            return 1, "forward"
        elif action == 1:
            return 1, "backward"
        elif action == 2:
            return 2, "forward"
        elif action == 3:
            return 2, "backward"
        elif action == 4:
            return 3, "forward"
        elif action == 5:
            return 3, "backward"

    def valid_moves(self):
        valid_actions = []
        current_pos = self.player_positions[self.current_player]
        for action in range(self.action_space.n):
            distance, direction = self._action_to_movement(action)
            if direction == "forward":
                new_pos = current_pos + distance
            else:  # 'backward'
                new_pos = current_pos - distance

            if 1 <= new_pos <= 20:
                valid_actions.append(action)
        return valid_actions

    def step(self, action):
        if self.done:
            return np.array(self.player_positions, dtype=np.int64), 0, True, False, {}

        if action not in self.valid_moves():
            self.done = True
            return (
                np.array(self.player_positions, dtype=np.int64),
                -10,
                True,
                False,
                {},
            )

        # Apply action
        distance, direction = self._action_to_movement(action)
        current_pos = self.player_positions[self.current_player]

        if direction == "forward":
            new_pos = current_pos + distance
        else:  # 'backward'
            new_pos = current_pos - distance

        # Move the player
        self.player_positions[self.current_player] = new_pos

        # Check for capture
        if new_pos == self.player_positions[1 - self.current_player]:
            self.done = True
            return (
                np.array(self.player_positions, dtype=np.int64),
                1,
                True,
                False,
                {},
            )

        # Switch player
        self.current_player = 1 - self.current_player

        return np.array(self.player_positions, dtype=np.int64), 0, False, False, {}

    def render(self):
        track = ["-"] * 20
        p1_pos = self.player_positions[0] - 1
        p2_pos = self.player_positions[1] - 1

        if p1_pos == p2_pos:
            track[p1_pos] = "P1/P2"
        else:
            track[p1_pos] = "P1"
            track[p2_pos] = "P2"

        track_str = ""
        for i in range(20):
            pos_label = f"{i+1:2}"
            track_str += f"|{pos_label}:{track[i]:^5}"
        track_str += "|\n"

        return track_str
