import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: move forward 1 or 2 cells (actions 0 and 1)
        self.action_space = spaces.Discrete(2)

        # Define observation space: track positions (integers 0 to 5 for each cell)
        self.observation_space = spaces.Box(low=0, high=5, shape=(5,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the track: positions 0 to 4 represent cells 1 to 5
        # 0: empty, 1: P1, 2: P2, 3: F, 4: P1/F, 5: P2/F
        self.track = np.zeros(5, dtype=np.int8)
        self.track[0] = 1  # Player 1 at cell 1
        self.track[2] = 3  # Flag at cell 3
        self.track[4] = 2  # Player 2 at cell 5

        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action not in [0, 1]:
            return self._get_obs(), -10, True, False, {}

        move_distance = action + 1  # Map action 0 to 1, action 1 to 2

        player_token = self.current_player
        opponent_token = 3 - self.current_player
        player_flag_token = self.current_player + 3

        player_pos = np.where(
            (self.track == player_token) | (self.track == player_flag_token)
        )[0][0]

        if self.current_player == 1:
            desired_pos = min(player_pos + move_distance, 2)
        else:
            desired_pos = max(player_pos - move_distance, 2)

        opponent_positions = np.where(
            (self.track == opponent_token) | (self.track == opponent_token + 3)
        )[0]
        if opponent_positions.size > 0:
            opponent_pos = opponent_positions[0]
            if self.current_player == 1 and desired_pos >= opponent_pos:
                desired_pos = opponent_pos - 1
            if self.current_player == 2 and desired_pos <= opponent_pos:
                desired_pos = opponent_pos + 1

        actual_move = (
            desired_pos - player_pos
            if self.current_player == 1
            else player_pos - desired_pos
        )
        if actual_move <= 0:
            self.current_player = opponent_token
            return self._get_obs(), 0, False, False, {}

        if self.track[player_pos] == player_flag_token:
            self.track[player_pos] = 3
        else:
            self.track[player_pos] = 0

        if self.track[desired_pos] == 3:
            self.track[desired_pos] = player_flag_token
            self.done = True
            return self._get_obs(), 1, True, False, {}

        self.track[desired_pos] = player_token
        self.current_player = opponent_token
        return self._get_obs(), 0, False, False, {}

    def render(self):
        cell_labels = {0: " ", 1: "P1", 2: "P2", 3: "F", 4: "P1/F", 5: "P2/F"}
        track_display = [cell_labels[cell] for cell in self.track]
        cell_numbers = ["[{}]".format(i + 1) for i in range(5)]
        track_str = "".join(cell_numbers) + "\n[" + "][".join(track_display) + "]"
        return track_str

    def valid_moves(self):
        valid_actions = []
        for action in [0, 1]:
            move_distance = action + 1

            player_token = self.current_player
            opponent_token = 3 - self.current_player
            player_flag_token = self.current_player + 3

            player_pos = np.where(
                (self.track == player_token) | (self.track == player_flag_token)
            )[0][0]

            if self.current_player == 1:
                desired_pos = min(player_pos + move_distance, 2)
            else:
                desired_pos = max(player_pos - move_distance, 2)

            opponent_positions = np.where(
                (self.track == opponent_token) | (self.track == opponent_token + 3)
            )[0]
            if opponent_positions.size > 0:
                opponent_pos = opponent_positions[0]
                if self.current_player == 1 and desired_pos >= opponent_pos:
                    desired_pos = opponent_pos - 1
                if self.current_player == 2 and desired_pos <= opponent_pos:
                    desired_pos = opponent_pos + 1

            actual_move = (
                desired_pos - player_pos
                if self.current_player == 1
                else player_pos - desired_pos
            )
            if actual_move > 0:
                valid_actions.append(action)

        return valid_actions

    def _get_obs(self):
        return self.track.copy()
