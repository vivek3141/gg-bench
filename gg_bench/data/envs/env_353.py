import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space (25 possible coordinates)
        self.action_space = spaces.Discrete(25)

        # Define observation space
        # Observation will be a numpy array of shape (25 + 12,) = (37,)
        # First 25 entries: grid state (-1, 0, 1)
        # Next 3 entries: current player's target coordinate indices (0-24)
        # Next 3 entries: claimed status of current player's targets (0 or 1)
        # Next 3 entries: opponent's target coordinate indices (0-24)
        # Next 3 entries: claimed status of opponent's targets (0 or 1)
        self.observation_space = spaces.Box(low=-1, high=25, shape=(37,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros(
            25, dtype=np.int8
        )  # 0: unclaimed, 1: claimed by Player 1, -1: claimed by Player -1

        self.current_player = 1  # Player 1 starts
        self.done = False

        self.claimed_coordinates = set()

        # Randomly select target coordinates for both players
        self.p1_targets = np.random.choice(25, size=3, replace=False)
        self.p1_targets_claimed = np.zeros(
            3, dtype=np.int8
        )  # 0: not claimed, 1: claimed

        # Ensure opponent's targets do not overlap with player's targets
        remaining_coords = list(set(range(25)) - set(self.p1_targets))
        self.p2_targets = np.random.choice(remaining_coords, size=3, replace=False)
        self.p2_targets_claimed = np.zeros(3, dtype=np.int8)

        return self._get_observation(), {}  # Return observation and info

    def _get_observation(self):
        obs = np.concatenate(
            (
                self.grid,
                self.p1_targets,
                self.p1_targets_claimed,
                self.p2_targets,
                self.p2_targets_claimed,
            )
        ).astype(np.int8)
        return obs

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                self.done,
                False,
                {},
            )  # Already done, no reward

        if action < 0 or action >= 25:
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid action, game over

        if self.grid[action] != 0:
            # Coordinate already claimed
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid move, game over

        # Proceed with the action
        reward = 0
        info = {}
        selected_coordinate = action

        # Mark the coordinate as claimed by current player
        self.grid[selected_coordinate] = self.current_player
        self.claimed_coordinates.add(selected_coordinate)

        # Check if the coordinate is on current player's target list
        if selected_coordinate in self._get_current_player_targets():
            # Claim it
            self._claim_target(self.current_player, selected_coordinate)
            # Announce
            # Can print or log if desired
            # print("Player {} has claimed one of their target coordinates.".format(self.current_player))

        # Check if the coordinate is on opponent's target list
        if selected_coordinate in self._get_opponent_player_targets():
            # Opponent reveals
            # print("Player {} reveals: 'That coordinate is on my target list.'".format(-self.current_player))
            # Remove from opponent's target list and select a new one
            self._replace_target(-self.current_player, selected_coordinate)

        # Check if current player has claimed all their targets
        if self._check_victory(self.current_player):
            self.done = True
            reward = 1
            return self._get_observation(), reward, self.done, False, info

        # Switch player
        self.current_player *= -1

        return self._get_observation(), reward, self.done, False, info

    def render(self):
        # Return a string representing the current grid state
        grid_str = ""
        for y in range(5, 0, -1):
            row_str = ""
            for x in range(1, 6):
                idx = (y - 1) * 5 + (x - 1)
                cell = self.grid[idx]
                if cell == 1:
                    row_str += " X "
                elif cell == -1:
                    row_str += " O "
                else:
                    row_str += " . "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        return [i for i in range(25) if self.grid[i] == 0]

    def _get_current_player_targets(self):
        if self.current_player == 1:
            return set(self.p1_targets)
        else:
            return set(self.p2_targets)

    def _get_opponent_player_targets(self):
        if self.current_player == 1:
            return set(self.p2_targets)
        else:
            return set(self.p1_targets)

    def _claim_target(self, player, coordinate):
        if player == 1:
            idx = np.where(self.p1_targets == coordinate)[0][0]
            self.p1_targets_claimed[idx] = 1
        else:
            idx = np.where(self.p2_targets == coordinate)[0][0]
            self.p2_targets_claimed[idx] = 1

    def _replace_target(self, player, coordinate):
        if player == 1:
            idx = np.where(self.p1_targets == coordinate)[0][0]
            unclaimed_coords = list(
                set(range(25)) - self.claimed_coordinates - set(self.p1_targets)
            )
            if unclaimed_coords:
                new_target = np.random.choice(unclaimed_coords)
                self.p1_targets[idx] = new_target
                self.p1_targets_claimed[idx] = 0
            else:
                # No more unclaimed coordinates to choose
                pass  # Leave the target as is, but claimed
        else:
            idx = np.where(self.p2_targets == coordinate)[0][0]
            unclaimed_coords = list(
                set(range(25)) - self.claimed_coordinates - set(self.p2_targets)
            )
            if unclaimed_coords:
                new_target = np.random.choice(unclaimed_coords)
                self.p2_targets[idx] = new_target
                self.p2_targets_claimed[idx] = 0
            else:
                # No more unclaimed coordinates to choose
                pass  # Leave the target as is, but claimed

    def _check_victory(self, player):
        if player == 1:
            return np.all(self.p1_targets_claimed == 1)
        else:
            return np.all(self.p2_targets_claimed == 1)
