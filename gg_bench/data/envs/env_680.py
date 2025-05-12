import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: Bids for Player 1 and Player 2, each from 0 to 10
        self.action_space = spaces.MultiDiscrete([11, 11])

        # Observations: [Player 1 Points, Player 2 Points, Player 1 Tower Height, Player 2 Tower Height]
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_points = [10, 10]  # Points for Player 1 and Player 2
        self.player_towers = [0, 0]  # Tower heights for Player 1 and Player 2
        self.done = False
        observation = np.array(self.player_points + self.player_towers, dtype=np.int32)
        return observation, {}

    def step(self, action):
        bid_p1, bid_p2 = action

        # Check for invalid bids
        if (
            bid_p1 < 0
            or bid_p1 > self.player_points[0]
            or bid_p2 < 0
            or bid_p2 > self.player_points[1]
        ):
            # Invalid move
            reward = -10
            self.done = True
            observation = np.array(
                self.player_points + self.player_towers, dtype=np.int32
            )
            return observation, reward, True, False, {}

        # Deduct points
        self.player_points[0] -= bid_p1
        self.player_points[1] -= bid_p2

        # Resolve bids
        if bid_p1 > bid_p2:
            self.player_towers[0] += 1
        elif bid_p2 > bid_p1:
            self.player_towers[1] += 1
        # In case of a tie, no blocks are added

        # Check for victory conditions
        reward = 0
        if self.player_towers[0] >= 5:
            reward = 1
            self.done = True
        elif self.player_towers[1] >= 5:
            reward = 0
            self.done = True
        elif self.player_points[1] == 0 and self.player_points[0] > 0:
            # Opponent cannot bid further; Player 1 wins
            reward = 1
            self.done = True
        elif self.player_points[0] == 0 and self.player_points[1] > 0:
            # Player 1 cannot bid further; Player 2 wins
            reward = 0
            self.done = True

        observation = np.array(self.player_points + self.player_towers, dtype=np.int32)
        return observation, reward, self.done, False, {}

    def render(self):
        state_str = (
            f"Player 1: Points={self.player_points[0]}, Tower Height={self.player_towers[0]}\n"
            f"Player 2: Points={self.player_points[1]}, Tower Height={self.player_towers[1]}"
        )
        return state_str

    def valid_moves(self):
        # Valid bids for Player 1 and Player 2
        valid_bids_p1 = list(range(self.player_points[0] + 1))
        valid_bids_p2 = list(range(self.player_points[1] + 1))
        return valid_bids_p1, valid_bids_p2
