import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - strike, 1 - shield, 2 - drain
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # Observation consists of:
        # [current_player,
        #  Player 1 Shadow Counter,
        #  Player 2 Shadow Counter,
        #  Player 1 Last Action (-1 if none),
        #  Player 2 Last Action (-1 if none),
        #  Player 1 Shield Active (0 or 1),
        #  Player 2 Shield Active (0 or 1)]
        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0, -1, -1, 0, 0]),
            high=np.array([2, 10, 10, 2, 2, 1, 1]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1
        self.player_counters = [10, 10]  # [Player 1, Player 2]
        self.last_actions = [-1, -1]  # [-1 indicates no action taken yet]
        self.shields_active = [0, 0]  # Shields active for [Player 1, Player 2]
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        if action not in self.valid_moves():
            self.done = True
            return self._get_observation(), -10, True, False, {}

        opponent = 1 if self.current_player == 2 else 2
        player_idx = self.current_player - 1
        opponent_idx = opponent - 1

        # Process action
        if action == 0:  # Strike
            self.last_actions[player_idx] = 0
            # Check if opponent's shield is active
            if self.shields_active[opponent_idx] == 0:
                self.player_counters[opponent_idx] -= 1
                if self.player_counters[opponent_idx] < 0:
                    self.player_counters[opponent_idx] = 0
        elif action == 1:  # Shield
            self.last_actions[player_idx] = 1
            self.shields_active[player_idx] = (
                1  # Shield will be active during opponent's next turn
            )
        elif action == 2:  # Drain
            self.last_actions[player_idx] = 2
            if self.player_counters[player_idx] >= 1:
                self.player_counters[player_idx] -= 1
            else:
                self.player_counters[player_idx] = 0
            # Check if opponent's shield is active
            if self.shields_active[opponent_idx] == 0:
                self.player_counters[opponent_idx] -= 2
                if self.player_counters[opponent_idx] < 0:
                    self.player_counters[opponent_idx] = 0

        # Reset opponent's shield if it was active
        if self.shields_active[opponent_idx] == 1:
            self.shields_active[opponent_idx] = 0  # Shield duration is over

        # Check for victory condition
        if self.player_counters[opponent_idx] == 0:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch current player
        self.current_player = opponent

        return self._get_observation(), 0, False, False, {}

    def render(self):
        return (
            f"Player 1 Shadow Counter: {self.player_counters[0]}\n"
            f"Player 2 Shadow Counter: {self.player_counters[1]}\n"
            f"Player 1 Shield Active: {'Yes' if self.shields_active[0] else 'No'}\n"
            f"Player 2 Shield Active: {'Yes' if self.shields_active[1] else 'No'}\n"
            f"Current Turn: Player {self.current_player}"
        )

    def valid_moves(self):
        player_idx = self.current_player - 1
        valid_actions = []

        # Strike is always valid
        valid_actions.append(0)

        # Shield can be used if it wasn't used on the player's last turn
        if self.last_actions[player_idx] != 1:
            valid_actions.append(1)

        # Drain can be used if:
        # - Player's Shadow Counter is 2 or higher
        # - Player did not use Drain on their last turn
        if self.player_counters[player_idx] >= 2 and self.last_actions[player_idx] != 2:
            valid_actions.append(2)

        return valid_actions

    def _get_observation(self):
        return np.array(
            [
                self.current_player,
                self.player_counters[0],
                self.player_counters[1],
                self.last_actions[0],
                self.last_actions[1],
                self.shields_active[0],
                self.shields_active[1],
            ],
            dtype=np.int32,
        )
