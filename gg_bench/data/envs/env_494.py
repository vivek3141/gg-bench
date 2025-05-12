import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple two-player game called Trail Blazer.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # The action_space is Discrete(3): move forward 1, 2, or 3 positions
        self.action_space = spaces.Discrete(3)

        # The observation_space consists of:
        # - positions[0]: Player 1's position (0 to 10)
        # - positions[1]: Player 2's position (0 to 10)
        # - current_player: 1 or 2
        low = np.array([0, 0, 1], dtype=np.int32)
        high = np.array([10, 10, 2], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.array(
            [0, 0], dtype=np.int32
        )  # Positions of Player 1 and Player 2
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array(
            [self.positions[0], self.positions[1], self.current_player], dtype=np.int32
        )
        return observation, {}

    def step(self, action):
        if self.done:
            observation = np.array(
                [self.positions[0], self.positions[1], self.current_player],
                dtype=np.int32,
            )
            return observation, 0, True, False, {}

        valid_actions = self.valid_moves()

        if not valid_actions:
            # No valid moves available, player cannot move and loses the game
            self.done = True
            observation = np.array(
                [self.positions[0], self.positions[1], self.current_player],
                dtype=np.int32,
            )
            return observation, -10, True, False, {}

        if action not in valid_actions:
            # Invalid action attempted
            self.done = True
            observation = np.array(
                [self.positions[0], self.positions[1], self.current_player],
                dtype=np.int32,
            )
            return observation, -10, True, False, {}

        # Valid action
        move_distance = action + 1  # Map action 0->move 1, 1->move 2, 2->move 3
        current_idx = self.current_player - 1
        opponent_idx = 1 - current_idx

        # Update position
        new_position = self.positions[current_idx] + move_distance
        self.positions[current_idx] = new_position

        # Check for victory
        if new_position >= 10:
            # Current player wins
            self.done = True
            observation = np.array(
                [self.positions[0], self.positions[1], self.current_player],
                dtype=np.int32,
            )
            return observation, 1, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1
        observation = np.array(
            [self.positions[0], self.positions[1], self.current_player], dtype=np.int32
        )
        return observation, 0, False, False, {}

    def render(self):
        track = ["_"] * 11  # Positions 0 to 10
        pos_p1 = min(self.positions[0], 10)
        pos_p2 = min(self.positions[1], 10)

        # Mark player positions
        if pos_p1 == pos_p2:
            track[pos_p1] = "X"  # Both players on the same spot (should not happen)
        else:
            track[pos_p1] = "P1"
            track[pos_p2] = "P2"

        track_repr = "Track: " + "  ".join([f"{i:2}" for i in range(11)]) + "\n"
        track_repr += "       " + "  ".join(f"{s:^2}" for s in track) + "\n"
        track_repr += f"Current Player: Player {self.current_player}"
        return track_repr

    def valid_moves(self):
        # Returns a list of valid action indices (0, 1, 2 corresponding to move 1, 2, 3 positions)
        current_idx = self.current_player - 1
        opponent_idx = 1 - current_idx
        current_pos = self.positions[current_idx]
        opponent_pos = self.positions[opponent_idx]

        valid_actions = []
        for i, move_distance in enumerate([1, 2, 3]):
            new_position = current_pos + move_distance
            if new_position != opponent_pos:
                valid_actions.append(i)
        return valid_actions
