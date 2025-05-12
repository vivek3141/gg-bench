import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: move forward 1, 2, or 3 positions (indices 0, 1, 2)
        self.action_space = spaces.Discrete(3)
        # Observation: positions of Player 1 and Player 2 on the ladder
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.array(
            [0, 0], dtype=np.int32
        )  # Positions of Player 1 and Player 2
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.positions.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.positions.copy(), 0, True, False, {}

        if action not in [0, 1, 2]:
            # Invalid action
            self.done = True
            return self.positions.copy(), -10, True, False, {}

        move = action + 1  # Map action indices to move steps (1, 2, or 3)

        player_idx = self.current_player - 1
        opponent_idx = 1 - player_idx
        player_pos = self.positions[player_idx]
        opponent_pos = self.positions[opponent_idx]

        intended_new_pos = player_pos + move

        # Movement rules:

        # 1. Cannot move beyond position 10
        if intended_new_pos > 10:
            self.done = True
            return self.positions.copy(), -10, True, False, {}

        # 2. Cannot land on or pass over the opponent if the opponent is ahead
        if player_pos < opponent_pos and intended_new_pos >= opponent_pos:
            self.done = True
            return self.positions.copy(), -10, True, False, {}

        # Valid move; update position
        self.positions[player_idx] = intended_new_pos

        # Check for win condition
        if intended_new_pos == 10:
            self.done = True
            return self.positions.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        return (
            self.positions.copy(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Visual representation of the number ladder and player positions
        ladder = ""
        for pos in range(11):
            token = ""
            if self.positions[0] == pos and self.positions[1] == pos:
                token = "[P1&P2]"
            elif self.positions[0] == pos:
                token = "[P1]"
            elif self.positions[1] == pos:
                token = "[P2]"
            else:
                token = "[   ]"
            ladder += f"{pos:2d}{token} "
        return ladder

    def valid_moves(self):
        # Return list of valid action indices for the current player
        moves = []
        player_idx = self.current_player - 1
        opponent_idx = 1 - player_idx
        player_pos = self.positions[player_idx]
        opponent_pos = self.positions[opponent_idx]

        for action in [0, 1, 2]:
            move = action + 1
            intended_new_pos = player_pos + move

            # Skip moves that exceed position 10
            if intended_new_pos > 10:
                continue

            # Skip moves that land on or pass over opponent if opponent is ahead
            if player_pos < opponent_pos and intended_new_pos >= opponent_pos:
                continue

            moves.append(action)

        return moves
