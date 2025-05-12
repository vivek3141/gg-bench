import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Actions: move forward by 1, 2, or 3 positions
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0 (move 1), 1 (move 2), 2 (move 3)

        # Observation space: positions of both players
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = [0, 0]  # Player A and Player B positions
        self.current_player = 0  # 0 for Player A, 1 for Player B
        self.done = False
        return np.array(self.player_positions), {}

    def step(self, action):
        if self.done:
            return np.array(self.player_positions), -10, True, False, {}

        # Map action to movement (action 0->move 1, action1->move2, action2->move3)
        move = action + 1

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return np.array(self.player_positions), -10, True, False, {}

        # Get current positions
        current_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[1 - self.current_player]

        # Calculate new position
        new_pos = current_pos + move

        # Move is valid, update position
        self.player_positions[self.current_player] = new_pos

        # Check for victory
        if new_pos == 10:
            self.done = True
            reward = 1
            return np.array(self.player_positions), reward, True, False, {}
        else:
            # Valid move, but not winning
            reward = -10

        # Switch player
        self.current_player = 1 - self.current_player

        return np.array(self.player_positions), reward, False, False, {}

    def render(self):
        number_line = ""
        for i in range(11):  # Positions 0 to 10
            markers = []
            if self.player_positions[0] == i and self.player_positions[1] == i:
                marker_str = "AB"
            elif self.player_positions[0] == i:
                marker_str = "A"
            elif self.player_positions[1] == i:
                marker_str = "B"
            else:
                marker_str = str(i)
            number_line += f"{marker_str:^4}"
        return number_line

    def valid_moves(self):
        valid_actions = []
        current_pos = self.player_positions[self.current_player]
        opponent_pos = self.player_positions[1 - self.current_player]

        for action in range(3):
            move = action + 1
            new_pos = current_pos + move

            # Can't move beyond position 10
            if new_pos > 10:
                continue

            # Can't move onto or past opponent's position
            if current_pos < opponent_pos <= new_pos:
                continue

            valid_actions.append(action)

        return valid_actions
