import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - move 1 step, 1 - move 2 steps
        self.action_space = spaces.Discrete(2)

        # Define observation space: positions of Player 1 and Player 2
        # Positions range from 0 to 10
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Tokens start at position 5
        self.positions = np.array([5, 5], dtype=np.int32)

        # Player 1 starts (represented by 1), Player 2 is -1
        self.current_player = 1  # 1 for Player 1, -1 for Player 2

        self.done = False

        return self.positions.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game already over
            return self.positions.copy(), -10, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return self.positions.copy(), -10, True, False, {}

        # Map action to steps: 0 => 1 step, 1 => 2 steps
        steps = action + 1

        # Get current and opponent's positions
        if self.current_player == 1:
            player_idx = 0
            opponent_idx = 1
            direction = 1  # Player 1 moves toward higher numbers
            opponent_base = 10
        else:
            player_idx = 1
            opponent_idx = 0
            direction = -1  # Player 2 moves toward lower numbers
            opponent_base = 0

        # Calculate the new position
        new_position = self.positions[player_idx] + direction * steps

        # Check if new position is within limits
        if new_position < 0 or new_position > 10:
            self.done = True
            return self.positions.copy(), -10, True, False, {}

        # Check if new position is occupied by opponent
        if new_position == self.positions[opponent_idx]:
            self.done = True
            return self.positions.copy(), -10, True, False, {}

        # Move is valid, update the position
        self.positions[player_idx] = new_position

        # Check for victory condition
        if self.positions[player_idx] == opponent_base:
            # Current player wins
            self.done = True
            return self.positions.copy(), 1, True, False, {}

        # Switch current player
        self.current_player *= -1

        return self.positions.copy(), 0, False, False, {}

    def render(self):
        # Create a visual representation of the number line
        number_line = ["."] * 11  # Positions from 0 to 10
        # Place tokens on the number line
        p1_pos = self.positions[0]
        p2_pos = self.positions[1]

        for i in range(0, 11):
            if i == p1_pos and i == p2_pos:
                number_line[i] = "B"  # Both players (should not happen in this game)
            elif i == p1_pos:
                number_line[i] = "1"
            elif i == p2_pos:
                number_line[i] = "2"

        line_str = " ".join(number_line)
        return f"Number Line: {line_str}\nPlayer 1 at position {p1_pos}, Player 2 at position {p2_pos}"

    def valid_moves(self):
        # Returns list of valid actions (indices of action_space) for the current player
        valid_actions = []
        for action in [0, 1]:
            steps = action + 1

            if self.current_player == 1:
                player_idx = 0
                opponent_idx = 1
                direction = 1
            else:
                player_idx = 1
                opponent_idx = 0
                direction = -1

            new_position = self.positions[player_idx] + direction * steps

            # Check if new position is within limits
            if new_position < 0 or new_position > 10:
                continue  # Invalid move, out of bounds

            # Check if new position is occupied by opponent
            if new_position == self.positions[opponent_idx]:
                continue  # Invalid move, occupied

            valid_actions.append(action)

        return valid_actions
