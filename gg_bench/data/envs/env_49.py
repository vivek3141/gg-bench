import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 -> move 1, 1 -> move 2, 2 -> move 3
        self.action_space = spaces.Discrete(3)

        # Observation space: positions of Player A and Player B, and current player indicator
        # Positions range from 0 to 20, current player indicator is -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1]), high=np.array([20, 20, 1]), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Both players start at position 0
        self.player_positions = {1: 0, -1: 0}  # 1 for Player A, -1 for Player B
        self.current_player = 1  # Player A starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game is over

        if action not in self.valid_moves():
            # Invalid move attempted
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {},
            )  # Game over due to invalid move

        move_distance = action + 1  # Map action to move distance (1, 2, or 3)
        opponent = -self.current_player
        opponent_pos = self.player_positions[opponent]
        current_pos = self.player_positions[self.current_player]

        # Desired new position
        desired_position = current_pos + move_distance

        # Handle blocking
        if desired_position == opponent_pos:
            # Must stop at the nearest unoccupied position behind opponent
            # Which is current position up to opponent position - 1
            blocked_position = opponent_pos - 1
            if blocked_position <= current_pos:
                # No forward movement possible
                desired_position = current_pos
            else:
                desired_position = blocked_position
        elif desired_position > opponent_pos:
            # If the path crosses over the opponent, need to check for occupation
            # In this game, only the landing position matters (no need to check intermediary positions)
            pass  # No action needed
        # No need to check for multiple occupied positions since only one opponent

        # Ensure desired_position does not exceed 20 (positions go up to 20)
        if desired_position > 20:
            desired_position = desired_position  # Allowed to surpass position 20 to win

        # Update player's position
        self.player_positions[self.current_player] = desired_position

        # Check for win condition
        if desired_position >= 20:
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Switch to the other player
        self.current_player *= -1
        return self._get_obs(), 0, False, False, {}  # Continue game

    def render(self):
        # Build the line representation
        line = ["."] * 21  # Positions 0 to 20
        for pos in range(21):
            if self.player_positions[1] == pos and self.player_positions[-1] == pos:
                line[pos] = "X"  # Both players at the same starting position
            elif self.player_positions[1] == pos:
                line[pos] = "A"
            elif self.player_positions[-1] == pos:
                line[pos] = "B"
        line_str = " ".join(line)
        return f"Line: {line_str}\nCurrent Player: {'A' if self.current_player == 1 else 'B'}"

    def valid_moves(self):
        valid_actions = []
        current_pos = self.player_positions[self.current_player]
        opponent = -self.current_player
        opponent_pos = self.player_positions[opponent]

        for action in range(3):  # Actions 0, 1, 2 correspond to moves 1, 2, 3
            move_distance = action + 1
            desired_position = current_pos + move_distance
            if desired_position > 20:
                # Allowed to move beyond position 20
                valid_actions.append(action)
                continue
            if desired_position == opponent_pos:
                # Would land on opponent, need to check if can stop before opponent
                blocked_position = opponent_pos - 1
                if blocked_position <= current_pos:
                    # No forward movement possible
                    continue
                else:
                    valid_actions.append(action)
            else:
                valid_actions.append(action)
        if not valid_actions:
            # If no valid moves, player must skip turn
            pass  # No action is added, resulting in an empty list
        return valid_actions

    def _get_obs(self):
        return np.array(
            [self.player_positions[1], self.player_positions[-1], self.current_player],
            dtype=np.float32,
        )
