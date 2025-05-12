import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: move 1, 2, or 3 positions ahead
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0 -> move 1, 1 -> move 2, 2 -> move 3

        # Define observation space
        # For each position from 0 to 20, we have 3 features:
        # [Obstacle presence, Player 1 presence, Player 2 presence]
        # Shape: (21, 3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(21, 3), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the track positions
        # Position 0 is always clear ('C')
        # Positions 1 to 20 have a 30% chance of being an obstacle ('O')
        self.track_length = 21  # Positions 0 to 20
        self.obstacles = np.zeros(self.track_length, dtype=np.float32)
        obstacle_positions = np.random.choice(
            [0, 1], size=self.track_length - 1, p=[0.7, 0.3]
        )
        self.obstacles[1:] = obstacle_positions  # Exclude position 0

        # Initialize player positions at position 0
        self.player_positions = {1: 0, 2: 0}  # Player 1 position  # Player 2 position

        # Set current player (Player 1 starts)
        self.current_player = 1

        # Game over flag
        self.done = False

        # Prepare the initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return current observation
            observation = self._get_observation()
            return observation, 0, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            observation = self._get_observation()
            return observation, -10, True, False, {}

        # Map action to move distance
        move_distance = action + 1  # Actions 0,1,2 correspond to moving 1,2,3 positions

        # Get current player position
        curr_pos = self.player_positions[self.current_player]

        # Determine new position (cannot go beyond position 20)
        new_pos = min(curr_pos + move_distance, 20)

        # Check positions for obstacles
        positions_to_check = np.arange(curr_pos + 1, new_pos + 1)
        obstacles_in_path = self.obstacles[positions_to_check]

        if np.any(obstacles_in_path == 1):
            # Obstacle encountered, move fails
            # Player remains at current position
            pass
        else:
            # Move successful, update player position
            self.player_positions[self.current_player] = new_pos

        # Check for win condition
        if self.player_positions[self.current_player] >= 20:
            # Current player wins
            observation = self._get_observation()
            self.done = True
            return observation, 1, True, False, {}

        # Check if the other player can make any valid moves
        self._switch_player()
        other_player_moves = self.valid_moves()
        if not other_player_moves:
            # Other player cannot make a move, current player wins
            observation = self._get_observation()
            self.done = True
            return observation, 1, True, False, {}

        # Switch back to current player for the next turn
        self._switch_player()

        # Prepare observation for the next player
        self._switch_player()
        observation = self._get_observation()
        self._switch_player()

        # Continue the game
        self._switch_player()  # Switch to the next player
        return observation, 0, False, False, {}

    def render(self):
        # Generate a visual representation of the track
        track_visual = ""
        for pos in range(self.track_length):
            cell = ""
            if self.obstacles[pos] == 1:
                cell += "O"  # Obstacle
            else:
                cell += "C"  # Clear
            if self.player_positions[1] == pos and self.player_positions[2] == pos:
                cell += " [P1&P2]"
            elif self.player_positions[1] == pos:
                cell += " [P1]"
            elif self.player_positions[2] == pos:
                cell += " [P2]"
            track_visual += f"Position {pos}: {cell}\n"
        return track_visual

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        curr_pos = self.player_positions[self.current_player]
        for action in range(3):  # Actions 0,1,2 correspond to moving 1,2,3 positions
            move_distance = action + 1
            new_pos = min(curr_pos + move_distance, 20)
            positions_to_check = np.arange(curr_pos + 1, new_pos + 1)
            if positions_to_check.size == 0:
                continue  # No positions to check
            obstacles_in_path = self.obstacles[positions_to_check]
            if np.any(obstacles_in_path == 1):
                continue  # Obstacle encountered, move not possible
            valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Generate the observation array
        # Shape: (21, 3) -- positions 0 to 20, features: [Obstacle, Player1, Player2]
        observation = np.zeros((self.track_length, 3), dtype=np.float32)
        # Set obstacle presence
        observation[:, 0] = self.obstacles
        # Set player positions
        observation[self.player_positions[1], 1] = 1  # Player 1
        observation[self.player_positions[2], 2] = 1  # Player 2
        return observation

    def _switch_player(self):
        # Switch to the other player
        self.current_player = 1 if self.current_player == 2 else 2
