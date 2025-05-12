import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 3 possible actions: move 1, 2, or 3 units
        self.action_space = spaces.Discrete(3)

        # Observation space consists of the marker position and the current player
        # Marker position ranges from -7 to +7
        # Current player is -1 (Player Left) or +1 (Player Right)
        self.observation_space = spaces.Box(
            low=np.array([-7, -1]), high=np.array([7, 1]), shape=(2,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.marker_position = 0
        # Start with Player Left (-1)
        self.current_player = -1  # -1 for Player Left, +1 for Player Right
        self.done = False
        observation = np.array(
            [self.marker_position, self.current_player], dtype=np.float32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return current state
            observation = np.array(
                [self.marker_position, self.current_player], dtype=np.float32
            )
            return observation, 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = np.array(
                [self.marker_position, self.current_player], dtype=np.float32
            )
            return observation, reward, True, False, {}

        # Execute action
        units = action + 1  # action 0->1 unit, action 1->2 units, action 2->3 units
        direction = self.current_player  # -1 for Player Left, +1 for Player Right
        new_position = self.marker_position + direction * units

        # Update marker position
        self.marker_position = new_position

        # Check for win condition
        if self.current_player == -1 and self.marker_position == -7:
            # Player Left wins
            self.done = True
            reward = 1
            observation = np.array(
                [self.marker_position, self.current_player], dtype=np.float32
            )
            return observation, reward, True, False, {}

        elif self.current_player == +1 and self.marker_position == 7:
            # Player Right wins
            self.done = True
            reward = 1
            observation = np.array(
                [self.marker_position, self.current_player], dtype=np.float32
            )
            return observation, reward, True, False, {}

        else:
            # Continue game
            reward = 0
            # Switch player
            self.current_player *= -1
            observation = np.array(
                [self.marker_position, self.current_player], dtype=np.float32
            )
            return observation, reward, False, False, {}

    def render(self):
        state_str = f"Current position: {self.marker_position}\n"
        if self.current_player == -1:
            state_str += "Player Left's turn."
        else:
            state_str += "Player Right's turn."
        return state_str

    def valid_moves(self):
        valid_actions = []
        for action in range(3):
            units = action + 1
            direction = self.current_player
            new_position = self.marker_position + direction * units
            if self.current_player == -1:
                # Player Left cannot move beyond -7
                if new_position >= -7:
                    valid_actions.append(action)
            elif self.current_player == +1:
                # Player Right cannot move beyond +7
                if new_position <= 7:
                    valid_actions.append(action)
        return valid_actions
