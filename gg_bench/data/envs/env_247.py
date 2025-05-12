import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Two possible actions - move toward own goal or move away from own goal
        self.action_space = spaces.Discrete(2)

        # Define observation space: position of the marker on the number line (positions 1 to 11)
        # Using Box space as per the requirement
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([11]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Starting position of the marker at the center (position 6)
        self.marker_position = 6

        # Player 1 starts the game
        self.current_player = 1

        # Game is not over
        self.done = False

        return (
            np.array([self.marker_position], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return (
                np.array([self.marker_position], dtype=np.int32),
                -10,
                True,
                False,
                {"Error": "Game is already over"},
            )

        # Determine goals and movement directions based on the current player
        if self.current_player == 1:
            goal = 1
            opponent_goal = 11
            direction_toward_goal = -1  # Moving towards lower numbers
            direction_away_from_goal = 1  # Moving towards higher numbers
        else:
            goal = 11
            opponent_goal = 1
            direction_toward_goal = 1  # Moving towards higher numbers
            direction_away_from_goal = -1  # Moving towards lower numbers

        # Process the action
        if action == 0:
            # Move towards own goal
            new_position = self.marker_position + direction_toward_goal
        elif action == 1:
            # Move away from own goal
            new_position = self.marker_position + direction_away_from_goal
        else:
            # Invalid action
            self.done = True
            return (
                np.array([self.marker_position], dtype=np.int32),
                -10,
                True,
                False,
                {"Error": "Invalid action"},
            )

        # Check if the new position is within the number line boundaries
        if new_position < 1 or new_position > 11:
            # Invalid move; attempting to move beyond the number line
            self.done = True
            return (
                np.array([self.marker_position], dtype=np.int32),
                -10,
                True,
                False,
                {"Error": "Attempted to move beyond the number line"},
            )

        # Update the marker position
        self.marker_position = new_position

        # Check for win conditions
        if self.marker_position == goal:
            # Current player wins
            self.done = True
            return np.array([self.marker_position], dtype=np.int32), 1, True, False, {}
        elif self.marker_position == opponent_goal:
            # Current player loses
            self.done = True
            return (
                np.array([self.marker_position], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )
        else:
            # Switch to the other player's turn
            self.current_player = 2 if self.current_player == 1 else 1
            return (
                np.array([self.marker_position], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        # Create a string representation of the number line with the marker
        number_line = ""
        for i in range(1, 12):  # Positions 1 to 11
            if i == self.marker_position:
                number_line += "* "
            else:
                number_line += f"{i} "
        return number_line.strip()  # Remove trailing whitespace

    def valid_moves(self):
        # Return a list of valid actions for the current player
        valid_actions = []
        if self.done:
            # No valid moves if the game is over
            return valid_actions

        # Check if moving toward own goal is possible
        if (self.current_player == 1 and self.marker_position > 1) or (
            self.current_player == 2 and self.marker_position < 11
        ):
            valid_actions.append(0)  # Action to move toward own goal

        # Check if moving away from own goal is possible
        if (self.current_player == 1 and self.marker_position < 11) or (
            self.current_player == 2 and self.marker_position > 1
        ):
            valid_actions.append(1)  # Action to move away from own goal

        return valid_actions
