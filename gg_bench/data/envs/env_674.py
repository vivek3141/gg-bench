import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are integers from 0 to 4, representing moves of 1 to 5 positions
        self.action_space = spaces.Discrete(5)

        # Observation is a vector of length 22:
        # - Positions 0 to 20 represent positions 1 to 21 on the number line
        #   with values: 1 if current player occupies the position
        #                -1 if opponent occupies the position
        #                0 if unoccupied
        # - Position 21 indicates the current player (1 for Player 1, -1 for Player 2)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(22,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.P1_position = 1
        self.P2_position = 21
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        move_length = action + 1  # Map action to move length (1-5)
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Process move for current player
        if self.current_player == 1:
            current_position = self.P1_position
            opponent_position = self.P2_position
            current_player_start = 1
            opponent_start = 21
        else:
            current_position = self.P2_position
            opponent_position = self.P1_position
            current_player_start = 21
            opponent_start = 1

        desired_position = current_position + move_length
        maximum_position = min(opponent_position - 1, opponent_start)

        if maximum_position < current_position:
            maximum_position = current_position  # Can't move forward

        new_position = min(desired_position, maximum_position)

        # Update position
        if self.current_player == 1:
            self.P1_position = new_position
        else:
            self.P2_position = new_position

        # Check for win conditions
        reward = 0
        if new_position == opponent_position:
            # Capturing the opponent
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}
        elif (
            self.P1_position >= self.P2_position
            and self.P1_position != self.P2_position
        ):
            # Players have passed each other without landing on the same position
            # Determine who is closer to the opponent's starting position
            dist_P1 = abs(self.P1_position - 21)  # P1's distance to opponent's start
            dist_P2 = abs(self.P2_position - 1)  # P2's distance to opponent's start

            if dist_P1 < dist_P2:
                winner = 1
            else:
                winner = 2

            self.done = True
            if winner == self.current_player:
                reward = 1
            else:
                reward = 0
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Provide a visual representation of the number line and players' positions
        line = "Position: "
        markers = "Markers:  "
        for pos in range(1, 22):
            line += f"{pos:2} "
            if self.P1_position == pos and self.P2_position == pos:
                markers += "P1/P2 "
            elif self.P1_position == pos:
                markers += " P1   "
            elif self.P2_position == pos:
                markers += " P2   "
            else:
                markers += "      "
        return line + "\n" + markers

    def valid_moves(self):
        # Return a list of valid move actions (indices 0 to 4)
        # Based on the current player's possible moves
        if self.done:
            return []

        if self.current_player == 1:
            current_position = self.P1_position
            opponent_position = self.P2_position
            opponent_start = 21
        else:
            current_position = self.P2_position
            opponent_position = self.P1_position
            opponent_start = 1

        valid_actions = []
        for move_length in range(1, 6):  # Moves from 1 to 5
            desired_position = current_position + move_length
            maximum_position = min(opponent_position - 1, opponent_start)
            if maximum_position < current_position:
                maximum_position = current_position  # Can't move forward
            new_position = min(desired_position, maximum_position)
            if new_position >= current_position and new_position <= maximum_position:
                action = move_length - 1
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        observation = np.zeros(22, dtype=np.float32)
        for pos in range(1, 22):
            index = pos - 1
            if self.P1_position == pos:
                if self.current_player == 1:
                    observation[index] = 1
                else:
                    observation[index] = -1
            elif self.P2_position == pos:
                if self.current_player == 2:
                    observation[index] = 1
                else:
                    observation[index] = -1
            else:
                observation[index] = 0
        observation[21] = 1 if self.current_player == 1 else -1
        return observation
