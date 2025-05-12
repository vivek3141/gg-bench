import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: actions correspond to moving 1, 2, or 3 positions
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0, 1, 2 correspond to moving 1, 2, or 3 steps

        # Define observation space
        # Observation consists of [Player 1 position, Player 2 position, current player]
        # Player positions are integers from 0 to 5 for Player 1 and 5 to 10 for Player 2
        # Current player: 1 for Player 1's turn, -1 for Player 2's turn
        self.observation_space = spaces.Box(
            low=np.array([0, 5, -1]), high=np.array([5, 10, 1]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.P1_pos = 0  # Player 1's starting position
        self.P2_pos = 10  # Player 2's starting position
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        observation = np.array(
            [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return the current state
            observation = np.array(
                [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
            )
            return observation, 0, True, False, {}

        valid_actions = self.valid_moves()
        if not valid_actions:
            # No valid moves, skip turn
            self.current_player *= -1
            observation = np.array(
                [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
            )
            return observation, 0, False, False, {}

        if action not in valid_actions:
            # Invalid move, apply penalty and end game
            self.done = True
            observation = np.array(
                [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
            )
            return observation, -10, True, False, {}

        move_distance = (
            action + 1
        )  # Convert action (0, 1, 2) to move distances (1, 2, 3)

        # Move the current player's token
        if self.current_player == 1:
            # Player 1's move
            new_pos = self.P1_pos + move_distance
            if new_pos > 5:
                new_pos = 5  # Cannot move past position 5
            self.P1_pos = new_pos

            if self.P1_pos == 5:
                # Player 1 reaches the Golden Token and wins
                self.done = True
                observation = np.array(
                    [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
                )
                return observation, 1, True, False, {}
        else:
            # Player 2's move
            new_pos = self.P2_pos - move_distance
            if new_pos < 5:
                new_pos = 5  # Cannot move past position 5
            self.P2_pos = new_pos

            if self.P2_pos == 5:
                if self.P1_pos == 5:
                    # Player 1 has already reached the Golden Token; Player 1 wins
                    self.done = True
                    observation = np.array(
                        [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
                    )
                    return observation, -10, True, False, {}
                else:
                    # Player 2 reaches the Golden Token and wins
                    self.done = True
                    observation = np.array(
                        [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
                    )
                    return observation, 1, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        observation = np.array(
            [self.P1_pos, self.P2_pos, self.current_player], dtype=np.int32
        )
        return observation, 0, False, False, {}

    def render(self):
        # Create a visual representation of the track
        track = ["[ ]"] * 11
        track[5] = "[G]"  # Golden Token at position 5
        track[self.P1_pos] = "[P1]" if self.P1_pos != 5 else "[P1/G]"
        track[self.P2_pos] = "[P2]" if self.P2_pos != 5 else "[P2/G]"
        track_str = "".join(track)
        print(track_str)

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        for action in range(3):
            move_distance = action + 1
            if self.current_player == 1:
                # Player 1's potential move
                potential_pos = self.P1_pos + move_distance
                if potential_pos <= 5:
                    valid_actions.append(action)
            else:
                # Player 2's potential move
                potential_pos = self.P2_pos - move_distance
                if potential_pos >= 5:
                    valid_actions.append(action)
        return valid_actions
