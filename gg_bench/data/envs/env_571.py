import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0,1,2 correspond to forces 1,2,3
        self.action_space = spaces.Discrete(3)

        # Observation space: token position (0-10), current player (-1 or 1)
        self.observation_space = spaces.Box(
            low=np.array([0, -1]),
            high=np.array([10, 1]),
            shape=(2,),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.token_position = 5  # Initial position
        self.current_player = 1  # Player 1 starts (1), Player 2 is (-1)
        self.done = False
        return np.array([self.token_position, self.current_player], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # Game is over
            return (
                np.array([self.token_position, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Check if current player can make any valid moves
        current_valid_actions = self.valid_moves()
        if not current_valid_actions:
            # Current player cannot make a valid move, they lose
            self.done = True
            reward = -1  # Current player loses
            return (
                np.array([self.token_position, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if action is valid
        if action not in current_valid_actions:
            # Invalid action
            self.done = True
            reward = -10  # Penalty for invalid move
            return (
                np.array([self.token_position, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Convert action to force
        force = action + 1  # action 0->force 1, action 1->force 2, action2->force3

        # Apply force towards player's goal
        if self.current_player == 1:
            # Player 1 moves token towards 0 (decreasing position)
            self.token_position -= force
            if self.token_position < 0:
                self.token_position = 0  # Ensure within bounds
        else:
            # Player 2 moves token towards 10 (increasing position)
            self.token_position += force
            if self.token_position > 10:
                self.token_position = 10  # Ensure within bounds

        # Check for win condition (before gravity)
        if (self.current_player == 1 and self.token_position == 0) or (
            self.current_player == -1 and self.token_position == 10
        ):
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.token_position, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply gravity effect
        if self.token_position < 5:
            self.token_position += 1  # Gravity pulls towards 5
            if self.token_position > 10:
                self.token_position = 10
        elif self.token_position > 5:
            self.token_position -= 1
            if self.token_position < 0:
                self.token_position = 0
        # Gravity has no effect if token_position == 5

        # Check for win condition (after gravity)
        if (self.current_player == 1 and self.token_position == 0) or (
            self.current_player == -1 and self.token_position == 10
        ):
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.token_position, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to next player
        self.current_player *= -1  # Switch players (1 -> -1, -1 -> 1)

        # Check if next player can make a valid move
        next_valid_actions = self.valid_moves()
        if not next_valid_actions:
            # Next player cannot make a valid move, current player wins
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.token_position, -self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # No win or loss, continue game
        reward = 0
        return (
            np.array([self.token_position, self.current_player], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        # Visual representation of the game state
        track = ""
        for position in range(0, 11):
            if position == self.token_position:
                track += "T"  # Token position
            else:
                track += "_"
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Track: {track}\nCurrent Player: {player_str}\n"

    def valid_moves(self):
        # Return list of valid actions (0,1,2) for which applying force does not overshoot goal
        valid_actions = []
        for action in range(3):  # Actions corresponding to forces 1,2,3
            force = action + 1
            if self.current_player == 1:
                # Player 1 moves token towards 0
                new_position = self.token_position - force
                if new_position >= 0:
                    valid_actions.append(action)
            else:
                # Player 2 moves token towards 10
                new_position = self.token_position + force
                if new_position <= 10:
                    valid_actions.append(action)
        return valid_actions
