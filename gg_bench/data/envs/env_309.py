import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(
            9
        )  # Actions are numbers from 1 to 9 (indices 0 to 8)
        self.max_tower_height = 10  # Maximum height of the tower
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_tower_height,), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tower = np.zeros(
            self.max_tower_height, dtype=np.int32
        )  # Initialize tower with zeros
        self.tower_position = 0  # Current position in the tower
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game is not over
        return self.tower.copy(), {}  # Return initial observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.tower.copy(), 0, True, False, {}

        number = action + 1  # Map action index to number (1 to 9)

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.tower.copy(), reward, True, False, {"reason": "Invalid move"}

        # Valid move, update tower
        if self.tower_position >= self.max_tower_height:
            # Tower is full (should not happen if max_tower_height is sufficient)
            self.done = True
            reward = -10  # Penalty for exceeding tower height
            return self.tower.copy(), reward, True, False, {"reason": "Tower is full"}

        self.tower[self.tower_position] = number
        self.tower_position += 1

        # Now check if the opponent can make any valid moves
        opponent_valid_actions = self.valid_moves()
        if not opponent_valid_actions:
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1  # Reward for winning
            return (
                self.tower.copy(),
                reward,
                True,
                False,
                {"reason": "Opponent cannot move"},
            )

        # Swap current player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0  # No immediate reward
        return (
            self.tower.copy(),
            reward,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def valid_moves(self, last_number=None):
        # Return list of valid action indices (0-8) for the current player
        if self.done:
            return []

        if last_number is None:
            if self.tower_position == 0:
                # Tower is empty, all numbers are valid
                return list(range(9))
            else:
                last_number = self.tower[self.tower_position - 1]

        valid_actions = []
        for action in range(9):
            number = action + 1
            if last_number % number != 0 and number % last_number != 0:
                valid_actions.append(action)
        return valid_actions

    def render(self):
        if self.tower_position == 0:
            return "The tower is empty."
        else:
            tower_numbers = self.tower[: self.tower_position].astype(int)
            tower_str = "Tower: " + " ".join(map(str, tower_numbers))
            return tower_str
