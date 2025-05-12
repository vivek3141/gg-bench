import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-7 inclusive for the 8 possible actions
        self.action_space = spaces.Discrete(8)

        # Observation space: current player's total and opponent's total, values from 1 to 100
        self.observation_space = spaces.Box(low=1, high=100, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_totals = [1, 1]  # Totals for Player 1 and Player 2
        self.current_player = 0  # Player 1 starts (index 0)
        self.done = False
        observation = np.array(
            [
                self.player_totals[self.current_player],
                self.player_totals[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return observation, {}

    def step(self, action):
        if self.done:
            # The game has ended
            observation = np.array(
                [
                    self.player_totals[self.current_player],
                    self.player_totals[1 - self.current_player],
                ],
                dtype=np.int32,
            )
            return observation, 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            observation = np.array(
                [
                    self.player_totals[self.current_player],
                    self.player_totals[1 - self.current_player],
                ],
                dtype=np.int32,
            )
            return observation, reward, True, False, {}

        # Map action index to operation and number
        operations = [
            "add",
            "add",
            "add",
            "add",
            "multiply",
            "multiply",
            "multiply",
            "multiply",
        ]
        numbers = [2, 3, 4, 5, 2, 3, 4, 5]
        operation = operations[action]
        number = numbers[action]

        current_total = self.player_totals[self.current_player]

        if operation == "add":
            new_total = current_total + number
        elif operation == "multiply":
            new_total = current_total * number

        # Check for game-ending conditions
        if new_total > 100:
            # Current player loses
            reward = -1
            self.done = True
        elif new_total == 100:
            # Current player wins
            reward = 1
            self.done = True
            self.player_totals[self.current_player] = new_total
        else:
            # Valid move, game continues
            reward = 0
            self.player_totals[self.current_player] = new_total
            self.current_player = 1 - self.current_player  # Switch player

        # Prepare observation for the next player
        observation = np.array(
            [
                self.player_totals[self.current_player],
                self.player_totals[1 - self.current_player],
            ],
            dtype=np.int32,
        )
        return observation, reward, self.done, False, {}

    def render(self):
        s = f"Player 1 Total: {self.player_totals[0]}\n"
        s += f"Player 2 Total: {self.player_totals[1]}\n"
        s += f"Player {self.current_player + 1}'s turn.\n"
        return s

    def valid_moves(self):
        valid_actions = []
        current_total = self.player_totals[self.current_player]
        # Action mappings
        operations = [
            "add",
            "add",
            "add",
            "add",
            "multiply",
            "multiply",
            "multiply",
            "multiply",
        ]
        numbers = [2, 3, 4, 5, 2, 3, 4, 5]

        for action in range(8):
            operation = operations[action]
            number = numbers[action]
            if operation == "add":
                new_total = current_total + number
            elif operation == "multiply":
                new_total = current_total * number

            if new_total <= 100:
                valid_actions.append(action)

        return valid_actions
