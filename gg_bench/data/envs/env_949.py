import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: digits 1-9 with attack or defend action, total of 18 possible actions
        self.action_space = spaces.Discrete(18)

        # Observation space: [player_life, opponent_life, digit_pool(9 digits availability)]
        # Life points range from 0 to 20
        # Digit availability: 1 (available), 0 (used)
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0] * 9),
            high=np.array([20, 20] + [1] * 9),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_life = 10
        self.opponent_life = 10
        self.digit_pool = np.ones(
            9, dtype=np.int32
        )  # 1 indicates the digit is available
        self.current_player = 1  # 1 for player, -1 for opponent
        self.done = False
        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {}

        digit_index = action // 2  # Index from 0 to 8 for digits 1 to 9
        action_type = action % 2  # 0 for attack, 1 for defend
        digit = digit_index + 1  # Actual digit from 1 to 9

        if digit_index < 0 or digit_index > 8 or self.digit_pool[digit_index] == 0:
            self.done = True
            return self.get_observation(), -10, True, False, {}

        self.digit_pool[digit_index] = 0  # Remove the digit from the pool

        if self.current_player == 1:
            if action_type == 0:  # Attack
                self.opponent_life -= digit
            else:  # Defend
                self.player_life = min(self.player_life + digit, 20)
        else:
            if action_type == 0:  # Attack
                self.player_life -= digit
            else:  # Defend
                self.opponent_life = min(self.opponent_life + digit, 20)

        # Check for victory
        if self.opponent_life <= 0 and self.current_player == 1:
            self.done = True
            return self.get_observation(), 1, True, False, {}
        elif self.player_life <= 0 and self.current_player == -1:
            self.done = True
            return self.get_observation(), 1, True, False, {}

        # Check if all digits have been used
        if np.sum(self.digit_pool) == 0:
            self.done = True
            if self.player_life > self.opponent_life:
                return (
                    self.get_observation(),
                    1 if self.current_player == 1 else -1,
                    True,
                    False,
                    {},
                )
            elif self.opponent_life > self.player_life:
                return (
                    self.get_observation(),
                    -1 if self.current_player == 1 else 1,
                    True,
                    False,
                    {},
                )
            else:
                # Sudden Death (not implemented)
                return self.get_observation(), 0, True, False, {}

        # Switch current player
        self.current_player *= -1

        return self.get_observation(), 0, False, False, {}

    def get_observation(self):
        if self.current_player == 1:
            return np.array(
                [self.player_life, self.opponent_life] + self.digit_pool.tolist(),
                dtype=np.int32,
            )
        else:
            return np.array(
                [self.opponent_life, self.player_life] + self.digit_pool.tolist(),
                dtype=np.int32,
            )

    def render(self):
        state = "Current Player: {}\n".format(
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        state += "Player 1 Life Points: {}\n".format(self.player_life)
        state += "Player 2 Life Points: {}\n".format(self.opponent_life)
        state += "Available Digits: {}\n".format(
            [i + 1 for i, available in enumerate(self.digit_pool) if available == 1]
        )
        return state

    def valid_moves(self):
        moves = []
        for digit_index, available in enumerate(self.digit_pool):
            if available == 1:
                moves.append(digit_index * 2)  # Attack action
                moves.append(digit_index * 2 + 1)  # Defend action
        return moves
