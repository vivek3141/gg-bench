import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 9 numbers * 2 actions (add or subtract) = 18 possible actions
        self.action_space = spaces.Discrete(18)

        # Define observation space
        # Observation includes:
        # - Current player's life total
        # - Opponent's life total
        # - Availability of numbers 1 to 9 (1 if available, 0 if not)
        # Life totals can range from -100 to 100
        # Numbers availability is 0 or 1
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(11,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_life = 10
        self.player2_life = 10
        self.numbers_available = np.ones(9, dtype=np.int32)  # Numbers 1 to 9 available
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}  # Return initial observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_observation(), -10, True, False, {}

        # Map action to number and action type
        number_index = action // 2
        number = number_index + 1  # Numbers 1 to 9
        action_type = action % 2  # 0 for add, 1 for subtract

        # Check if the number is available
        if self.numbers_available[number_index] == 0 or number < 1 or number > 9:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid action, proceed
        self.numbers_available[number_index] = 0  # Remove number from available list

        if action_type == 0:
            # Add to own life total
            if self.current_player == 1:
                self.player1_life += number
            else:
                self.player2_life += number
        else:
            # Subtract from opponent's life total
            if self.current_player == 1:
                self.player2_life -= number
            else:
                self.player1_life -= number

        # Check for win condition
        if (self.current_player == 1 and self.player2_life <= 0) or (
            self.current_player == 2 and self.player1_life <= 0
        ):
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check if no numbers are left
        if np.sum(self.numbers_available) == 0:
            self.done = True
            return self._get_observation(), 0, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Prepare life totals
        life_totals = f"Player 1 Life Total: {self.player1_life}\nPlayer 2 Life Total: {self.player2_life}\n"
        # Prepare available numbers
        available_numbers = [
            str(i + 1) for i in range(9) if self.numbers_available[i] == 1
        ]
        numbers_str = "Available Numbers: " + " ".join(available_numbers) + "\n"
        # Indicate current player
        current_player_str = f"Current Player: Player {self.current_player}\n"
        return life_totals + numbers_str + current_player_str

    def valid_moves(self):
        valid_actions = []
        for i in range(9):
            if self.numbers_available[i] == 1:
                valid_actions.append(i * 2)  # Action to add number to own life total
                valid_actions.append(
                    i * 2 + 1
                )  # Action to subtract number from opponent's life total
        return valid_actions

    def _get_observation(self):
        # Observation includes current player's life total, opponent's life total, and numbers availability
        if self.current_player == 1:
            observation = np.concatenate(
                (
                    np.array([self.player1_life, self.player2_life], dtype=np.int32),
                    self.numbers_available,
                )
            )
        else:
            observation = np.concatenate(
                (
                    np.array([self.player2_life, self.player1_life], dtype=np.int32),
                    self.numbers_available,
                )
            )
        return observation
