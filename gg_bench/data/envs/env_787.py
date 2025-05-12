import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions correspond to numbers 1-9
        # Observation space includes shared total, Player 1's numbers, Player 2's numbers
        self.observation_space = spaces.Box(
            low=np.array([0] + [0] * 18, dtype=np.float32),
            high=np.array([68] + [1] * 18, dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_total = 0
        self.player_numbers = {
            1: {num: 1 for num in range(1, 10)},  # Player 1's available numbers
            2: {num: 1 for num in range(1, 10)},  # Player 2's available numbers
        }
        self.current_player = 1  # Player 1 starts
        self.done = False
        self._update_observation()
        return self.observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation, -10, True, False, {}  # Game already over

        number_chosen = action + 1  # Actions are 0-8 corresponding to numbers 1-9
        if self.player_numbers[self.current_player].get(number_chosen, 0) == 0:
            # Invalid move: number already used or not in player's numbers
            self.done = True
            return self.observation, -10, True, False, {}  # Immediate loss

        # Valid move
        self.player_numbers[self.current_player][
            number_chosen
        ] = 0  # Mark number as used
        self.shared_total += number_chosen  # Add to shared total

        # Check for win condition
        if self.shared_total == 50:
            self.done = True
            self._update_observation()
            return self.observation, 1, True, False, {}  # Current player wins

        # Check for losing conditions
        if self.shared_total > 50:
            self.done = True
            self._update_observation()
            return (
                self.observation,
                -10,
                True,
                False,
                {},
            )  # Exceeded 50, current player loses

        # Check if player has no valid moves and total is less than 50
        if not any(self.player_numbers[self.current_player].values()):
            if self.shared_total < 50:
                self.done = True
                self._update_observation()
                return (
                    self.observation,
                    -10,
                    True,
                    False,
                    {},
                )  # No moves left, current player loses

        # Penalize valid move to encourage winning quickly
        reward = -10

        # Switch players
        self.current_player = 2 if self.current_player == 1 else 1

        # Update observation
        self._update_observation()
        return self.observation, reward, False, False, {}  # Continue game

    def render(self):
        player1_numbers = [
            num for num, available in self.player_numbers[1].items() if available
        ]
        player2_numbers = [
            num for num, available in self.player_numbers[2].items() if available
        ]
        render_str = f"Shared Total: {self.shared_total}\n"
        render_str += f"Player 1's Available Numbers: {sorted(player1_numbers)}\n"
        render_str += f"Player 2's Available Numbers: {sorted(player2_numbers)}\n"
        render_str += f"Current Player: Player {self.current_player}\n"
        return render_str

    def valid_moves(self):
        return [
            num - 1  # Adjusting for zero-based indexing
            for num, available in self.player_numbers[self.current_player].items()
            if available
        ]

    def _update_observation(self):
        # Observation includes shared total, player 1's numbers, player 2's numbers
        player1_nums = [value for key, value in sorted(self.player_numbers[1].items())]
        player2_nums = [value for key, value in sorted(self.player_numbers[2].items())]
        self.observation = np.array(
            [self.shared_total] + player1_nums + player2_nums, dtype=np.float32
        )
