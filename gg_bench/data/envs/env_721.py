import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # 0: Choose first number, add it to total
        # 1: Choose first number, subtract it from total
        # 2: Choose last number, add it to total
        # 3: Choose last number, subtract it from total
        self.action_space = spaces.Discrete(4)

        # Observation space:
        # - Shared sequence: array of length 9 (numbers 1 to 9 or 0 if removed)
        # - Current player's total
        # - Opponent's total
        self.observation_space = spaces.Box(
            low=-20, high=20, shape=(11,), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_sequence = list(range(1, 10))  # Numbers from 1 to 9
        self.player_totals = {1: 0, -1: 0}
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )  # No reward after game is over

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Map action to selected number and operation
        if action == 0:
            selected_number = self.shared_sequence[0]
            operation = "add"
            self.shared_sequence.pop(0)
        elif action == 1:
            selected_number = self.shared_sequence[0]
            operation = "subtract"
            self.shared_sequence.pop(0)
        elif action == 2:
            selected_number = self.shared_sequence[-1]
            operation = "add"
            self.shared_sequence.pop(-1)
        elif action == 3:
            selected_number = self.shared_sequence[-1]
            operation = "subtract"
            self.shared_sequence.pop(-1)

        # Update the player's total
        if operation == "add":
            self.player_totals[self.current_player] += selected_number
        else:
            self.player_totals[self.current_player] -= selected_number

        reward = 0
        # Check for win condition
        current_total = self.player_totals[self.current_player]
        opponent_total = self.player_totals[-self.current_player]

        if current_total == 15:
            if opponent_total != 15:
                # Current player wins
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, {}
            else:
                # Both players have 15, game continues
                pass  # Continue the game
        else:
            if opponent_total == 15:
                # Opponent has 15, game continues
                pass  # Continue the game

        # Check if all numbers have been used
        if len(self.shared_sequence) == 0:
            if current_total != 15 and opponent_total != 15:
                # Neither player has 15, game ends in loss
                self.done = True
                reward = 0
                return self._get_observation(), reward, True, False, {}
            elif current_total == 15 and opponent_total == 15:
                # Both players have 15, but no numbers left, game ends in loss
                self.done = True
                reward = 0
                return self._get_observation(), reward, True, False, {}
            else:
                # Game continues until a player reaches 15 while the opponent does not
                # However, since no numbers are left, the game ends
                self.done = True
                if current_total == 15 and opponent_total != 15:
                    reward = 1  # Current player wins
                else:
                    reward = 0  # Current player loses or draw
                return self._get_observation(), reward, True, False, {}
        else:
            # Switch players
            self.current_player *= -1
            return (
                self._get_observation(),
                reward,
                False,
                False,
                {},
            )  # Continue the game

    def _get_observation(self):
        # Create observation array
        shared_sequence_padded = np.array(
            self.shared_sequence + [0] * (9 - len(self.shared_sequence)),
            dtype=np.float32,
        )
        current_total = self.player_totals[self.current_player]
        opponent_total = self.player_totals[-self.current_player]
        observation = np.concatenate(
            (shared_sequence_padded, [current_total, opponent_total])
        )
        return observation

    def render(self):
        # Return a string representation of the sequence and totals
        sequence_str = " ".join(["[{}]".format(n) for n in self.shared_sequence])
        total_str = "Player 1 Total: {}\nPlayer 2 Total: {}".format(
            self.player_totals[1], self.player_totals[-1]
        )
        return "Sequence: {}\n{}\n".format(sequence_str, total_str)

    def valid_moves(self):
        if self.done:
            return []
        if len(self.shared_sequence) == 0:
            return []

        valid_actions = []
        if len(self.shared_sequence) >= 1:
            # Actions for first number
            valid_actions.extend([0, 1])
            # Actions for last number (if different from first)
            if len(self.shared_sequence) > 1:
                valid_actions.extend([2, 3])
            else:
                valid_actions.extend([2, 3])  # First and last numbers are the same
        return valid_actions
