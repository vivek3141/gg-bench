import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 10
        self.action_space = spaces.Discrete(
            10
        )  # Actions 0-9 correspond to numbers 1-10

        # Define observation space
        # obs[0]: Current player's cumulative total (0 to 60)
        # obs[1]: Opponent's cumulative total (0 to 60)
        # obs[2]: Number the current player cannot select (0 for none, 1 to 10)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([60, 60, 10], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_totals = [0, 0]  # Index 0: Player A, Index 1: Player B
        self.last_selected_numbers = [0, 0]  # Last selected number for each player

        # Randomly decide which player takes the first turn
        self.current_player = np.random.choice([0, 1])  # 0: Player A, 1: Player B

        self.done = False
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to number (actions 0-9 correspond to numbers 1-10)
        selected_number = action + 1

        # Get the opponent's last selected number
        opponent_last_number = self.last_selected_numbers[1 - self.current_player]

        # Check if selected number is valid (cannot be opponent's last selection)
        if selected_number == opponent_last_number:
            # Invalid move
            self.done = True
            reward = -1  # Negative reward for invalid move
            return self._get_observation(), reward, True, False, {}

        # Calculate new cumulative total
        new_total = self.player_totals[self.current_player] + selected_number

        # Check for loss condition: exceeding 60
        if new_total > 60:
            # Player loses
            self.player_totals[self.current_player] = new_total
            self.last_selected_numbers[self.current_player] = selected_number
            self.done = True
            reward = -1  # Negative reward for losing
            return self._get_observation(), reward, True, False, {}

        # Update player's total and last selected number
        self.player_totals[self.current_player] = new_total
        self.last_selected_numbers[self.current_player] = selected_number

        # Check for victory condition: cumulative total is prime and >50
        if new_total > 50 and self._is_prime(new_total):
            # Current player wins
            self.done = True
            reward = 1  # Positive reward for winning
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player

        # Check if the next player can make any valid moves
        if not self.valid_moves():
            # Next player cannot move; current player wins
            self.current_player = 1 - self.current_player  # Switch back to winner
            self.done = True
            reward = 1  # Positive reward for winning
            return self._get_observation(), reward, True, False, {}

        # Game continues
        reward = 0  # No reward for valid move
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Generate a textual representation of the game state
        s = f"Player {chr(65 + self.current_player)}'s Turn\n"
        s += f"Your Total: {self.player_totals[self.current_player]}\n"
        s += f"Opponent's Total: {self.player_totals[1 - self.current_player]}\n"
        forbidden_number = self.last_selected_numbers[1 - self.current_player]
        if forbidden_number:
            s += f"Number you cannot select: {forbidden_number}\n"
        else:
            s += "You can select any number between 1 and 10.\n"
        return s

    def valid_moves(self):
        # Return a list of valid actions (indices) for the current player
        forbidden_number = self.last_selected_numbers[1 - self.current_player]
        valid_actions = []
        for action in range(10):  # Actions 0-9 correspond to numbers 1-10
            selected_number = action + 1
            if selected_number == forbidden_number:
                continue
            potential_total = self.player_totals[self.current_player] + selected_number
            if potential_total > 60:
                continue
            valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Create the observation array
        obs = np.array(
            [
                self.player_totals[self.current_player],  # Current player's total
                self.player_totals[1 - self.current_player],  # Opponent's total
                self.last_selected_numbers[
                    1 - self.current_player
                ],  # Number cannot select
            ],
            dtype=np.float32,
        )
        return obs

    @staticmethod
    def _is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
