import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 5 possible numbers (indices 0-4) and 5 possible opponent positions (indices 0-4)
        # So total actions = 5 * 5 = 25
        self.action_space = spaces.Discrete(25)
        # Observation space:
        # - Positions 0-4: Own numbers available (1 if available, 0 if used)
        # - Position 5: Number of opponent's remaining numbers (integer from 0 to 5)
        self.observation_space = spaces.Box(low=0, high=5, shape=(6,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Each player's remaining numbers
        self.player_numbers = [
            [1, 2, 3, 4, 5],  # Player 1's numbers
            [1, 2, 3, 4, 5],  # Player 2's numbers
        ]
        # Current player: 0 (Player 1) or 1 (Player 2)
        self.current_player = 0
        # Game over flag
        self.done = False

        # Initialize observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        own_numbers = self.player_numbers[self.current_player]
        opponent_numbers = self.player_numbers[1 - self.current_player]

        # Decode action
        own_number_index = action // 5  # 0-4
        opponent_position = action % 5  # 0-4

        # Validate own number index
        if own_number_index < 0 or own_number_index >= 5:
            return self._get_observation(), -10, True, False, {}

        # Validate own number availability
        own_number_value = own_number_index + 1  # Numbers are 1-5
        if own_number_value not in own_numbers:
            return self._get_observation(), -10, True, False, {}

        # Validate opponent position
        if opponent_position < 0 or opponent_position >= len(opponent_numbers):
            return self._get_observation(), -10, True, False, {}

        # Get opponent's number
        opponent_number_value = opponent_numbers[opponent_position]

        # Reveal Phase and Resolution
        if own_number_value > opponent_number_value:
            # Attacker wins, defender's number is eliminated
            del opponent_numbers[opponent_position]
        elif own_number_value < opponent_number_value:
            # Defender wins, attacker's number is eliminated
            own_numbers.remove(own_number_value)
        else:
            # Tie, both numbers are eliminated
            del opponent_numbers[opponent_position]
            own_numbers.remove(own_number_value)

        # Update the players' numbers
        self.player_numbers[self.current_player] = own_numbers
        self.player_numbers[1 - self.current_player] = opponent_numbers

        # Check for game over condition
        if len(opponent_numbers) == 0:
            # Current player wins
            self.done = True
            reward = 1
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Switch current player
        self.current_player = 1 - self.current_player

        # Prepare observation for the next player
        observation = self._get_observation()
        reward = 0
        return observation, reward, False, False, {}

    def render(self):
        own_numbers = self.player_numbers[self.current_player]
        opponent_numbers = self.player_numbers[1 - self.current_player]
        render_str = f"Player {self.current_player + 1}'s turn\n"
        render_str += f"Your Remaining Numbers: {own_numbers}\n"
        render_str += f"Opponent's Remaining Numbers: {len(opponent_numbers)} numbers\n"
        return render_str

    def valid_moves(self):
        own_numbers = self.player_numbers[self.current_player]
        opponent_numbers = self.player_numbers[1 - self.current_player]
        valid_actions = []
        for own_number_value in own_numbers:
            own_number_index = own_number_value - 1  # Indices 0-4
            for opponent_position in range(len(opponent_numbers)):
                action = own_number_index * 5 + opponent_position
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        own_numbers = self.player_numbers[self.current_player]
        opponent_numbers = self.player_numbers[1 - self.current_player]

        # Own numbers available: 1 if available, 0 if used
        own_numbers_available = np.zeros(5, dtype=np.int32)
        for number in own_numbers:
            own_numbers_available[number - 1] = 1  # Indices 0-4

        # Number of opponent's remaining numbers
        opponent_numbers_remaining = len(opponent_numbers)

        observation = np.concatenate(
            [
                own_numbers_available,
                np.array([opponent_numbers_remaining], dtype=np.int32),
            ]
        )
        return observation
