import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is digits 0-9
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # Positions 0-9: counts of digits 0-9 in opponent's number
        # Position 10: current player's target number
        # Position 11: opponent's target number
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(12,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_target_numbers = [25, 25]  # Player 1 and Player 2 target numbers
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False

        # Build initial observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def _get_observation(self):
        # Get digits of opponent's number
        opponent_number = self.player_target_numbers[1 - self.current_player]

        digit_counts = np.zeros(10, dtype=np.int32)
        for digit_char in str(opponent_number):
            digit = int(digit_char)
            digit_counts[digit] += 1

        current_player_number = self.player_target_numbers[self.current_player]
        opponent_player_number = self.player_target_numbers[1 - self.current_player]

        observation = np.concatenate(
            [digit_counts, [current_player_number, opponent_player_number]]
        )

        return observation

    def valid_moves(self):
        # Returns a list of valid digits that can be subtracted
        opponent_number = self.player_target_numbers[1 - self.current_player]
        current_player_number = self.player_target_numbers[self.current_player]
        valid_moves = []

        # Get digits from opponent's number
        digits = [int(d) for d in str(opponent_number)]

        # Each occurrence is considered separately, but only one digit can be subtracted per turn
        # We can have multiple same digits if they occur multiple times
        digit_counts = {}
        for d in digits:
            digit_counts[d] = digit_counts.get(d, 0) + 1

        for digit in digit_counts:
            if 0 <= digit <= 9:
                if current_player_number - digit >= 0:
                    valid_moves.append(digit)
        return valid_moves

    def step(self, action):
        """
        Parameters:
        - action: integer in [0,9], representing the digit to subtract
        """

        if self.done:
            # The game is over
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if not valid_actions:
            # Current player has no valid moves, loses
            self.done = True
            reward = -10  # Current player loses
            return self._get_observation(), reward, True, False, {}

        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10  # Current player loses
            return self._get_observation(), reward, True, False, {}

        # Valid action, subtract digit from current player's number
        self.player_target_numbers[self.current_player] -= action

        # Check for win condition
        if self.player_target_numbers[self.current_player] == 0:
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Switch to next player
        self.current_player = 1 - self.current_player

        # Since the game is not over, reward is 0
        reward = 0

        return self._get_observation(), reward, False, False, {}

    def render(self):
        player_num = self.current_player + 1
        current_player_number = self.player_target_numbers[self.current_player]
        opponent_player_number = self.player_target_numbers[1 - self.current_player]
        digits_in_opponent_number = [int(d) for d in str(opponent_player_number)]

        render_str = f"Player {player_num}'s turn:\n"
        render_str += f"Your target number: {current_player_number}\n"
        render_str += f"Opponent's number: {opponent_player_number}\n"
        render_str += "Available digits to subtract: "
        render_str += ", ".join(str(d) for d in digits_in_opponent_number)
        render_str += "\n"

        return render_str

    def valid_moves(self):
        # Returns a list of valid digits that can be subtracted
        opponent_number = self.player_target_numbers[1 - self.current_player]
        current_player_number = self.player_target_numbers[self.current_player]
        valid_moves = []

        # Get digits from opponent's number
        digits = [int(d) for d in str(opponent_number)]

        # Each occurrence is considered separately, but only one digit can be subtracted per turn
        # We can have multiple same digits if they occur multiple times
        for digit in digits:
            if 0 <= digit <= 9:
                if current_player_number - digit >= 0:
                    valid_moves.append(digit)

        return valid_moves
