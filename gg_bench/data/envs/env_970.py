import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Digits 1-9 represented as actions 0-8
        self.action_space = spaces.Discrete(9)

        # Observation space: [Shared Number, Player's digits (9), Opponent's digits (9), Player's score, Opponent's score]
        # Shared Number ranges from 1 to 1e10
        # Digits are 0 (used) or 1 (available)
        # Scores from 0 to 12
        low = np.array([1.0] + [0] * 18 + [0, 0], dtype=np.float32)
        high = np.array([1e10] + [1] * 18 + [12, 12], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 1

        # Digits available to each player: 1 - available, 0 - used
        self.player_digits = np.ones(9, dtype=np.int8)
        self.opponent_digits = np.ones(9, dtype=np.int8)

        self.player_score = 0
        self.opponent_score = 0

        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False

        return self._get_observation(), {}

    def _get_observation(self):
        # Shared Number
        observation = [float(self.shared_number)]

        # Order digits and scores based on current player
        if self.current_player == 1:
            player_digits = self.player_digits
            opponent_digits = self.opponent_digits
            player_score = self.player_score
            opponent_score = self.opponent_score
        else:
            player_digits = self.opponent_digits
            opponent_digits = self.player_digits
            player_score = self.opponent_score
            opponent_score = self.player_score

        # Add player's and opponent's digits
        observation.extend(player_digits)
        observation.extend(opponent_digits)

        # Add player's and opponent's scores
        observation.append(float(player_score))
        observation.append(float(opponent_score))

        return np.array(observation, dtype=np.float32)

    def has_digits(self, player):
        if player == 1:
            return np.sum(self.player_digits) > 0
        else:
            return np.sum(self.opponent_digits) > 0

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if current player has digits
        if not self.has_digits(self.current_player):
            # Switch to other player
            self.current_player *= -1

            if not self.has_digits(self.current_player):
                # Neither player has digits, game over
                self.done = True
                # Determine reward (0 since no player reached exactly 12)
                return self._get_observation(), 0, True, False, {}

        # Convert action to digit
        chosen_digit = action + 1

        # Check if action is valid
        if self.current_player == 1:
            if self.player_digits[action] == 0:
                # Invalid move
                self.done = True
                return self._get_observation(), -10, True, False, {}
        else:
            if self.opponent_digits[action] == 0:
                # Invalid move
                self.done = True
                return self._get_observation(), -10, True, False, {}

        # Multiply Shared Number
        self.shared_number *= chosen_digit

        # Remove digit from player's set
        if self.current_player == 1:
            self.player_digits[action] = 0
        else:
            self.opponent_digits[action] = 0

        # Check divisibility by 6
        if self.shared_number % 6 == 0:
            points_earned = 2
        else:
            points_earned = 1

        # Update player's score and check for reset or win
        if self.current_player == 1:
            self.player_score += points_earned
            if self.player_score > 12:
                self.player_score = 6
            if self.player_score == 12:
                self.done = True
                return self._get_observation(), 1, True, False, {}
        else:
            self.opponent_score += points_earned
            if self.opponent_score > 12:
                self.opponent_score = 6
            if self.opponent_score == 12:
                self.done = True
                return self._get_observation(), 1, True, False, {}

        # Switch to next player
        self.current_player *= -1

        # Check if next player has digits
        if not self.has_digits(self.current_player):
            # Switch back to current player
            self.current_player *= -1
            if not self.has_digits(self.current_player):
                # Neither player has digits, game over
                self.done = True
                return self._get_observation(), 0, True, False, {}

        return self._get_observation(), 0, False, False, {}

    def render(self):
        output = ""
        output += f"Shared Number: {self.shared_number}\n"

        if self.current_player == 1:
            player_digits = self.player_digits
            opponent_digits = self.opponent_digits
            player_score = self.player_score
            opponent_score = self.opponent_score
        else:
            player_digits = self.opponent_digits
            opponent_digits = self.player_digits
            player_score = self.opponent_score
            opponent_score = self.player_score

        output += f"Current Player's Available Digits: {[i+1 for i in range(9) if player_digits[i]==1]}\n"
        output += f"Current Player's Score: {player_score}\n"
        output += f"Opponent's Available Digits: {[i+1 for i in range(9) if opponent_digits[i]==1]}\n"
        output += f"Opponent's Score: {opponent_score}\n"
        output += f"It is Player {'1' if self.current_player==1 else '2'}'s turn.\n"
        return output

    def valid_moves(self):
        if self.current_player == 1:
            return [i for i in range(9) if self.player_digits[i] == 1]
        else:
            return [i for i in range(9) if self.opponent_digits[i] == 1]
