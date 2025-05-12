import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to two-digit numbers where digits are from 1 to 9
        self.action_space = spaces.Discrete(
            81
        )  # 9 digits for each position: 9 * 9 = 81 combinations

        # Observation space:
        # - Player 1 secret digits: indices 0 and 1
        # - Player 2 secret digits: indices 2 and 3
        # - Current player: index 4 (1 or -1)
        # - Feedbacks for Player 1: indices 5 to 85 (81 elements)
        # - Feedbacks for Player 2: indices 86 to 166 (81 elements)
        obs_low = np.zeros(167, dtype=np.int32)
        obs_high = np.ones(167, dtype=np.int32) * 16  # Max feedback is 16

        obs_low[0:4] = 1  # Secret digits range from 1 to 9
        obs_high[0:4] = 9

        obs_low[4] = -1  # Current player indicator (-1 or 1)
        obs_high[4] = 1

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Generate secret numbers for both players (digits from 1 to 9)
        self.player1_secret = np.random.randint(1, 10, size=(2,), dtype=np.int32)
        self.player2_secret = np.random.randint(1, 10, size=(2,), dtype=np.int32)

        # Initialize feedback arrays
        self.feedbacks_player1 = np.zeros(81, dtype=np.int32)
        self.feedbacks_player2 = np.zeros(81, dtype=np.int32)

        # Assemble the initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Initialize reward
        reward = 0
        terminated = False
        truncated = False

        # Validate action
        if not self.action_space.contains(action):
            reward = -10
            terminated = True
            observation = self._get_observation()
            return observation, reward, terminated, truncated, {}

        # Convert action to guessed number (two digits)
        guess_digits = self.action_to_digits(action)

        # Get opponent's secret number
        opponent_secret = (
            self.player2_secret if self.current_player == 1 else self.player1_secret
        )

        # Calculate feedback
        feedback = abs(guess_digits[0] - opponent_secret[0]) + abs(
            guess_digits[1] - opponent_secret[1]
        )

        # Update feedbacks array
        if self.current_player == 1:
            self.feedbacks_player1[action] = feedback
        else:
            self.feedbacks_player2[action] = feedback

        # Check for win
        if feedback == 0:
            reward = 1
            terminated = True  # Current player wins
            observation = self._get_observation()
            return observation, reward, terminated, truncated, {}

        # Switch current player
        self.current_player *= -1

        # Return observation
        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}

    def render(self):
        output = f"Player 1 secret number: {self.player1_secret[0]} {self.player1_secret[1]}\n"
        output += f"Player 2 secret number: {self.player2_secret[0]} {self.player2_secret[1]}\n"
        output += f"Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n\n"

        output += "Player 1 guesses and feedbacks:\n"
        for action in range(81):
            feedback = self.feedbacks_player1[action]
            if feedback > 0:
                digits = self.action_to_digits(action)
                output += f"  Guessed: {digits[0]} {digits[1]}, Feedback: {feedback}\n"

        output += "\nPlayer 2 guesses and feedbacks:\n"
        for action in range(81):
            feedback = self.feedbacks_player2[action]
            if feedback > 0:
                digits = self.action_to_digits(action)
                output += f"  Guessed: {digits[0]} {digits[1]}, Feedback: {feedback}\n"

        return output

    def valid_moves(self):
        return list(range(81))

    def action_to_digits(self, action):
        digit1 = action // 9 + 1  # Digit 1 ranges from 1 to 9
        digit2 = action % 9 + 1  # Digit 2 ranges from 1 to 9
        return np.array([digit1, digit2], dtype=np.int32)

    def digits_to_action(self, digits):
        digit1, digit2 = digits
        action = (digit1 - 1) * 9 + (digit2 - 1)
        return action

    def _get_observation(self):
        obs = np.zeros(167, dtype=np.int32)
        # Secret numbers
        obs[0:2] = self.player1_secret
        obs[2:4] = self.player2_secret
        # Current player
        obs[4] = self.current_player
        # Feedbacks
        obs[5:86] = self.feedbacks_player1  # Indices 5 to 85
        obs[86:167] = self.feedbacks_player2  # Indices 86 to 166
        return obs
