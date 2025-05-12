import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 27 possible guesses (numbers from 0 to 26)
        self.action_space = spaces.Discrete(27)

        # Define observation space:
        # We keep track of up to 'max_turns' turns.
        # Each turn consists of: [player, digit1, digit2, digit3, feedback]
        self.max_turns = 20  # Maximum number of turns in the game
        self.observation_space = spaces.Box(
            low=-1, high=3, shape=(self.max_turns, 5), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts (1 or -1)
        self.turn_counter = 0
        self.done = False
        self.truncated = False

        # Secret codes for each player (digits from 1 to 3)
        self.player_codes = {
            1: self.generate_secret_code(),
            -1: self.generate_secret_code(),
        }

        # Initialize history with -1 (unused slots)
        self.history = np.full((self.max_turns, 5), -1, dtype=np.int32)

        return self.history, {}  # Return observation and info

    def generate_secret_code(self):
        # Generate a secret code with digits from 1 to 3
        return np.random.randint(1, 4, size=3)  # Array of 3 digits

    def action_to_guess(self, action):
        # Map action (0-26) to guess (array of 3 digits between 1 and 3)
        d1 = action // 9  # 0-2
        d2 = (action % 9) // 3  # 0-2
        d3 = action % 3  # 0-2
        return np.array([d1 + 1, d2 + 1, d3 + 1], dtype=np.int32)

    def step(self, action):
        if self.done:
            return self.history, 0, True, self.truncated, {}

        # Validate action
        if action not in self.valid_moves():
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            return self.history, reward, terminated, truncated, {}

        # Convert action to guess
        guess = self.action_to_guess(action)

        # Opponent's secret code
        opponent = -self.current_player
        secret_code = self.player_codes[opponent]

        # Calculate feedback (number of exact matches)
        exact_matches = np.sum(guess == secret_code)

        # Update history
        if self.turn_counter < self.max_turns:
            self.history[self.turn_counter] = [
                self.current_player,
                guess[0],
                guess[1],
                guess[2],
                exact_matches,
            ]
        else:
            # Exceeded maximum number of turns
            self.done = True
            reward = 0
            terminated = True
            truncated = True
            return self.history, reward, terminated, truncated, {}

        self.turn_counter += 1

        # Check for win
        if exact_matches == 3:
            self.done = True
            reward = 1
            terminated = True
            truncated = False
            return self.history, reward, terminated, truncated, {}

        # Switch to next player
        self.current_player *= -1

        return self.history, 0, False, False, {}

    def render(self):
        # Build a string representation of the game state
        s = "Turn\tPlayer\tGuess\tFeedback\n"
        for i in range(self.turn_counter):
            player = self.history[i][0]
            guess = self.history[i][1:4]
            feedback = self.history[i][4]
            s += f"{i + 1}\t{player}\t{guess[0]}-{guess[1]}-{guess[2]}\t{feedback} exact matches\n"
        return s

    def valid_moves(self):
        # All actions from 0 to 26 are valid in this game
        return list(range(27))
