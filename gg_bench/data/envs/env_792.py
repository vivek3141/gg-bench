import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action_space is a Discrete space of 125 possible guesses (5*5*5)
        self.action_space = spaces.Discrete(125)

        # The observation_space is a Box space containing:
        # - Current player's last guess digits (3 integers from 0 to 5)
        # - Feedback received (hits and misses, integers from 0 to 3)
        # - Opponent's last guess digits (3 integers from 0 to 5)
        # - Feedback given to opponent (hits and misses, integers from 0 to 3)
        # - Turn count (integer)
        # - Current player ID (0 or 1)
        self.observation_space = spaces.Box(
            low=np.array([0] * 12),
            high=np.array([5] * 3 + [3] * 2 + [5] * 3 + [3] * 2 + [1000, 1]),
            shape=(12,),
            dtype=np.int8,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.secret_code_p1 = np.random.randint(1, 6, size=3)
        self.secret_code_p2 = np.random.randint(1, 6, size=3)
        self.last_guess_p1 = np.array([0, 0, 0], dtype=np.int8)
        self.last_feedback_p1 = np.array([0, 0], dtype=np.int8)  # Hits, Misses
        self.last_guess_p2 = np.array([0, 0, 0], dtype=np.int8)
        self.last_feedback_p2 = np.array([0, 0], dtype=np.int8)  # Hits, Misses
        self.turn_count = 0
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map the action index to the guess digits
        guess = self._index_to_code(action)
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Check for valid action
        if not self._is_valid_guess(guess):
            reward = -10
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Process the guess
        if self.current_player == 0:
            # Player 1's turn
            self.last_guess_p1 = guess
            feedback = self._compute_feedback(guess, self.secret_code_p2)
            self.last_feedback_p1 = np.array(feedback, dtype=np.int8)
            # Check for win
            if feedback[0] == 3:
                reward = 1
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, info
        else:
            # Player 2's turn
            self.last_guess_p2 = guess
            feedback = self._compute_feedback(guess, self.secret_code_p1)
            self.last_feedback_p2 = np.array(feedback, dtype=np.int8)
            # Check for win
            if feedback[0] == 3:
                reward = 1
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, info

        # No win, continue the game
        self.turn_count += 1
        # Switch to the other player
        self.current_player = 1 - self.current_player

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        output = ""
        output += f"Turn: {self.turn_count}\n"
        output += f"Current Player: {'Player 1' if self.current_player == 0 else 'Player 2'}\n"
        output += (
            "Player 1's last guess: " + " ".join(map(str, self.last_guess_p1)) + "\n"
        )
        output += f"Player 1's feedback: Hits: {self.last_feedback_p1[0]} Misses: {self.last_feedback_p1[1]}\n"
        output += (
            "Player 2's last guess: " + " ".join(map(str, self.last_guess_p2)) + "\n"
        )
        output += f"Player 2's feedback: Hits: {self.last_feedback_p2[0]} Misses: {self.last_feedback_p2[1]}\n"
        return output

    def valid_moves(self):
        if self.done:
            return []
        return list(range(125))

    # Helper functions
    def _index_to_code(self, index):
        d1 = index // 25
        index = index % 25
        d2 = index // 5
        d3 = index % 5
        return np.array([d1 + 1, d2 + 1, d3 + 1], dtype=np.int8)

    def _is_valid_guess(self, guess):
        return np.all((guess >= 1) & (guess <= 5))

    def _compute_feedback(self, guess, code):
        hits = np.sum(guess == code)
        code_counts = np.bincount(code, minlength=6)
        guess_counts = np.bincount(guess, minlength=6)
        common_counts = np.minimum(code_counts, guess_counts)
        total_matches = np.sum(common_counts[1:])
        misses = total_matches - hits
        return (hits, misses)

    def _get_observation(self):
        observation = np.concatenate(
            [
                self.last_guess_p1 if self.current_player == 0 else self.last_guess_p2,
                (
                    self.last_feedback_p1
                    if self.current_player == 0
                    else self.last_feedback_p2
                ),
                self.last_guess_p2 if self.current_player == 0 else self.last_guess_p1,
                (
                    self.last_feedback_p2
                    if self.current_player == 0
                    else self.last_feedback_p1
                ),
                np.array([self.turn_count, self.current_player], dtype=np.int8),
            ]
        )
        return observation
