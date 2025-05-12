import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Request a clue, 1-30 - Guess the number n (n from 1 to 30)
        self.action_space = spaces.Discrete(31)

        # Observation space:
        # Clues received by both players: 2 * M (14 clues each)
        # Guesses made by both players: 2 * N (30 numbers each)
        # Total observation length: 2*(14+30) = 88
        self.observation_space = spaces.Box(low=0, high=1, shape=(88,), dtype=np.int8)

        # Possible clues (14 clues)
        self.clues = [
            "The number is even.",
            "The number is odd.",
            "The number is less than 10.",
            "The number is between 10 and 20 inclusive.",
            "The number is greater than 20.",
            "The number is divisible by 2.",
            "The number is divisible by 3.",
            "The number is divisible by 5.",
            "The number is prime.",
            "The number is not prime.",
            "The number is a perfect square.",
            "The number is not a perfect square.",
            "The number is less than 15.",
            "The number is greater than or equal to 15.",
        ]
        self.M = len(self.clues)
        self.N = 30  # Numbers from 1 to 30
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Initialize codes for both players
        self.codes = [rng.integers(1, 31), rng.integers(1, 31)]
        # Initialize clues for both players
        self.clue_lists = []
        for code in self.codes:
            valid_clues = self.get_valid_clues(code)
            clue_indices = rng.choice(valid_clues, size=3, replace=False)
            self.clue_lists.append(clue_indices.tolist())

        # Clues received and guesses made
        self.clues_received = [
            np.zeros(self.M, dtype=np.int8),
            np.zeros(self.M, dtype=np.int8),
        ]
        self.guesses_made = [
            np.zeros(self.N, dtype=np.int8),
            np.zeros(self.N, dtype=np.int8),
        ]

        self.current_player = 0  # 0 or 1
        self.clues_given = [0, 0]  # Number of clues given to each player
        self.done = False

        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {}

        reward = 0
        info = {}
        opponent = 1 - self.current_player

        # Action 0: Request a clue
        if action == 0:
            if self.clues_given[self.current_player] >= 3:
                # Invalid action
                self.done = True
                reward = -10
                return self.get_observation(), reward, True, False, info
            else:
                # Provide next clue from opponent's clue list
                clue_index = self.clue_lists[opponent][
                    self.clues_given[self.current_player]
                ]
                self.clues_received[self.current_player][clue_index] = 1
                self.clues_given[self.current_player] += 1
                self.switch_player()
                return self.get_observation(), reward, False, False, info

        elif 1 <= action <= 30:
            guess = action
            code_opponent = self.codes[opponent]
            self.guesses_made[self.current_player][guess - 1] = 1
            if guess == code_opponent:
                # Correct guess, current player wins
                reward = 1
                self.done = True
                return self.get_observation(), reward, True, False, info
            else:
                # Incorrect guess
                self.switch_player()
                return self.get_observation(), reward, False, False, info
        else:
            # Invalid action
            self.done = True
            reward = -10
            return self.get_observation(), reward, True, False, info

    def render(self):
        opponent = 1 - self.current_player
        output = ""
        output += f"Player {self.current_player + 1}'s turn.\n"
        output += "Your clues received:\n"
        received_clues_indices = np.where(
            self.clues_received[self.current_player] == 1
        )[0]
        for idx in received_clues_indices:
            output += f"- {self.clues[idx]}\n"
        output += "Your guesses made:\n"
        guesses_made_indices = np.where(self.guesses_made[self.current_player] == 1)[0]
        for idx in guesses_made_indices:
            output += f"- {idx + 1}\n"
        output += "Opponent's guesses made:\n"
        opponent_guesses_indices = np.where(self.guesses_made[opponent] == 1)[0]
        for idx in opponent_guesses_indices:
            output += f"- {idx + 1}\n"
        return output

    def valid_moves(self):
        valid_actions = []
        if self.clues_given[self.current_player] < 3:
            valid_actions.append(0)  # Request a clue
        valid_actions.extend(range(1, 31))  # Guess numbers 1-30
        return valid_actions

    def get_observation(self):
        observation = np.concatenate(
            (
                self.clues_received[0],
                self.guesses_made[0],
                self.clues_received[1],
                self.guesses_made[1],
            )
        )
        return observation.astype(np.float32)

    def switch_player(self):
        self.current_player = 1 - self.current_player

    def get_valid_clues(self, code):
        valid_clues = []
        if code % 2 == 0:
            valid_clues.append(0)  # Even
        else:
            valid_clues.append(1)  # Odd
        if code < 10:
            valid_clues.append(2)
        if 10 <= code <= 20:
            valid_clues.append(3)
        if code > 20:
            valid_clues.append(4)
        if code % 2 == 0:
            valid_clues.append(5)
        if code % 3 == 0:
            valid_clues.append(6)
        if code % 5 == 0:
            valid_clues.append(7)
        if self.is_prime(code):
            valid_clues.append(8)
        else:
            valid_clues.append(9)
        if self.is_perfect_square(code):
            valid_clues.append(10)
        else:
            valid_clues.append(11)
        if code < 15:
            valid_clues.append(12)
        else:
            valid_clues.append(13)
        return valid_clues

    @staticmethod
    def is_prime(n):
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

    @staticmethod
    def is_perfect_square(n):
        return int(np.sqrt(n)) ** 2 == n
