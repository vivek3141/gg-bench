import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0-19: Guess numbers 1-20
        # Actions 20-39: Ask predefined questions 0-19
        self.action_space = spaces.Discrete(40)

        # Observation space: answers to predefined questions (20 questions)
        # -1: Not yet asked, 0: No, 1: Yes
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select secret numbers for both players (1 to 20)
        self.secret_numbers = [
            self.np_random.randint(1, 21),
            self.np_random.randint(1, 21),
        ]

        # Set current player (0 or 1)
        self.current_player = 0

        # Initialize question answers for both players
        self.question_answers = [
            np.full(20, -1, dtype=np.int8),
            np.full(20, -1, dtype=np.int8),
        ]

        # Flags to indicate if players have used their guess
        self.has_guessed = [False, False]

        self.done = False

        # Return observation for current player
        observation = self.observation()
        info = {}  # Additional info if needed
        return observation, info

    def step(self, action):
        if self.done:
            return self.observation(), -10, True, False, {}  # Game already over

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.observation(), -10, True, False, {}  # Invalid move

        reward = 0
        opponent = 1 - self.current_player

        if action >= 0 and action < 20:
            # Player chooses to guess the opponent's number
            if self.has_guessed[self.current_player]:
                self.done = True
                return self.observation(), -10, True, False, {}  # Already guessed

            guessed_number = action + 1
            self.has_guessed[self.current_player] = True

            if guessed_number == self.secret_numbers[opponent]:
                # Correct guess, player wins
                reward = 1
                self.done = True
            else:
                # Incorrect guess, player loses
                reward = -1
                self.done = True
        else:
            # Player asks a question
            question_idx = action - 20

            # Get the opponent's answer to the question
            answer = self.evaluate_question(self.secret_numbers[opponent], question_idx)

            # Update the current player's knowledge base
            self.question_answers[self.current_player][question_idx] = answer

            # No immediate reward for asking a question
            reward = 0

            # Game continues
            self.done = False

            # Switch turn to opponent
            self.current_player = opponent

        observation = self.observation()
        info = {}
        return (
            observation,
            reward,
            self.done,
            False,
            info,
        )  # obs, reward, terminated, truncated, info

    def observation(self):
        # Return the observation for the current player
        return self.question_answers[self.current_player]

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        actions = []

        # Can guess if haven't already guessed
        if not self.has_guessed[self.current_player]:
            actions.extend(range(0, 20))  # Guess actions

        # Can ask any question (even if previously asked)
        actions.extend(range(20, 40))  # Question actions

        return actions

    def evaluate_question(self, number, question_idx):
        # Implement the predefined questions
        if question_idx == 0:
            # Is your number greater than 10?
            return 1 if number > 10 else 0
        elif question_idx == 1:
            # Is your number even?
            return 1 if number % 2 == 0 else 0
        elif question_idx == 2:
            # Is your number a prime number?
            return 1 if is_prime(number) else 0
        elif question_idx == 3:
            # Is your number a multiple of 3?
            return 1 if number % 3 == 0 else 0
        elif question_idx == 4:
            # Is your number greater than 5?
            return 1 if number > 5 else 0
        elif question_idx == 5:
            # Is your number greater than 15?
            return 1 if number > 15 else 0
        elif question_idx == 6:
            # Is your number less than 5?
            return 1 if number < 5 else 0
        elif question_idx == 7:
            # Is your number less than 15?
            return 1 if number < 15 else 0
        elif question_idx == 8:
            # Is your number a square number?
            return 1 if number in [1, 4, 9, 16] else 0
        elif question_idx == 9:
            # Is your number a multiple of 4?
            return 1 if number % 4 == 0 else 0
        elif question_idx == 10:
            # Is your number a multiple of 5?
            return 1 if number % 5 == 0 else 0
        elif question_idx == 11:
            # Is your number a multiple of 2 and 3?
            return 1 if number % 6 == 0 else 0
        elif question_idx == 12:
            # Is your number less than or equal to 10?
            return 1 if number <= 10 else 0
        elif question_idx == 13:
            # Is your number between 5 and 15 inclusive?
            return 1 if 5 <= number <= 15 else 0
        elif question_idx == 14:
            # Is your number a multiple of 7?
            return 1 if number % 7 == 0 else 0
        elif question_idx == 15:
            # Is your number a power of 2?
            return 1 if number in [1, 2, 4, 8, 16] else 0
        elif question_idx == 16:
            # Is your number in the set {1, 3, 5, 7, 9}?
            return 1 if number in [1, 3, 5, 7, 9] else 0
        elif question_idx == 17:
            # Is your number in the set {11, 13, 17, 19}?
            return 1 if number in [11, 13, 17, 19] else 0
        elif question_idx == 18:
            # Is your number less than 11?
            return 1 if number < 11 else 0
        elif question_idx == 19:
            # Is your number greater than 11?
            return 1 if number > 11 else 0
        else:
            raise ValueError("Invalid question index")

    def render(self):
        # Render the game state for the current player
        observation = self.observation()
        output = f"Current Player: Player {self.current_player + 1}\n"
        output += (
            f"Has Guessed: {'Yes' if self.has_guessed[self.current_player] else 'No'}\n"
        )
        output += "Question Answers:\n"
        for idx, answer in enumerate(observation):
            status = "Not Asked" if answer == -1 else ("Yes" if answer == 1 else "No")
            output += f"  Q{idx}: {status}\n"
        return output


def is_prime(n):
    # Returns True if n is a prime number
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
