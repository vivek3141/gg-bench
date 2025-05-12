import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(35)
        self.observation_space = spaces.Box(low=0, high=10, shape=(26,), dtype=np.int8)

        # Initialize game state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign secret numbers to both players
        self.player_numbers = {
            1: self.np_random.randint(1, 11),
            -1: self.np_random.randint(1, 11),
        }
        # Initialize current player
        self.current_player = 1
        # Questions history: 0 - not asked, 1 - Yes, 2 - No
        self.questions_history = np.zeros(25, dtype=np.int8)
        # Game status
        self.done = False
        # Construct initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Initialize reward
        reward = -10  # Default reward for a valid move
        terminated = False
        truncated = False

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid move results in immediate loss
            reward = -10
            terminated = True
            observation = self._get_observation()
            info = {}
            return observation, reward, terminated, truncated, info

        # Process action
        if action <= 24:
            # Action is a question
            # Get the question and parameter
            question = self._get_question(action)
            opponent_number = self.player_numbers[-self.current_player]
            answer = self._answer_question(question, opponent_number)
            # Update questions history
            self.questions_history[action] = 1 if answer else 2
            # Game continues
            terminated = False
        else:
            # Action is a guess
            guess_number = action - 24
            opponent_number = self.player_numbers[-self.current_player]
            if guess_number == opponent_number:
                # Correct guess, current player wins
                reward = 1
                terminated = True
            else:
                # Incorrect guess, current player loses
                reward = -10
                terminated = True

        # Prepare observation
        observation = self._get_observation()

        # Switch player turn if game is not over
        if not terminated:
            self.current_player *= -1

        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        lines = []
        lines.append(f"Player {self.current_player}'s turn.")
        lines.append(f"Your secret number: {self.player_numbers[self.current_player]}")
        lines.append("Questions asked and answers:")
        for idx, status in enumerate(self.questions_history):
            if status != 0:
                question = self._get_question(idx)
                answer = "Yes" if status == 1 else "No"
                lines.append(f"Q{idx + 1}: {question['text']} Answer: {answer}")
        return "\n".join(lines)

    def valid_moves(self):
        valid_actions = []
        # Add all questions that have not been asked yet
        for idx, status in enumerate(self.questions_history):
            if status == 0:
                valid_actions.append(idx)
        # Add all guesses (indices 25 to 34)
        valid_actions.extend(range(25, 35))
        return valid_actions

    def _get_observation(self):
        obs = np.zeros(26, dtype=np.int8)
        # Player's own secret number
        obs[0] = self.player_numbers[self.current_player]
        # Questions history
        obs[1:] = self.questions_history
        return obs

    def _get_question(self, action_idx):
        # Define the mapping from action index to question
        if 0 <= action_idx <= 8:
            # Is your number greater than X? X = 1 to 9
            X = action_idx + 1
            return {
                "type": "greater",
                "value": X,
                "text": f"Is your number greater than {X}?",
            }
        elif 9 <= action_idx <= 17:
            # Is your number less than X? X = 2 to 10
            X = action_idx - 8 + 1
            return {
                "type": "less",
                "value": X,
                "text": f"Is your number less than {X}?",
            }
        elif 18 <= action_idx <= 21:
            # Is your number divisible by X? X = 2,3,4,5
            X = action_idx - 17 + 1
            return {
                "type": "divisible",
                "value": X + 1,
                "text": f"Is your number divisible by {X + 1}?",
            }
        elif action_idx == 22:
            # Is your number even?
            return {"type": "even", "text": "Is your number even?"}
        elif action_idx == 23:
            # Is your number odd?
            return {"type": "odd", "text": "Is your number odd?"}
        elif action_idx == 24:
            # Is your number a prime number?
            return {"type": "prime", "text": "Is your number a prime number?"}
        else:
            raise ValueError("Invalid action index for a question.")

    def _answer_question(self, question, number):
        if question["type"] == "greater":
            return number > question["value"]
        elif question["type"] == "less":
            return number < question["value"]
        elif question["type"] == "divisible":
            return number % question["value"] == 0
        elif question["type"] == "even":
            return number % 2 == 0
        elif question["type"] == "odd":
            return number % 2 == 1
        elif question["type"] == "prime":
            return self._is_prime(number)
        else:
            return False

    def _is_prime(self, num):
        if num <= 1:
            return False
        elif num <= 3:
            return True
        elif num % 2 == 0:
            return False
        else:
            for i in range(3, int(num**0.5) + 1, 2):
                if num % i == 0:
                    return False
            return True
