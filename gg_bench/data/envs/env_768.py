import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        self.numbers = list(range(1, 10))  # Numbers 1 to 9
        self.num_numbers = len(self.numbers)
        self.actions = []
        for num1, num2 in combinations(self.numbers, 2):
            self.actions.append((num1, num2, "+"))
            self.actions.append((num1, num2, "*"))
        self.action_mapping = {i: action for i, action in enumerate(self.actions)}
        self.action_space = spaces.Discrete(len(self.actions))

        # Define observation space
        # Observation consists of:
        # - Pool vector (positions 0-8): 1 if number is available, 0 if not
        # - Player 1's score (position 9), scaled between 0 and 1
        # - Player 2's score (position 10), scaled between 0 and 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the pool: 1 if number is available, 0 if not
        self.pool = np.ones(9, dtype=np.float32)
        self.available_numbers = set(range(1, 10))
        # Initialize player scores
        self.player1_score = 0
        self.player2_score = 0
        # Current player: 1 or 2
        self.current_player = 1
        self.done = False
        # Prepare the observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Validate action
        if action not in self.valid_moves():
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        num1, num2, operation = self.action_mapping[action]

        # Remove the numbers from the pool
        self.pool[num1 - 1] = 0
        self.pool[num2 - 1] = 0
        self.available_numbers.discard(num1)
        self.available_numbers.discard(num2)

        # Calculate result
        if operation == "+":
            result = num1 + num2
        elif operation == "*":
            result = num1 * num2

        # Update current player's score
        if self.current_player == 1:
            self.player1_score += result
            player_score = self.player1_score
        else:
            self.player2_score += result
            player_score = self.player2_score

        # Check for win condition
        if player_score == 50:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}
        elif player_score > 50:
            # Exceeded 50, invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}
        else:
            # Continue game
            # Check if opponent can make a move
            self.current_player = 3 - self.current_player  # Switch player
            if not self.valid_moves():
                # Opponent cannot make a move
                self.done = True
                reward = 1  # Current player wins
                return self._get_observation(), reward, True, False, {}
            else:
                # Game continues
                reward = 0
                return self._get_observation(), reward, False, False, {}

    def render(self):
        pool_numbers = [str(i + 1) for i in range(9) if self.pool[i] == 1]
        pool_str = ", ".join(pool_numbers)
        return (
            f"Player 1 Score: {self.player1_score}\n"
            f"Player 2 Score: {self.player2_score}\n"
            f"Available Numbers: {pool_str}\n"
            f"Current Player: Player {self.current_player}\n"
        )

    def valid_moves(self):
        valid_actions = []
        # Generate all possible pairs from available numbers
        available_numbers = sorted(list(self.available_numbers))
        for num1, num2 in combinations(available_numbers, 2):
            for operation in ["+", "*"]:
                action = self._get_action_index(num1, num2, operation)
                result = self._calculate_result(num1, num2, operation)
                if self.current_player == 1:
                    new_score = self.player1_score + result
                else:
                    new_score = self.player2_score + result
                if new_score <= 50:
                    valid_actions.append(action)
        return valid_actions

    def _get_action_index(self, num1, num2, operation):
        # Get the action index for given numbers and operation
        try:
            index = self.actions.index((num1, num2, operation))
        except ValueError:
            # Action not in the initial action list, invalid action
            index = None
        return index

    def _calculate_result(self, num1, num2, operation):
        if operation == "+":
            return num1 + num2
        elif operation == "*":
            return num1 * num2

    def _get_observation(self):
        # Pool vector
        pool_vector = self.pool.copy()
        # Scores scaled between 0 and 1
        player1_score_scaled = self.player1_score / 50.0
        player2_score_scaled = self.player2_score / 50.0
        observation = np.concatenate(
            (pool_vector, [player1_score_scaled, player2_score_scaled])
        ).astype(np.float32)
        return observation
