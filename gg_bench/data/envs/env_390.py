import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 20 discrete actions representing combinations of numbers (1-10) and operations (add, multiply)
        # Actions 0-19 correspond to:
        # 0: (1, add), 1: (1, multiply), 2: (2, add), ..., 19: (10, multiply)
        self.action_space = spaces.Discrete(20)

        # Define observation space
        # Observation consists of:
        # - Player 1 Score (integer from 0 to 50)
        # - Player 2 Score (integer from 0 to 50)
        # - Counts of available numbers from 1 to 10 (each can be 0, 1, or 2)
        # Total observation space shape: (12,)
        low = np.array([0] * 12)
        high = np.array([50, 50] + [2] * 10)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize number pool: each number from 1 to 10 appears twice
        self.number_pool = {i: 2 for i in range(1, 11)}

        # Initialize player scores
        self.player_scores = {1: 0, -1: 0}

        # Player 1 starts (represented by 1); Player 2 is represented by -1
        self.current_player = 1

        # Game state
        self.done = False

        # Return initial observation and empty info
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        # Observation:
        # [Player 1 Score, Player 2 Score, Counts of numbers 1 to 10]
        obs = np.zeros(12, dtype=np.int32)
        obs[0] = self.player_scores[1]
        obs[1] = self.player_scores[-1]
        counts = [self.number_pool.get(i, 0) for i in range(1, 11)]
        obs[2:] = counts
        return obs

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_observation(), 0, True, False, {}

        # Map action to (number, operation)
        number = (action // 2) + 1  # Numbers 1 to 10
        operation = "add" if action % 2 == 0 else "multiply"

        # Check if the selected number is available in the pool
        if self.number_pool.get(number, 0) <= 0:
            # Invalid move: number not available
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Apply operation
        current_score = self.player_scores[self.current_player]
        if operation == "add":
            new_score = current_score + number
        elif operation == "multiply":
            if current_score == 0:
                new_score = 0  # Multiplying zero results in zero
            else:
                new_score = current_score * number
        else:
            # Invalid operation (should not occur)
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if new score exceeds 50
        if new_score > 50:
            new_score = 0  # Score resets to 0

        # Update player's score
        self.player_scores[self.current_player] = new_score

        # Remove the selected number from the pool
        self.number_pool[number] -= 1
        if self.number_pool[number] == 0:
            del self.number_pool[number]

        # Check for winning condition
        if new_score == 50:
            self.done = True
            # Current player wins
            return self._get_observation(), 1, True, False, {}

        # Check if all numbers have been used
        if sum(self.number_pool.values()) == 0:
            self.done = True
            # Determine the winner based on who is closest to 50 without exceeding it
            opp_player = -self.current_player
            current_player_score = self.player_scores[self.current_player]
            opp_player_score = self.player_scores[opp_player]

            # Compare scores
            if current_player_score == opp_player_score:
                # Last player to have taken a turn loses (current player)
                return self._get_observation(), -1, True, False, {}
            elif current_player_score > opp_player_score:
                # Current player wins
                return self._get_observation(), 1, True, False, {}
            else:
                # Opponent wins; current player loses
                return self._get_observation(), -1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        # Valid move; game continues
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Generate a string representing the current state of the game
        s = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        s += f"Player 1 Score: {self.player_scores[1]}\n"
        s += f"Player 2 Score: {self.player_scores[-1]}\n"
        s += "Available Numbers:\n"
        pool_numbers = []
        for num in range(1, 11):
            count = self.number_pool.get(num, 0)
            pool_numbers.extend([num] * count)
        s += ", ".join(map(str, sorted(pool_numbers)))
        return s

    def valid_moves(self):
        # Return a list of valid action indices based on the available numbers
        valid_actions = []
        for action in range(20):
            number = (action // 2) + 1
            if self.number_pool.get(number, 0) > 0:
                valid_actions.append(action)
        return valid_actions
