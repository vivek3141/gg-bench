import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            low=np.zeros(13, dtype=np.float32),
            high=np.array([1] * 10 + [200, 200, 20], dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate the initial number list
        self.number_list = self.np_random.choice(
            np.arange(1, 21), size=10, replace=False
        )
        self.available_numbers = np.ones(10, dtype=np.int32)

        # Initialize scores
        self.scores = {1: 0, 2: 0}

        # Initialize last number selected for each player
        self.last_number_selected = {1: 0, 2: 0}

        # Set the current player (1 or 2)
        self.current_player = 1

        # Game not over
        self.done = False

        # Form initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_observation(), 0, True, False, {}

        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        if self.available_numbers[action] == 0:
            # Number already selected
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid action, process the move
        selected_number = self.number_list[action]

        # Update current player's score
        self.scores[self.current_player] += selected_number

        # Remove the selected number from available numbers
        self.available_numbers[action] = 0

        # Check for Snatch opportunity
        opponent = 3 - self.current_player
        opponent_last_number = self.last_number_selected[opponent]

        if opponent_last_number != 0 and selected_number % opponent_last_number == 0:
            # Snatch occurs
            snatch_points = min(opponent_last_number, self.scores[opponent])
            self.scores[opponent] -= snatch_points
            self.scores[self.current_player] += snatch_points

        # Update last number selected
        self.last_number_selected[self.current_player] = selected_number

        # Check if the game is over
        if np.sum(self.available_numbers) == 0:
            # Game over
            self.done = True
            if self.scores[self.current_player] > self.scores[opponent]:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses or ties
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player = opponent

        return self._get_observation(), 0, False, False, {}

    def render(self):
        output = "Available Numbers: {}\n".format(
            [
                num
                for idx, num in enumerate(self.number_list)
                if self.available_numbers[idx] == 1
            ]
        )
        output += "Player 1 Score: {}\n".format(self.scores[1])
        output += "Player 2 Score: {}\n".format(self.scores[2])
        output += "Player {}'s turn.\n".format(self.current_player)
        return output

    def valid_moves(self):
        return [i for i in range(10) if self.available_numbers[i] == 1]

    def _get_observation(self):
        # Concatenate available numbers, current player's score, opponent's score, and opponent's last selected number
        observation = np.concatenate(
            [
                self.available_numbers.astype(np.float32),
                np.array(
                    [
                        self.scores[self.current_player],
                        self.scores[3 - self.current_player],
                        self.last_number_selected[3 - self.current_player],
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return observation
