import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Add, 1 - Multiply
        self.action_space = spaces.Discrete(2)

        # Observations: [current player's score, opponent's score]
        self.observation_space = spaces.Box(low=0, high=50, shape=(2,), dtype=np.int32)

        # Initialize the environment state
        self.scores = [0, 0]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the random number generator
        self.np_random, seed = seeding.np_random(seed)

        # Reset the game state
        self.scores = [0, 0]
        self.current_player = self.np_random.integers(0, 2)  # Random starting player
        self.done = False

        # Prepare the initial observation
        own_score = self.scores[self.current_player]
        opponent_score = self.scores[1 - self.current_player]
        observation = np.array([own_score, opponent_score], dtype=np.int32)

        return observation, {}

    def step(self, action):
        if self.done:
            return (
                np.array(
                    [
                        self.scores[self.current_player],
                        self.scores[1 - self.current_player],
                    ],
                    dtype=np.int32,
                ),
                0,
                self.done,
                False,
                {},
            )

        # Validate action
        if action not in [0, 1]:
            return (
                np.array(
                    [
                        self.scores[self.current_player],
                        self.scores[1 - self.current_player],
                    ],
                    dtype=np.int32,
                ),
                -10,
                True,
                False,
                {},
            )  # Invalid action ends the game

        # Perform action
        if action == 0:
            # Add
            rand_num = self.np_random.integers(1, 11)  # 1 to 10 inclusive
            new_score = self.scores[self.current_player] + rand_num
        elif action == 1:
            # Multiply
            rand_num = self.np_random.integers(2, 6)  # 2 to 5 inclusive
            new_score = self.scores[self.current_player] * rand_num

        # Check for score reset
        if new_score > 50:
            new_score = 0

        # Update the current player's score
        self.scores[self.current_player] = new_score

        # Check for win condition
        if new_score == 50:
            self.done = True
            reward = 1
        else:
            reward = -10  # Penalty for valid move

        # Prepare the observation
        own_score = self.scores[self.current_player]
        opponent_score = self.scores[1 - self.current_player]
        observation = np.array([own_score, opponent_score], dtype=np.int32)

        # Switch players if the game is not over
        if not self.done:
            self.current_player = 1 - self.current_player

        return observation, reward, self.done, False, {}

    def render(self):
        return f"Player 1 Score: {self.scores[0]}, Player 2 Score: {self.scores[1]}"

    def valid_moves(self):
        return [0, 1]
