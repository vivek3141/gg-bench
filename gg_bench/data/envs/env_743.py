import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            10
        )  # Actions from 0 to 9, representing numbers 1 to 10

        # Observation: [current_player (1 or 2), player1_score, player2_score]
        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0]), high=np.array([2, 50, 50]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_score = 0
        self.player2_score = 0
        self.current_player = 1  # 1 or 2
        self.done = False
        observation = np.array(
            [self.current_player, self.player1_score, self.player2_score],
            dtype=np.int32,
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over, cannot take any more actions
            observation = np.array(
                [self.current_player, self.player1_score, self.player2_score],
                dtype=np.int32,
            )
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        if action < 0 or action >= 10:
            # Invalid action
            observation = np.array(
                [self.current_player, self.player1_score, self.player2_score],
                dtype=np.int32,
            )
            self.done = True
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        selected_number = action + 1  # Map action index to number 1-10

        reward = 0

        # Apply game rules
        if selected_number % 2 == 1:
            # Odd number, add to current player's own score
            if self.current_player == 1:
                self.player1_score += selected_number
            else:
                self.player2_score += selected_number
        else:
            # Even number, add to opponent's score
            if self.current_player == 1:
                self.player2_score += selected_number
            else:
                self.player1_score += selected_number

        # Check for win/loss conditions
        if self.current_player == 1:
            # Check for Player 1
            if self.player1_score == 25:
                reward = 1  # Win
                self.done = True
            elif self.player1_score > 25:
                reward = -1  # Lose
                self.done = True
            elif self.player2_score > 25:
                reward = 1  # Player 1 wins because Player 2 exceeded 25
                self.done = True
            elif self.player2_score == 25:
                reward = -1  # Player 1 loses because Player 2 reached exactly 25
                self.done = True
        else:
            # Check for Player 2
            if self.player2_score == 25:
                reward = 1  # Win
                self.done = True
            elif self.player2_score > 25:
                reward = -1  # Lose
                self.done = True
            elif self.player1_score > 25:
                reward = 1  # Player 2 wins because Player 1 exceeded 25
                self.done = True
            elif self.player1_score == 25:
                reward = -1  # Player 2 loses because Player 1 reached exactly 25
                self.done = True

        # Prepare observation
        observation = np.array(
            [self.current_player, self.player1_score, self.player2_score],
            dtype=np.int32,
        )

        if not self.done:
            # Switch current player
            self.current_player = 2 if self.current_player == 1 else 1

        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        output = f"Current Player: Player {self.current_player}\n"
        output += f"Player 1 Score: {self.player1_score}\n"
        output += f"Player 2 Score: {self.player2_score}\n"
        return output

    def valid_moves(self):
        return list(
            range(10)
        )  # All moves from 0 to 9 are valid (representing numbers 1 to 10)
