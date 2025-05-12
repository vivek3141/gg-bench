import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space: numbers from 1 to 10, actions 0 to 9
        self.action_space = spaces.Discrete(10)

        # Define observation space: [current player's score, opponent's score, last number used against current player, last number used against opponent]
        # Scores can range from 0 to some maximum, we'll set an upper bound higher than the starting score
        # Last numbers used range from 0 (no number used yet) to 10
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1000, 1000, 10, 10]),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = [100, 100]  # Index 0: Player 1, Index 1: Player 2
        self.last_numbers_used = [
            0,
            0,
        ]  # Index 0: Last number used against Player 1, Index 1: Last number used against Player 2
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        # Observation is from the perspective of current player
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def _get_observation(self):
        # Returns the observation from the perspective of the current player
        current_score = self.player_scores[self.current_player]
        opponent_score = self.player_scores[1 - self.current_player]
        last_number_used_against_current = self.last_numbers_used[self.current_player]
        last_number_used_against_opponent = self.last_numbers_used[
            1 - self.current_player
        ]
        observation = np.array(
            [
                current_score,
                opponent_score,
                last_number_used_against_current,
                last_number_used_against_opponent,
            ],
            dtype=np.int32,
        )
        return observation

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")
        # Convert action to number N (1 to 10)
        N = action + 1  # Actions 0-9 correspond to numbers 1-10
        # Check if the action is valid
        last_number_used_against_current = self.last_numbers_used[self.current_player]
        if N == last_number_used_against_current:
            # Invalid move, turn is forfeited
            reward = -10
            terminated = False
            # Do not change scores or last numbers
            # Switch to the next player
            self.current_player = 1 - self.current_player
        else:
            # Valid move
            # Subtract N from opponent's score
            self.player_scores[1 - self.current_player] -= N
            # Update last number used against opponent
            self.last_numbers_used[1 - self.current_player] = N
            # Check for victory
            if self.player_scores[1 - self.current_player] <= 0:
                # Current player wins
                reward = 1
                terminated = True
                self.done = True
            else:
                reward = 0
                terminated = False
                # Swap players
                self.current_player = 1 - self.current_player
        # Generate observation
        observation = self._get_observation()
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        # Return a string representation of the current state
        s = f"Player {self.current_player + 1}'s turn\n"
        s += f"Player 1 Score: {self.player_scores[0]}\n"
        s += f"Player 2 Score: {self.player_scores[1]}\n"
        s += f"Last number used against Player 1: {self.last_numbers_used[0]}\n"
        s += f"Last number used against Player 2: {self.last_numbers_used[1]}\n"
        return s

    def valid_moves(self):
        # Returns list of valid actions for the current player
        # Valid actions are numbers from 1 to 10 (actions 0 to 9), excluding the last number used against the current player
        last_number_used_against_current = self.last_numbers_used[self.current_player]
        valid_actions = [
            i for i in range(10) if (i + 1) != last_number_used_against_current
        ]
        return valid_actions
