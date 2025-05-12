import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_score=50):
        super(CustomEnv, self).__init__()

        self.target_score = target_score

        # Define action space: 10 possible actions (numbers 1 to 10)
        self.action_space = spaces.Discrete(10)

        # Define observation space: numbers_status (10,), cumulative scores (2,)
        # numbers_status values: -1 (opponent), 0 (available), 1 (current player)
        # cumulative scores range from 0 to 55
        low = np.array([-1] * 10 + [0] * 2, dtype=np.int16)
        high = np.array([1] * 10 + [55] * 2, dtype=np.int16)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int16)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.numbers_status = np.zeros(
            10, dtype=np.int8
        )  # 0: available, 1: player 1, -1: player 2
        self.players_cumulative_scores = [0, 0]  # Index 0: player 1, Index 1: player 2
        self.current_player = 1  # 1 or -1
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_observation(), 0, True, False, {}
        if action < 0 or action >= 10:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}
        if self.numbers_status[action] != 0:
            # Number already taken
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Valid action
        self.numbers_status[action] = self.current_player
        number_value = action + 1  # Numbers are from 1 to 10
        current_player_index = 0 if self.current_player == 1 else 1
        self.players_cumulative_scores[current_player_index] += number_value

        # Check for victory
        if self.players_cumulative_scores[current_player_index] >= self.target_score:
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}
        else:
            reward = 0

        # Switch player
        self.current_player *= -1

        return self._get_observation(), reward, False, False, {}

    def render(self):
        numbers_taken = [
            "-" if s == 0 else ("P1" if s == 1 else "P2") for s in self.numbers_status
        ]
        numbers_display = ["{}:{}".format(i + 1, numbers_taken[i]) for i in range(10)]
        board_str = "Numbers status:\n" + " | ".join(numbers_display) + "\n"
        scores_str = "Player 1 Score: {}\nPlayer 2 Score: {}\n".format(
            self.players_cumulative_scores[0], self.players_cumulative_scores[1]
        )
        current_player_str = "Current Player: {}\n".format(
            "Player 1" if self.current_player == 1 else "Player 2"
        )
        return board_str + scores_str + current_player_str

    def valid_moves(self):
        return [i for i in range(10) if self.numbers_status[i] == 0]

    def _get_observation(self):
        # Adjust numbers_status to the current player's perspective
        obs_numbers_status = (
            self.numbers_status * self.current_player
        )  # Current player's numbers are 1, opponent's -1
        current_player_index = 0 if self.current_player == 1 else 1
        opponent_index = 1 - current_player_index
        current_player_score = self.players_cumulative_scores[current_player_index]
        opponent_score = self.players_cumulative_scores[opponent_index]
        observation = np.concatenate(
            (
                obs_numbers_status,
                np.array([current_player_score, opponent_score], dtype=np.int16),
            )
        )
        return observation
