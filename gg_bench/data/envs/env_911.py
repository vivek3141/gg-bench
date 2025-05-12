import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # The action space is Discrete(20), actions are integers from 0 to 19
        # Corresponding to selecting numbers from 1 to 20
        self.action_space = spaces.Discrete(20)

        # The observation space is a Box of shape (22,)
        # [0]: current player's total score (0 to 50)
        # [1]: opponent's total score (0 to 50)
        # [2:22]: availability of numbers 1 to 20 (1 if available, 0 if not)
        low = np.array([0, 0] + [0] * 20, dtype=np.int32)
        high = np.array([50, 50] + [1] * 20, dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.ones(20, dtype=np.int32)  # Numbers from 1 to 20 available
        self.scores = [0, 0]  # Both players start with 0
        self.current_player = 0  # Start with player 0
        self.done = False
        # Observation is [current player score, opponent score, number pool]
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        if self.done:
            # If the game is over, no further moves can be made
            return self._get_obs(), 0, True, False, {}
        selected_number = action + 1  # Number selected is from 1 to 20
        # Check if the number is available
        if self.number_pool[action] == 0:
            # Invalid move: number not available
            self.done = True
            reward = -10
            terminated = True
            return self._get_obs(), reward, terminated, False, {}
        # Check if adding the number exceeds 50
        if self.scores[self.current_player] + selected_number > 50:
            # Invalid move: total would exceed 50
            self.done = True
            reward = -10
            terminated = True
            return self._get_obs(), reward, terminated, False, {}
        # Valid move
        self.scores[self.current_player] += selected_number
        self.number_pool[action] = 0  # Remove the number from the pool
        # Check if current player's total score is exactly 50 (win condition)
        if self.scores[self.current_player] == 50:
            self.done = True
            reward = 1
            terminated = True
            return self._get_obs(), reward, terminated, False, {}
        # Switch to the next player
        self.current_player = 1 - self.current_player
        # Return observation, reward, terminated, truncated, info
        return self._get_obs(), 0, False, False, {}

    def render(self):
        render_str = f"Player {self.current_player + 1}'s turn\n"
        render_str += f"Player 1 Score: {self.scores[0]}\n"
        render_str += f"Player 2 Score: {self.scores[1]}\n"
        available_numbers = [str(i + 1) for i in range(20) if self.number_pool[i] == 1]
        render_str += f'Available Numbers:\n{", ".join(available_numbers)}\n'
        return render_str

    def valid_moves(self):
        # Return a list of indices of valid moves
        valid_actions = []
        for i in range(20):
            if self.number_pool[i] == 1:
                number = i + 1
                if self.scores[self.current_player] + number <= 50:
                    valid_actions.append(i)
        return valid_actions

    def _get_obs(self):
        # Observation is [current player score, opponent score, number pool]
        observation = np.concatenate(
            (
                np.array(
                    [
                        self.scores[self.current_player],
                        self.scores[1 - self.current_player],
                    ]
                ),
                self.number_pool,
            )
        )
        return observation
