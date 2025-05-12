import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_height=15):
        super(CustomEnv, self).__init__()
        self.target_height = target_height

        # Define action and observation space
        # Actions correspond to selecting numbers from 1 to 9 (indexes 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space consists of:
        # - Current player's used numbers (9 binary indicators)
        # - Current player's total height (scalar)
        # - Opponent's used numbers (9 binary indicators)
        # - Opponent's total height (scalar)
        low = np.array([0] * 20)
        high = np.array([1] * 9 + [self.target_height] + [1] * 9 + [self.target_height])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0  # Player 0 starts
        self.player_used_numbers = [
            np.zeros(9, dtype=np.int32),  # Player 0's used numbers
            np.zeros(9, dtype=np.int32),  # Player 1's used numbers
        ]
        self.player_total_heights = [0, 0]  # Player 0's and Player 1's total heights
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Please reset the environment.")

        reward = 0
        terminated = False
        truncated = False

        # Check if the current player has any valid moves
        valid_moves_list = self.valid_moves()
        if not valid_moves_list:
            # Current player loses
            reward = -10
            terminated = True
            observation = self._get_observation()
            return observation, reward, terminated, truncated, {}

        # Validate the action
        if action not in valid_moves_list:
            # Invalid move
            reward = -10
            terminated = True
            observation = self._get_observation()
            return observation, reward, terminated, truncated, {}

        # Apply the action
        number_chosen = action + 1  # Numbers are from 1 to 9
        self.player_used_numbers[self.current_player][action] = 1
        self.player_total_heights[self.current_player] += number_chosen

        # Check if current player wins by reaching the target height
        if self.player_total_heights[self.current_player] == self.target_height:
            # Current player wins
            reward = 1
            terminated = True
            observation = self._get_observation()
            return observation, reward, terminated, truncated, {}

        # Check if the next player has any valid moves
        next_player = 1 - self.current_player
        self.current_player = next_player  # Switch to next player
        if not self.valid_moves():
            # Next player loses, current player wins
            reward = 1
            terminated = True
            observation = self._get_observation()
            return observation, reward, terminated, truncated, {}

        # Game continues
        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}

    def render(self):
        player_0_numbers = [
            str(i + 1) for i in range(9) if self.player_used_numbers[0][i] == 1
        ]
        player_1_numbers = [
            str(i + 1) for i in range(9) if self.player_used_numbers[1][i] == 1
        ]
        render_str = f"Player {self.current_player + 1}'s turn.\n"
        render_str += "-------------------------------\n"
        render_str += f"Player 1's Tower: {', '.join(player_0_numbers)}\n"
        render_str += f"Total Height: {self.player_total_heights[0]}\n"
        render_str += "-------------------------------\n"
        render_str += f"Player 2's Tower: {', '.join(player_1_numbers)}\n"
        render_str += f"Total Height: {self.player_total_heights[1]}\n"
        render_str += "-------------------------------\n"
        return render_str

    def valid_moves(self):
        valid_actions = []
        current_used_numbers = self.player_used_numbers[self.current_player]
        current_total_height = self.player_total_heights[self.current_player]
        for action in range(9):
            number = action + 1
            if (
                current_used_numbers[action] == 0
                and current_total_height + number <= self.target_height
            ):
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Prepare the observation from the perspective of the current player
        own_used_numbers = self.player_used_numbers[self.current_player]
        own_total_height = self.player_total_heights[self.current_player]
        opponent = 1 - self.current_player
        opponent_used_numbers = self.player_used_numbers[opponent]
        opponent_total_height = self.player_total_heights[opponent]
        observation = np.concatenate(
            [
                own_used_numbers,
                [own_total_height],
                opponent_used_numbers,
                [opponent_total_height],
            ]
        ).astype(np.int32)
        return observation
