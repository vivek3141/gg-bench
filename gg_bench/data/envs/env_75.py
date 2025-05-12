import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 8 possible split positions (between 1 and 8)
        # For each split position, there are 2 choices (left or right part)
        self.action_space = spaces.Discrete(16)
        # Observation space includes:
        # - The sequence (length up to 9, padded with -1)
        # - Player scores (2 values)
        # - Current player indicator (1 or 2)
        # Total length: 9 + 2 + 1 = 12
        self.observation_space = spaces.Box(
            low=-1, high=45, shape=(12,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.random.permutation(np.arange(1, 10)).tolist()
        self.player_scores = {1: 0, 2: 0}
        self.current_player = 1
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to split position and part choice
        split_position = action // 2 + 1  # Split positions are from 1 to 8
        part_choice = action % 2  # 0 for left part, 1 for right part

        # Validate action
        if not (1 <= split_position <= len(self.sequence) - 1):
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Split the sequence
        left_part = self.sequence[:split_position]
        right_part = self.sequence[split_position:]

        # Select the part to keep
        if part_choice == 0:
            selected_part = left_part
            remaining_sequence = right_part
        else:
            selected_part = right_part
            remaining_sequence = left_part

        # Update score
        self.player_scores[self.current_player] += sum(selected_part)

        # Update sequence for the next turn
        self.sequence = remaining_sequence

        # Check if the game has ended
        if len(self.sequence) == 0:
            # Game ends; compare scores
            opponent = 1 if self.current_player == 2 else 2
            if self.player_scores[self.current_player] > self.player_scores[opponent]:
                reward = 1
            else:
                reward = 0
            self.done = True
            return self._get_observation(), reward, True, False, {}
        elif len(self.sequence) == 1:
            # Last number goes to the next player
            self.current_player = 1 if self.current_player == 2 else 2
            self.player_scores[self.current_player] += self.sequence[0]
            self.sequence = []

            # Game ends; compare scores
            if (
                self.player_scores[self.current_player]
                > self.player_scores[1 if self.current_player == 2 else 2]
            ):
                reward = 1
            else:
                reward = 0
            self.done = True
            return self._get_observation(), reward, True, False, {}
        else:
            # Switch player
            self.current_player = 1 if self.current_player == 2 else 2
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def _get_observation(self):
        # Pad the sequence to length 9 with -1
        padded_sequence = self.sequence + [-1] * (9 - len(self.sequence))
        obs = np.array(
            padded_sequence
            + [self.player_scores[1], self.player_scores[2], self.current_player],
            dtype=np.int32,
        )
        return obs

    def render(self):
        sequence_str = ", ".join(str(num) for num in self.sequence)
        return (
            f"Current Sequence: [{sequence_str}]\n"
            f"Player 1 Score: {self.player_scores[1]}\n"
            f"Player 2 Score: {self.player_scores[2]}\n"
            f"Current Player: Player {self.current_player}\n"
        )

    def valid_moves(self):
        valid_actions = []
        n = len(self.sequence)
        if n >= 2:
            for split_position in range(1, n):
                for part_choice in [0, 1]:
                    action = (split_position - 1) * 2 + part_choice
                    valid_actions.append(action)
        return valid_actions
