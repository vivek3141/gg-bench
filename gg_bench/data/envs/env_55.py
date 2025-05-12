import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers 0 to 19 represent numbers 1 to 20
        self.action_space = Discrete(20)

        # Observation space:
        # - 20 elements for available numbers (1 if available, 0 if taken)
        # - 4 elements for player 1's sequence (padded with zeros)
        # - 4 elements for player 2's sequence (padded with zeros)
        # - 1 element for the current player (1 or 2)
        self.observation_space = Box(low=0, high=20, shape=(29,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.available_numbers = np.ones(
            20, dtype=np.int32
        )  # 1 if available, 0 if taken
        self.player_sequences = {1: [], 2: []}  # Sequences for both players
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        # Prepare the observation array
        avail_numbers = self.available_numbers.copy()

        # Pad sequences to length 4
        seq1 = self.player_sequences[1] + [0] * (4 - len(self.player_sequences[1]))
        seq2 = self.player_sequences[2] + [0] * (4 - len(self.player_sequences[2]))

        # Include current player
        current_player = np.array([self.current_player], dtype=np.int32)

        # Concatenate all parts to form the observation
        observation = np.concatenate([avail_numbers, seq1, seq2, current_player])
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0
        terminated = False
        truncated = False
        info = {}

        number = action + 1  # Convert action index to number

        # Check if number is available
        if self.available_numbers[action] == 0:
            # Invalid move: number already taken
            reward = -10
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        player_seq = self.player_sequences[self.current_player]

        # Check sequence validity
        if len(player_seq) == 0:
            valid_move = True  # Any number can start the sequence
        else:
            last_number = player_seq[-1]
            if number > last_number and (number - last_number) in [1, 2, 3]:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move: does not satisfy sequence rule
            reward = -10
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Valid move: update state
        self.available_numbers[action] = 0
        player_seq.append(number)

        # Check for win condition
        if len(player_seq) >= 4:
            reward = 1
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        output = ""
        output += "Available Numbers: "
        for idx, val in enumerate(self.available_numbers):
            if val == 1:
                output += f"{idx + 1} "
        output += "\n"
        output += f"Player 1's sequence: {self.player_sequences[1]}\n"
        output += f"Player 2's sequence: {self.player_sequences[2]}\n"
        output += f"Current Player: Player {self.current_player}\n"
        return output

    def valid_moves(self):
        valid_actions = []
        player_seq = self.player_sequences[self.current_player]

        for idx in range(20):
            if self.available_numbers[idx] == 1:
                number = idx + 1
                if len(player_seq) == 0:
                    valid_actions.append(idx)
                else:
                    last_number = player_seq[-1]
                    if number > last_number and (number - last_number) in [1, 2, 3]:
                        valid_actions.append(idx)
        return valid_actions
