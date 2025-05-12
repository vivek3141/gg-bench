import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: pick from start, 1: pick from end

        # Observation space: shared_sequence (20,) + player_number (10,) + opponent_number (10,)
        # Total: (40,) array
        self.observation_space = spaces.Box(low=-1, high=9, shape=(40,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate shared_sequence: list of 20 random digits 0-9
        self.shared_sequence = self.np_random.integers(0, 10, size=20).tolist()
        # Initialize player_numbers
        self.player_numbers = {1: [], 2: []}
        # Current player
        self.current_player = 1
        # Who started first
        self.first_player = self.current_player
        # Game over flag
        self.done = False
        return self._get_observation(), {}

    def _get_observation(self):
        # Prepare observation
        # Pad the sequences to fixed length
        shared_sequence_padded = self.shared_sequence + [-1] * (
            20 - len(self.shared_sequence)
        )
        player_number_padded = self.player_numbers[self.current_player] + [-1] * (
            10 - len(self.player_numbers[self.current_player])
        )
        opponent_player = 1 if self.current_player == 2 else 2
        opponent_number_padded = self.player_numbers[opponent_player] + [-1] * (
            10 - len(self.player_numbers[opponent_player])
        )
        observation = np.array(
            shared_sequence_padded + player_number_padded + opponent_number_padded,
            dtype=np.int32,
        )
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0
        terminated = False
        truncated = False

        # Check if action is valid
        if action not in [0, 1]:
            # Invalid action
            self.done = True
            terminated = True
            return self._get_observation(), -10, True, False, {}

        # Check if shared_sequence is empty before action
        if len(self.shared_sequence) == 0:
            # No more digits to pick, game should have been over
            self.done = True
            terminated = True
            # Determine winner
            player_num = int(
                "".join(map(str, self.player_numbers[self.current_player]))
            )
            opponent_player = 1 if self.current_player == 2 else 2
            opponent_num = int("".join(map(str, self.player_numbers[opponent_player])))
            if player_num > opponent_num:
                reward = 1  # Current player wins
            elif player_num < opponent_num:
                reward = -1
            else:
                # Numbers are equal, second player wins
                if self.current_player != self.first_player:
                    reward = 1  # Current player wins
                else:
                    reward = -1
            return self._get_observation(), reward, True, False, {}

        # Process action
        if action == 0:
            # pick from start
            digit = self.shared_sequence.pop(0)
        elif action == 1:
            # pick from end
            digit = self.shared_sequence.pop(-1)

        # Append to player's number
        self.player_numbers[self.current_player].append(digit)

        # Check if shared_sequence is now empty
        if len(self.shared_sequence) == 0:
            # Game over after current player's move
            self.done = True
            terminated = True
            # Determine winner
            player_num = int(
                "".join(map(str, self.player_numbers[self.current_player]))
            )
            opponent_player = 1 if self.current_player == 2 else 2
            opponent_num = int(
                "".join(map(str, self.player_numbers[self.current_player]))
            )
            opponent_num = int("".join(map(str, self.player_numbers[opponent_player])))
            if player_num > opponent_num:
                reward = 1  # Current player wins
            elif player_num < opponent_num:
                reward = -1
            else:
                # Numbers are equal, second player wins
                if self.current_player != self.first_player:
                    reward = 1  # Current player wins
                else:
                    reward = -1
        else:
            # Game not over, reward is 0
            reward = 0
            terminated = False

        # Switch turns
        self.current_player = 1 if self.current_player == 2 else 2

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        output = "Shared Sequence: {}\n".format(
            " ".join(map(str, self.shared_sequence))
        )
        output += "Player 1's Number: {}\n".format(
            "".join(map(str, self.player_numbers[1]))
        )
        output += "Player 2's Number: {}\n".format(
            "".join(map(str, self.player_numbers[2]))
        )
        output += "Current Player: Player {}\n".format(self.current_player)
        return output

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1]
