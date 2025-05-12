import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: pick left, 1: pick right

        # Observation space is an array of length 12
        # Indices 0-9: sequence (zeros where numbers have been removed)
        # Index 10: player's total sum
        # Index 11: opponent's total sum
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(12,), dtype=np.int32
        )

        # Initialize the environment's state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np_random, seed = gym.utils.seeding.np_random(seed)
        self.np_random = np_random

        # Initialize sequence
        length = self.np_random.choice([6, 8, 10])  # Even number between 6 and 10
        sequence_list = list(self.np_random.randint(1, 10, size=length))
        self.sequence = deque(sequence_list)

        # Initialize full_sequence for observation
        self.full_sequence = list(self.sequence) + [0] * (10 - length)

        self.player_sums = {1: 0, -1: 0}  # Players 1 and -1
        self.current_player = 1
        self.done = False

        # Return initial observation
        observation = self.get_observation()
        return observation, {}  # Return observation and info

    def get_observation(self):
        obs_sequence = np.array(self.full_sequence, dtype=np.int32)
        obs_player_total = self.player_sums[self.current_player]
        obs_opponent_total = self.player_sums[-self.current_player]
        observation = np.concatenate(
            (obs_sequence, [obs_player_total, obs_opponent_total])
        )
        return observation

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.get_observation(), -10, True, {}

        # Take the action
        if action == 0:  # pick leftmost
            picked_number = self.sequence.popleft()
        elif action == 1:  # pick rightmost
            picked_number = self.sequence.pop()

        # Update full_sequence for observation
        current_length = len(self.sequence)
        self.full_sequence = list(self.sequence) + [0] * (10 - current_length)

        # Update player's total sum
        self.player_sums[self.current_player] += picked_number

        # Check if game is over
        if not self.sequence:
            self.done = True
            player_total = self.player_sums[self.current_player]
            opponent_total = self.player_sums[-self.current_player]
            if player_total > opponent_total:
                reward = 1
            elif player_total < opponent_total:
                reward = -1
            else:
                # Tiebreaker: the last player to pick loses in case of a tie
                reward = -1
            return self.get_observation(), reward, True, {}

        # Switch current player
        self.current_player *= -1

        # Return observation
        return self.get_observation(), 0, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        sequence_str = "[" + ", ".join(str(x) for x in self.sequence) + "]"
        player_total = self.player_sums[self.current_player]
        opponent_total = self.player_sums[-self.current_player]
        render_str = f"Current Sequence: {sequence_str}\n"
        render_str += f"Player {self.current_player} Total: {player_total}\n"
        render_str += f"Player {-self.current_player} Total: {opponent_total}\n"
        return render_str

    def valid_moves(self):
        if self.sequence:
            return [0, 1]  # Both left and right actions are valid
        else:
            return []  # No valid moves if sequence is empty
