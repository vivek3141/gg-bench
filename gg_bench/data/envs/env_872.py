import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Gymnasium Environment for the Reverse Avoidance game.

    The environment is designed for self-play reinforcement learning.
    The action_space is a Discrete space with 10 actions:
        - Actions 0-8: Append digits '1' to '9' to the sequence
        - Action 9: Reverse the sequence

    The observation_space is a Box space representing the current sequence of digits,
    with padding zeros up to a maximum sequence length.

    The reward is:
        - +1 if the current player wins (opponent has no valid moves)
        - -10 if the current player loses (creates a sequence divisible by 7 or makes an invalid move)
        - 0 otherwise (game continues)
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: Append '1' to '9' (indices 0-8), Reverse sequence (index 9)
        self.action_space = spaces.Discrete(10)

        # Observation space: array of digits, padded with zeros if sequence is shorter than max length
        self.max_sequence_length = 100  # Set max sequence length to 100
        # Valid digits are from 1 to 9, zeros are used for padding
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_sequence_length,), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sequence = np.zeros(self.max_sequence_length, dtype=np.int8)
        self.sequence_length = 1
        self.current_sequence[0] = 1  # initial sequence is '1'
        self.current_player = 1  # player 1 starts
        self.done = False
        info = {}
        observation = self.current_sequence.copy()
        return observation, info

    def step(self, action):
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action, current player loses
            reward = -10
            terminated = True
            truncated = False
            info = {}
            observation = self.current_sequence.copy()
            return observation, reward, terminated, truncated, info

        # Apply the action
        if action >= 0 and action <= 8:
            # Append digit '1' to '9' (actions 0-8)
            if self.sequence_length >= self.max_sequence_length:
                # Cannot append, sequence is at max length; invalid action
                reward = -10
                terminated = True
                truncated = False
                info = {}
                observation = self.current_sequence.copy()
                return observation, reward, terminated, truncated, info
            else:
                digit_to_append = action + 1  # Map action 0 to digit '1', etc.
                self.current_sequence[self.sequence_length] = digit_to_append
                self.sequence_length += 1
        elif action == 9:
            # Reverse the sequence
            sequence = self.current_sequence[: self.sequence_length]
            reversed_sequence = sequence[::-1]
            self.current_sequence[: self.sequence_length] = reversed_sequence
        else:
            pass  # Should not happen

        # Now check if sequence is divisible by 7
        sequence_number = int(
            "".join(map(str, self.current_sequence[: self.sequence_length]))
        )
        if sequence_number % 7 == 0:
            # Current player loses
            reward = -10
            terminated = True
            truncated = False
            info = {}
            observation = self.current_sequence.copy()
            return observation, reward, terminated, truncated, info

        # Switch to next player
        self.current_player = 3 - self.current_player  # Switch players

        # Check if next player has any valid moves
        valid_moves_next_player = self.valid_moves()
        if not valid_moves_next_player:
            # Next player has no valid moves, current player wins
            # Switch back to current player for consistency
            self.current_player = 3 - self.current_player
            reward = 1
            terminated = True
        else:
            # Game continues
            reward = 0
            terminated = False

        truncated = False
        info = {}
        observation = self.current_sequence.copy()
        return observation, reward, terminated, truncated, info

    def render(self):
        # Return a string representation of the state
        sequence_str = "".join(map(str, self.current_sequence[: self.sequence_length]))
        return f"Current sequence: {sequence_str}"

    def valid_moves(self):
        # Return a list of valid moves (action indices) in the current state
        valid_moves = []

        # Check for reverse action
        sequence = self.current_sequence[: self.sequence_length]
        reversed_sequence = sequence[::-1]
        reversed_number = int("".join(map(str, reversed_sequence)))
        if reversed_number % 7 != 0:
            valid_moves.append(9)  # Action index for reversing is 9

        # Check for appending digits '1' to '9' (actions 0-8)
        if self.sequence_length < self.max_sequence_length:
            for action in range(
                9
            ):  # Actions 0 to 8 represent appending digits '1' to '9'
                digit = action + 1  # Digits '1' to '9'
                new_sequence = np.copy(self.current_sequence[: self.sequence_length])
                new_sequence = np.append(new_sequence, digit)
                new_number = int("".join(map(str, new_sequence)))
                if new_number % 7 != 0:
                    valid_moves.append(action)
        return valid_moves
