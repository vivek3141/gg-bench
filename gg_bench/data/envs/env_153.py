import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum word length
        self.MAX_WORD_LEN = 15  # Adjust as needed

        # Define action and observation space
        self.action_space = spaces.Discrete(self.MAX_WORD_LEN * 2)
        self.observation_space = spaces.Box(
            low=0, high=26, shape=(self.MAX_WORD_LEN,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting word (e.g., 'BALANCE')
        self.starting_word = "BALANCE"
        self.word_length = len(self.starting_word)

        # Map letters to integers: 'A'->1, 'B'->2, ..., 'Z'->26
        self.letter_to_int = {chr(i + 64): i for i in range(1, 27)}
        self.int_to_letter = {i: chr(i + 64) for i in range(1, 27)}

        # Initialize the state
        self.state = np.zeros(self.MAX_WORD_LEN, dtype=np.int32)
        for i, letter in enumerate(self.starting_word.upper()):
            self.state[i] = self.letter_to_int.get(letter, 0)

        self.current_player = 1  # Player 1 starts
        self.done = False
        self.reward = 0

        return self.state.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.state.copy(), 0, True, False, {}

        # Map action to position and remove_length
        position = action // 2
        remove_length = (action % 2) + 1

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.state.copy(), -10, True, False, {}

        # Remove letters
        available_positions = np.where(self.state > 0)[0]
        if position not in available_positions:
            self.done = True
            return self.state.copy(), -10, True, False, {}

        # Check if the letters to remove exist
        if position + remove_length > self.MAX_WORD_LEN:
            self.done = True
            return self.state.copy(), -10, True, False, {}

        # Check if the positions to remove have letters
        indices_to_remove = []
        for i in range(remove_length):
            idx = position + i
            if idx in available_positions:
                indices_to_remove.append(idx)
            else:
                self.done = True
                return self.state.copy(), -10, True, False, {}

        # Remove the letters
        for idx in indices_to_remove:
            self.state = np.delete(self.state, idx)
            self.state = np.append(self.state, 0)
            available_positions = np.where(self.state > 0)[0]

        # Check if the game is over
        if np.all(self.state == 0):
            self.done = True
            return self.state.copy(), 1, True, False, {}

        # No reward for a valid move that doesn't end the game
        reward = 0

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.state.copy(), reward, False, False, {}

    def render(self):
        # Visual representation of the current word
        letters = [
            self.int_to_letter.get(num, " ") if num > 0 else " " for num in self.state
        ]
        word_str = "".join(letters).strip()
        return f"Current word: {word_str}"

    def valid_moves(self):
        valid_actions = []
        available_positions = np.where(self.state > 0)[0]

        for position in available_positions:
            # Remove one letter
            action_id = position * 2
            valid_actions.append(action_id)

            # Remove two letters if possible
            if position + 1 in available_positions:
                action_id = position * 2 + 1
                valid_actions.append(action_id)

        return valid_actions
