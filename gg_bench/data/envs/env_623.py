import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 125 possible guesses (numbers from 1 to 5 for each digit in a 3-digit code)
        self.action_space = spaces.Discrete(125)

        # Observation space: 201 integers
        # - First element: current player (1 or 2)
        # - Next 200 elements: game history (up to 20 turns, each turn has 10 elements)
        #   - For each turn:
        #     - Player 1's guess (3 digits)
        #     - Player 1's feedback (CDCP, CDIP)
        #     - Player 2's guess (3 digits)
        #     - Player 2's feedback (CDCP, CDIP)
        # Unused entries are filled with -1
        self.observation_space = spaces.Box(
            low=-1, high=5, shape=(201,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.history = []
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Generate secret codes for both players (digits 1 to 5)
        self.np_random = np.random.RandomState()
        self.player1_code = self._generate_secret_code()
        self.player2_code = self._generate_secret_code()

        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            observation = self._get_observation()
            return observation, -10, True, False, {}

        # Map action to code guess
        guess = self._action_to_code(action)

        # Process guess for current player
        if self.current_player == 1:
            secret_code = self.player2_code
        else:
            secret_code = self.player1_code

        # Compute feedback
        cdcp, cdip = self._compute_feedback(guess, secret_code)

        # Update history
        turn_record = {
            "player": self.current_player,
            "guess": guess,
            "cdcp": cdcp,
            "cdip": cdip,
        }
        self.history.append(turn_record)

        # Check for win
        if cdcp == 3:
            self.done = True
            reward = 1
        else:
            reward = 0

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Get observation
        observation = self._get_observation()

        return observation, reward, self.done, False, {}

    def render(self):
        history_str = ""
        for idx, turn in enumerate(self.history):
            player = turn["player"]
            guess = turn["guess"]
            cdcp = turn["cdcp"]
            cdip = turn["cdip"]
            history_str += f"Turn {idx + 1}, Player {player}:\n"
            history_str += f"  Guess: {guess}\n"
            history_str += f"  Feedback - CDCP: {cdcp}, CDIP: {cdip}\n"
        return history_str

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(125))

    def _generate_secret_code(self):
        return tuple(self.np_random.randint(1, 6, size=3))

    def _action_to_code(self, action):
        d1 = action // 25
        remainder = action % 25
        d2 = remainder // 5
        d3 = remainder % 5
        return (d1 + 1, d2 + 1, d3 + 1)  # Digits are from 1 to 5

    def _compute_feedback(self, guess, code):
        # Correct Digit Correct Position (CDCP)
        cdcp = sum(guess[i] == code[i] for i in range(3))

        # Correct Digit Incorrect Position (CDIP)
        guess_counts = Counter(guess)
        code_counts = Counter(code)
        total_matches = sum(min(guess_counts[d], code_counts[d]) for d in guess_counts)
        cdip = total_matches - cdcp

        return cdcp, cdip

    def _get_observation(self):
        # Prepare observation array
        observation = np.full(201, -1, dtype=np.int32)

        # Set current player indicator
        observation[0] = self.current_player

        # Fill in the history
        idx = 1  # Start from index 1
        for turn_idx in range(0, len(self.history), 2):
            # Each full turn consists of Player 1 and Player 2 actions
            if turn_idx < len(self.history):
                # Player 1's turn
                turn_p1 = self.history[turn_idx]
                observation[idx : idx + 3] = turn_p1["guess"]
                observation[idx + 3] = turn_p1["cdcp"]
                observation[idx + 4] = turn_p1["cdip"]
            else:
                break

            idx += 5  # Move to next player

            if turn_idx + 1 < len(self.history):
                # Player 2's turn
                turn_p2 = self.history[turn_idx + 1]
                observation[idx : idx + 3] = turn_p2["guess"]
                observation[idx + 3] = turn_p2["cdcp"]
                observation[idx + 4] = turn_p2["cdip"]
            else:
                break

            idx += 5  # Move to next turn

        return observation
