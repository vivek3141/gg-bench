import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, digit_sum_limit=15):
        super(CustomEnv, self).__init__()

        self.digit_sum_limit = digit_sum_limit

        # Define action and observation space
        # Actions correspond to digits 1-9 (indices 0-8)
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # For each player:
        # - Digit sum (scalar)
        # - Used digits (array of length 9, entries 0 or 1)
        # - Digit history (array of length 9, entries 0 or digits 1-9)
        # Current player indicator (0 or 1)
        self.observation_space = spaces.Box(
            low=0,
            high=45,  # Maximum possible digit sum is 45 (sum of digits 1-9)
            shape=(2 * (1 + 9 + 9) + 1,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0  # Player 0 starts
        self.done = False

        # Initialize players' states
        self.player_states = [
            {
                "digit_sum": 0,
                "used_digits": np.zeros(9, dtype=np.int32),  # Digits 1-9
                "digit_history": np.zeros(
                    9, dtype=np.int32
                ),  # Order of digits selected
                "number": 0,  # Constructed number
                "turns": 0,  # Number of turns taken
            },
            {
                "digit_sum": 0,
                "used_digits": np.zeros(9, dtype=np.int32),
                "digit_history": np.zeros(9, dtype=np.int32),
                "number": 0,
                "turns": 0,
            },
        ]

        return self._get_obs(), {}

    def step(self, action):
        player = self.current_player
        opponent = 1 - self.current_player
        info = {}

        if self.done:
            return self._get_obs(), 0, True, False, info

        # Map action to digit (1-9)
        digit = action + 1

        # Check if digit is valid (not used and does not exceed digit sum limit)
        valid_moves = self.valid_moves()
        if action not in valid_moves:
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, info

        # Update player's state
        self.player_states[player]["used_digits"][action] = 1
        idx = np.where(self.player_states[player]["digit_history"] == 0)[0][0]
        self.player_states[player]["digit_history"][idx] = digit
        self.player_states[player]["digit_sum"] += digit
        self.player_states[player]["turns"] += 1

        # Append digit to the number
        self.player_states[player]["number"] = int(
            f"{self.player_states[player]['number']}{digit}"
        )

        # Check if digit sum exceeds the limit
        if self.player_states[player]["digit_sum"] > self.digit_sum_limit:
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, info

        # Check if both players cannot make a move
        self.current_player = opponent  # Switch player
        if not self.valid_moves():
            self.current_player = player  # Switch back
            if not self.valid_moves():
                # Game ends, determine winner
                self.done = True
                player_number = self.player_states[player]["number"]
                opponent_number = self.player_states[opponent]["number"]

                if player_number > opponent_number:
                    reward = 1
                elif player_number < opponent_number:
                    reward = -1
                else:
                    # Tie-breaker: fewer turns wins
                    if (
                        self.player_states[player]["turns"]
                        < self.player_states[opponent]["turns"]
                    ):
                        reward = 1
                    elif (
                        self.player_states[player]["turns"]
                        > self.player_states[opponent]["turns"]
                    ):
                        reward = -1
                    else:
                        # It's a draw
                        reward = 0
                return self._get_obs(), reward, True, False, info

        # Continue the game
        self.current_player = opponent  # Switch to opponent's turn
        reward = 0
        return self._get_obs(), reward, False, False, info

    def render(self):
        player_0 = self.player_states[0]
        player_1 = self.player_states[1]
        output = "Current Game State:\n"
        output += f"Digit Sum Limit: {self.digit_sum_limit}\n\n"
        output += f"Player 1's Number: {player_0['number']}\n"
        output += f"Player 1's Digit Sum: {player_0['digit_sum']}\n"
        output += (
            f"Player 1's Used Digits: {np.where(player_0['used_digits']==1)[0]+1}\n\n"
        )
        output += f"Player 2's Number: {player_1['number']}\n"
        output += f"Player 2's Digit Sum: {player_1['digit_sum']}\n"
        output += (
            f"Player 2's Used Digits: {np.where(player_1['used_digits']==1)[0]+1}\n\n"
        )
        output += f"It's Player {self.current_player + 1}'s turn.\n"
        return output

    def valid_moves(self):
        player = self.current_player
        available_digits = np.where(self.player_states[player]["used_digits"] == 0)[0]
        valid_moves = []
        for idx in available_digits:
            digit = idx + 1
            if self.player_states[player]["digit_sum"] + digit <= self.digit_sum_limit:
                valid_moves.append(idx)
        return valid_moves

    def _get_obs(self):
        obs = []

        for player_state in self.player_states:
            # Digit sum (scalar)
            obs.append(player_state["digit_sum"])
            # Used digits (array of length 9)
            obs.extend(player_state["used_digits"])
            # Digit history (array of length 9)
            obs.extend(player_state["digit_history"])

        # Current player indicator
        obs.append(self.current_player)

        return np.array(obs, dtype=np.int32)
