import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 9 digits * 3 positions = 27 possible actions
        self.action_space = spaces.Discrete(27)

        # Define observation space:
        # - Player's own number slots (3 positions)
        # - Opponent's number slots (3 positions)
        # - Available digits (9 digits)
        # Total size = 3 + 3 + 9 = 15
        self.observation_space = spaces.Box(low=0, high=9, shape=(15,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.available_digits = list(range(1, 10))  # Digits 1-9
        self.player_slots = [0, 0, 0]  # Current player's slots (Hundreds, Tens, Units)
        self.opponent_slots = [0, 0, 0]  # Opponent's slots
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        self.total_turns = 0  # Total number of turns taken

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Map action to digit and position
        digit_index = action // 3
        position_index = action % 3
        digit = digit_index + 1  # Digits are 1-9
        position = position_index  # Positions: 0-Hundreds, 1-Tens, 2-Units

        # Check for invalid move: digit not available
        if digit not in self.available_digits:
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Get the current player's slots
        if self.current_player == 1:
            player_slots = self.player_slots
        else:
            player_slots = self.opponent_slots

        # Check for invalid move: position already filled
        if player_slots[position] != 0:
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Valid move: update game state
        self.available_digits.remove(digit)
        player_slots[position] = digit
        self.total_turns += 1

        # Check if the game has ended
        if self.total_turns == 6:
            self.done = True
            # Calculate the final numbers for both players
            player_number = (
                self.player_slots[0] * 100
                + self.player_slots[1] * 10
                + self.player_slots[2]
            )
            opponent_number = (
                self.opponent_slots[0] * 100
                + self.opponent_slots[1] * 10
                + self.opponent_slots[2]
            )

            # Determine the winner and assign reward
            if player_number > opponent_number:
                if self.current_player == 1:
                    reward = 1  # Current player wins
                else:
                    reward = 0  # Current player loses
            else:
                if self.current_player == -1:
                    reward = 1  # Current player wins
                else:
                    reward = 0  # Current player loses
            return self._get_observation(), reward, self.done, False, {}
        else:
            # Continue the game: switch to the other player
            self.current_player *= -1
            reward = 0
            return self._get_observation(), reward, self.done, False, {}

    def _get_observation(self):
        # Create the observation array
        observation = np.zeros(15, dtype=np.int32)
        observation[0:3] = self.player_slots
        observation[3:6] = self.opponent_slots
        # Available digits (1 if available, 0 if not)
        for i in range(9):
            observation[6 + i] = 1 if (i + 1) in self.available_digits else 0
        return observation

    def render(self):
        # Return a string representation of the current game state
        s = "Available Digits: " + " ".join(map(str, self.available_digits)) + "\n"
        s += (
            f"Player's Number:   [ {self._slot_str(self.player_slots[0])} ] "
            f"[ {self._slot_str(self.player_slots[1])} ] [ {self._slot_str(self.player_slots[2])} ]\n"
        )
        s += (
            f"Opponent's Number: [ {self._slot_str(self.opponent_slots[0])} ] "
            f"[ {self._slot_str(self.opponent_slots[1])} ] [ {self._slot_str(self.opponent_slots[2])} ]\n"
        )
        return s

    def _slot_str(self, digit):
        return str(digit) if digit != 0 else "_"

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for digit in self.available_digits:
            digit_index = digit - 1
            for position in range(3):
                # Get the current player's slots
                if self.current_player == 1:
                    player_slots = self.player_slots
                else:
                    player_slots = self.opponent_slots
                if player_slots[position] == 0:
                    action = digit_index * 3 + position
                    valid_actions.append(action)
        return valid_actions
