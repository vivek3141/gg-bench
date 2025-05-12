import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 54 possible actions: 9 digits * 6 slots
        self.action_space = spaces.Discrete(54)

        # Observation space: 6 slots (integers 0-9) and 9 digit availability flags (0 or 1)
        self.observation_space = spaces.Box(low=0, high=9, shape=(15,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 6 slots: indices 0-2 for Player 0, indices 3-5 for Player 1
        self.slots = np.zeros(6, dtype=np.int8)
        # Digit availability: 1 if available, 0 if used
        self.digits_available = np.ones(9, dtype=np.int8)
        # Current player: 0 or 1
        self.current_player = 0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # Observation is the combination of slots and digit availability
        return np.concatenate([self.slots, self.digits_available])

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_obs(), 0, True, False, {}

        # Map action index to digit and slot
        action_digit = action // 6 + 1  # Digit from 1 to 9
        action_slot = action % 6  # Slot from 0 to 5

        # Check if digit is available
        if self.digits_available[action_digit - 1] == 0:
            # Invalid move: digit not available
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check if slot is empty
        if self.slots[action_slot] != 0:
            # Invalid move: slot already filled
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move: place digit in slot and update availability
        self.digits_available[action_digit - 1] = 0
        self.slots[action_slot] = action_digit

        # Check if all slots are filled
        if np.all(self.slots != 0):
            # Game over, calculate results
            self.done = True
            player0_result = self.calculate_equation(self.slots[0:3])
            player1_result = self.calculate_equation(self.slots[3:6])

            # Determine winner
            if player0_result > player1_result:
                winner = 0
            elif player1_result > player0_result:
                winner = 1
            else:
                # Tie-breaker: second player wins
                winner = 1

            # Assign reward
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1
            return self._get_obs(), reward, True, False, {}
        else:
            # Switch to the other player
            self.current_player = 1 - self.current_player
            return self._get_obs(), 0, False, False, {}

    def calculate_equation(self, slots):
        # Calculate the result of __ + __ × __
        A, B, C = slots
        return A + B * C

    def render(self):
        # Visual representation of the game state
        s = ""
        s += f"Player 0's equation: {self.slot_string(self.slots[0:3])}\n"
        s += f"Player 1's equation: {self.slot_string(self.slots[3:6])}\n"
        s += (
            "Available digits: "
            + " ".join(
                str(i + 1)
                for i, avail in enumerate(self.digits_available)
                if avail == 1
            )
            + "\n"
        )
        s += f"Current player: Player {self.current_player}"
        return s

    def slot_string(self, slots):
        # String representation of a player's equation
        s = ""
        s += (str(slots[0]) if slots[0] != 0 else "__") + " + "
        s += (str(slots[1]) if slots[1] != 0 else "__") + " × "
        s += str(slots[2]) if slots[2] != 0 else "__"
        return s

    def valid_moves(self):
        # List of valid moves as action indices
        moves = []
        for digit in range(1, 10):
            if self.digits_available[digit - 1] == 1:
                for slot in range(6):
                    if self.slots[slot] == 0:
                        action = (digit - 1) * 6 + slot
                        moves.append(action)
        return moves
