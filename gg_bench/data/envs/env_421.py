import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 actions:
        # Actions 0-3: Flip own bit at positions 0 to 3
        # Actions 4-7: Reset opponent's bit at positions 0 to 3
        # Action 8: Pass
        self.action_space = spaces.Discrete(9)
        # Observation space: 8 bits, first 4 are current player's bits, next 4 are opponent's bits
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.zeros((2, 4), dtype=np.int8)  # bits[player][bit_position]
        self.current_player = 0  # Player 0 starts
        self.done = False
        self.last_player = None  # To track who made the last move
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game already finished
            observation = self._get_observation()
            return observation, -10, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, {}

        # Apply the action
        if action == 8:
            # Pass action
            pass
        elif action >= 0 and action <= 3:
            # Flip own bit at position action
            if self.bits[self.current_player][action] == 0:
                self.bits[self.current_player][action] = 1
            else:
                # Invalid move, can't flip bit that's already 1
                self.done = True
                observation = self._get_observation()
                return observation, -10, True, False, {}
        elif action >= 4 and action <= 7:
            # Reset opponent's bit at position action - 4
            opponent = 1 - self.current_player
            bit_pos = action - 4
            if self.bits[opponent][bit_pos] == 1:
                self.bits[opponent][bit_pos] = 0
            else:
                # Invalid move, can't reset bit that's already 0
                self.done = True
                observation = self._get_observation()
                return observation, -10, True, False, {}
        else:
            # Invalid action
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, {}

        self.last_player = self.current_player

        # Check win condition
        if np.all(self.bits[self.current_player] == 1):
            # Current player wins
            self.done = True
            observation = self._get_observation()
            return observation, 1, True, False, {}

        # Switch to next player
        self.current_player = 1 - self.current_player

        # Check if next player has any valid moves
        if not self._has_valid_moves(self.current_player):
            # Next player has no valid moves, switch back
            self.current_player = 1 - self.current_player
            if not self._has_valid_moves(self.current_player):
                # Both players have no valid moves, game ends
                self.done = True
                # Determine winner by number of bits set to 1
                player_ones = np.sum(self.bits[self.current_player])
                opponent_ones = np.sum(self.bits[1 - self.current_player])
                if player_ones > opponent_ones:
                    # Current player wins
                    observation = self._get_observation()
                    return observation, 1, True, False, {}
                elif player_ones < opponent_ones:
                    # Opponent wins, but since we are always returning from current player's perspective
                    observation = self._get_observation()
                    return observation, -1, True, False, {}
                else:
                    # Tie, player who made the last move wins
                    if self.last_player == self.current_player:
                        # Current player wins
                        observation = self._get_observation()
                        return observation, 1, True, False, {}
                    else:
                        # Opponent wins
                        observation = self._get_observation()
                        return observation, -1, True, False, {}

        # Game continues
        observation = self._get_observation()
        return observation, 0, False, False, {}

    def render(self):
        # Return a visual representation of the game state
        lines = []
        lines.append(f"Current Player: Player {self.current_player + 1}")
        lines.append(f"Player 1 bits: {''.join(map(str, self.bits[0]))}")
        lines.append(f"Player 2 bits: {''.join(map(str, self.bits[1]))}")
        return "\n".join(lines)

    def valid_moves(self):
        # Return a list of valid actions
        valid_actions = []
        own_bits = self.bits[self.current_player]
        opponent_bits = self.bits[1 - self.current_player]

        # Actions 0-3: Flip own bit at positions 0 to 3
        for i in range(4):
            if own_bits[i] == 0:
                valid_actions.append(i)

        # Actions 4-7: Reset opponent's bit at positions 0 to 3
        for i in range(4):
            if opponent_bits[i] == 1:
                valid_actions.append(i + 4)

        # If no other valid moves, include 'pass' action
        if len(valid_actions) == 0:
            valid_actions.append(8)  # Pass action

        return valid_actions

    def _get_observation(self):
        # Observation is current player's bits followed by opponent's bits
        own_bits = self.bits[self.current_player]
        opponent_bits = self.bits[1 - self.current_player]
        observation = np.concatenate([own_bits, opponent_bits])
        return observation

    def _has_valid_moves(self, player):
        own_bits = self.bits[player]
        opponent_bits = self.bits[1 - player]
        can_flip_own = np.any(own_bits == 0)
        can_reset_opponent = np.any(opponent_bits == 1)
        return can_flip_own or can_reset_opponent
