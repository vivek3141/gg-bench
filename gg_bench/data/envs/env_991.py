import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Discrete actions corresponding to number tokens
        self.number_tokens = [1, 2, 3, 4, 6, 7, 8, 9]
        self.action_space = spaces.Discrete(len(self.number_tokens))

        # Define observation space
        # Observation includes:
        # - Running total (0 to 25)
        # - Player 1's hand (8 numbers, 0 or 1)
        # - Player 2's hand (8 numbers, 0 or 1)
        low = np.zeros(17, dtype=np.int32)
        high = np.ones(17, dtype=np.int32)
        high[0] = 25  # Running total can be from 0 to 25
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the running total
        self.running_total = 0

        # Initialize players' hands: All numbers are available at the start
        self.player_hands = {
            1: np.ones(len(self.number_tokens), dtype=np.int32),
            2: np.ones(len(self.number_tokens), dtype=np.int32),
        }

        # Randomly select who goes first
        self.current_player = self.np_random.choice([1, 2])

        # Game state flags
        self.done = False
        self.truncated = False

        # Build the initial observation
        observation = self._get_obs()

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no further moves can be made
            observation = self._get_obs()
            return observation, 0, True, False, {}

        # Selected number based on action
        selected_number = self.number_tokens[action]

        # Check if the selected number is available in current player's hand
        if self.player_hands[self.current_player][action] == 0:
            # Invalid move: number not available
            self.done = True
            reward = -10
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Calculate the new running total
        new_total = self.running_total + selected_number

        # Check if the move results in a forbidden running total (multiple of 5 excluding 25)
        if new_total % 5 == 0 and new_total != 25:
            # Invalid move: running total is a forbidden multiple of 5
            self.done = True
            reward = -10  # Penalize invalid move
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Update the game state
        self.running_total = new_total
        self.player_hands[self.current_player][action] = 0  # Mark the number as used

        # Check for win condition
        if self.running_total == 25:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Check if the current player has any valid moves left
        if not self._has_valid_moves(self.current_player):
            # Current player loses because they have no valid moves
            self.done = True
            reward = -10  # Penalize losing
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player has any valid moves
        if not self._has_valid_moves(self.current_player):
            # Next player loses because they have no valid moves
            self.done = True
            # Reward the current player as the opponent cannot move
            reward = 1 if self.current_player != self.current_player else -10
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Continue the game
        reward = 0  # No reward for a valid move
        observation = self._get_obs()
        return observation, reward, False, False, {}

    def render(self):
        player1_hand = self._hand_to_string(self.player_hands[1])
        player2_hand = self._hand_to_string(self.player_hands[2])
        output = (
            f"Running Total: {self.running_total}\n"
            f"Player 1 Hand: {player1_hand}\n"
            f"Player 2 Hand: {player2_hand}\n"
            f"Current Player: Player {self.current_player}"
        )
        return output

    def valid_moves(self):
        valid_actions = []
        current_hand = self.player_hands[self.current_player]
        for idx, available in enumerate(current_hand):
            if available == 1:
                selected_number = self.number_tokens[idx]
                new_total = self.running_total + selected_number
                if new_total % 5 != 0 or new_total == 25:
                    valid_actions.append(idx)
        return valid_actions

    def _get_obs(self):
        # Observation includes running total and both players' hands
        observation = np.zeros(17, dtype=np.int32)
        observation[0] = self.running_total
        observation[1:9] = self.player_hands[1]
        observation[9:17] = self.player_hands[2]
        return observation

    def _has_valid_moves(self, player):
        # Check if the player has any valid moves
        current_hand = self.player_hands[player]
        for idx, available in enumerate(current_hand):
            if available == 1:
                selected_number = self.number_tokens[idx]
                new_total = self.running_total + selected_number
                if new_total % 5 != 0 or new_total == 25:
                    return True
        return False

    def _hand_to_string(self, hand):
        # Convert the player's hand to a string representation
        available_numbers = [
            str(self.number_tokens[idx])
            for idx, available in enumerate(hand)
            if available == 1
        ]
        return ", ".join(available_numbers)
