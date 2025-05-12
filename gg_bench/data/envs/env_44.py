import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(10), actions 0 to 9 representing block values 1 to 10
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # - First 20 elements: Tower state (block values from base to top), zero-padded
        # - Next 10 elements: Current player's hand (1 if block is available, 0 if not)
        # - Next 10 elements: Opponent's hand (1 if block is available, 0 if not)
        # Total length: 40
        self.observation_space = spaces.Box(low=0, high=10, shape=(40,), dtype=np.int32)

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the tower as an empty list
        self.tower = []

        # Initialize each player's hand with blocks 1-10 (available blocks marked as 1)
        self.player_hands = {
            1: np.ones(10, dtype=np.int32),  # Player 1's blocks
            -1: np.ones(10, dtype=np.int32),  # Player -1's blocks
        }

        # Set the current player (1 or -1)
        self.current_player = 1

        # Game termination flag
        self.terminated = False

        return self._get_observation(), {}  # Return observation and info

    def _get_observation(self):
        # First 20 elements: Tower state, padded with zeros if necessary
        tower_array = np.zeros(20, dtype=np.int32)
        tower_height = min(len(self.tower), 20)
        tower_array[:tower_height] = self.tower[:20]

        # Next 10 elements: Current player's hand
        current_hand = self.player_hands[self.current_player]

        # Next 10 elements: Opponent's hand
        opponent_hand = self.player_hands[-self.current_player]

        # Combine into a single observation array
        observation = np.concatenate([tower_array, current_hand, opponent_hand])

        return observation

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, {}

        # Map action index to block value (0 -> 1, ..., 9 -> 10)
        block_value = action + 1

        # Check if the block is available in the current player's hand
        if self.player_hands[self.current_player][action] == 0:
            # Invalid move: Block not available
            self.terminated = True
            return self._get_observation(), -10, True, False, {}

        # Apply the balance rule to check if the block can be placed
        if len(self.tower) == 0:
            # First block can be any block
            valid_move = True
        elif len(self.tower) == 1:
            # Only one block beneath
            top_block = self.tower[-1]
            valid_move = block_value <= top_block
        else:
            # At least two blocks beneath
            top_block = self.tower[-1]
            second_block = self.tower[-2]
            valid_move = block_value <= (top_block + second_block)

        if not valid_move:
            # Invalid move: Balance rule violated
            self.terminated = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: Update the tower and player's hand
        self.tower.append(block_value)
        self.player_hands[self.current_player][action] = 0  # Remove block from hand

        # Check for winning condition: Current player has placed all their blocks
        if np.sum(self.player_hands[self.current_player]) == 0:
            self.terminated = True
            return self._get_observation(), 1, True, False, {}

        # Check if the opponent has any valid moves
        opponent_valid_moves = self._get_valid_moves(-self.current_player)
        if len(opponent_valid_moves) == 0:
            # Opponent cannot make a valid move: Current player wins
            self.terminated = True
            return self._get_observation(), 1, True, False, {}

        # Switch the current player
        self.current_player *= -1

        return self._get_observation(), 0, False, False, {}

    def _get_valid_moves(self, player):
        valid_moves = []
        player_hand = self.player_hands[player]
        available_actions = np.where(player_hand == 1)[0]  # Indices of available blocks

        for action in available_actions:
            block_value = action + 1  # Map action to block value
            if len(self.tower) == 0:
                # First move: Any block is valid
                valid_moves.append(action)
            elif len(self.tower) == 1:
                top_block = self.tower[-1]
                if block_value <= top_block:
                    valid_moves.append(action)
            else:
                top_block = self.tower[-1]
                second_block = self.tower[-2]
                if block_value <= (top_block + second_block):
                    valid_moves.append(action)
        return valid_moves

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        return self._get_valid_moves(self.current_player)

    def render(self):
        # Generate a visual representation of the game state
        tower_display = "Tower (Top to Bottom): " + str(self.tower[::-1])
        current_player_hand = (
            np.where(self.player_hands[self.current_player] == 1)[0] + 1
        )
        opponent_player_hand = (
            np.where(self.player_hands[-self.current_player] == 1)[0] + 1
        )
        hand_display = (
            f"Current Player ({self.current_player}) Hand: {list(current_player_hand)}\n"
            f"Opponent Player ({-self.current_player}) Hand: {list(opponent_player_hand)}\n"
        )
        game_state = tower_display + "\n" + hand_display
        return game_state
