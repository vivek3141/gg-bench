import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (pick leftmost token), 1 (pick rightmost token)
        self.action_space = spaces.Discrete(2)

        # Define observation space
        # Observation consists of:
        # - token sequence: 15 positions, values 0 (no token) to 3 (1: Red, 2: Blue, 3: Green)
        # - counts of collected tokens for both players: 6 counts (Red, Blue, Green for each player)
        # Total length: 15 + 6 = 21
        self.observation_space = spaces.Box(low=0, high=5, shape=(21,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the token sequence
        # 1: Red, 2: Blue, 3: Green
        self.token_sequence = np.array([1, 2, 3] * 5, dtype=np.int32)

        # Positions start from 0 to n-1
        self.left_index = 0
        self.right_index = len(self.token_sequence) - 1

        # Initialize the tokens collected by each player
        # Player 1: index 0, Player 2: index 1
        self.collected_tokens = np.zeros(
            (2, 3), dtype=np.int32
        )  # rows: players, cols: Red, Blue, Green

        # Current player: 0 for Player 1, 1 for Player 2
        self.current_player = 0

        self.done = False

        # Prepare the observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid move after game is over

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid action

        # Take the action
        if action == 0:
            # Pick leftmost token
            token = self.token_sequence[self.left_index]
            self.left_index += 1
        elif action == 1:
            # Pick rightmost token
            token = self.token_sequence[self.right_index]
            self.right_index -= 1
        else:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid action

        # Add token to current player's collection
        player = self.current_player
        if token == 1:
            self.collected_tokens[player][0] += 1  # Red
        elif token == 2:
            self.collected_tokens[player][1] += 1  # Blue
        elif token == 3:
            self.collected_tokens[player][2] += 1  # Green

        # Check if game is over
        if self.left_index > self.right_index:
            self.done = True

            # Determine the winner
            reward = 0
            winner = self._determine_winner()
            if winner == player:
                reward = 1
            else:
                reward = -1  # Losing player gets negative reward
            return self._get_observation(), reward, True, False, {}

        # Switch to the other player
        self.current_player = 1 - self.current_player

        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Visual representation of the game state
        # Remaining token sequence
        remaining_tokens = self.token_sequence[self.left_index : self.right_index + 1]
        token_str = ""
        for idx, token in enumerate(remaining_tokens):
            pos = self.left_index + idx + 1  # Positions numbered from 1
            color = self._token_to_color(token)
            token_str += f"Position {pos}: {color}\n"

        # Player collections
        collections_str = ""
        for player in [0, 1]:
            collections_str += f"Player {player+1}'s Collection:\n"
            collections_str += f"  Red Tokens: {self.collected_tokens[player][0]}\n"
            collections_str += f"  Blue Tokens: {self.collected_tokens[player][1]}\n"
            collections_str += f"  Green Tokens: {self.collected_tokens[player][2]}\n"

        return token_str + collections_str

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1]

    def _get_observation(self):
        # Observation consists of:
        # - token sequence: positions occupied by tokens (0 if no token)
        # - counts of collected tokens for current player
        # - counts of collected tokens for opponent
        # We'll represent the token sequence with zeros outside the current left and right indices
        sequence = np.zeros(15, dtype=np.int32)
        sequence[self.left_index : self.right_index + 1] = self.token_sequence[
            self.left_index : self.right_index + 1
        ]

        # Collected tokens
        player_tokens = self.collected_tokens[self.current_player]
        opponent_tokens = self.collected_tokens[1 - self.current_player]

        observation = np.concatenate([sequence, player_tokens, opponent_tokens])
        return observation

    def _determine_winner(self):
        # Returns 0 if current player wins, 1 if opponent wins, None if tie
        player_counts = self.collected_tokens[self.current_player]
        opponent_counts = self.collected_tokens[1 - self.current_player]

        # Check for color majority
        for color in range(3):
            if player_counts[color] >= 3 and opponent_counts[color] < 3:
                return self.current_player
            if opponent_counts[color] >= 3 and player_counts[color] < 3:
                return 1 - self.current_player

        # No color majority, check total tokens
        player_total = player_counts.sum()
        opponent_total = opponent_counts.sum()
        if player_total > opponent_total:
            return self.current_player
        elif opponent_total > player_total:
            return 1 - self.current_player
        else:
            # Tie-breaker: highest number of any single color
            for color in range(3):
                if player_counts[color] > opponent_counts[color]:
                    return self.current_player
                elif opponent_counts[color] > player_counts[color]:
                    return 1 - self.current_player

            # Should not reach here, as tie is impossible with odd number of tokens
            return None

    def _token_to_color(self, token):
        if token == 1:
            return "Red"
        elif token == 2:
            return "Blue"
        elif token == 3:
            return "Green"
        else:
            return "None"
