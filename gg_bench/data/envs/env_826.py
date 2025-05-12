import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space (45 possible actions: 15 tokens * 3 stacks)
        self.action_space = spaces.Discrete(45)

        # Observation space: 15 (pool tokens) + 3 (current player's stacks) +
        # 3 (opponent's stacks) + 1 (current player indicator) = 22 elements
        # pool_tokens: 0 or 1
        # stack totals: 0 to 15
        # current player indicator: 1 or -1
        low = np.array([0] * 15 + [0] * 6 + [-1], dtype=np.float32)
        high = np.array([1] * 15 + [15] * 6 + [1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(22,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Tokens in the pool: 1 indicates available, 0 indicates taken
        self.pool_tokens = np.ones(15, dtype=np.int8)

        # Players' stacks: for each player, have an array of stack totals
        self.player_stack_totals = np.zeros(
            (2, 3), dtype=np.int8
        )  # 2 players, 3 stacks each

        # Current player: 0 for Player A, 1 for Player B
        self.current_player = 0

        # Game over flag
        self.done = False

        # Return the initial observation and empty info
        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        # Concatenate pool_tokens, current player's stacks, opponent's stacks, current player indicator
        observation = np.concatenate(
            [
                self.pool_tokens.astype(np.float32),
                self.player_stack_totals[self.current_player].astype(np.float32),
                self.player_stack_totals[1 - self.current_player].astype(np.float32),
                np.array([1.0], dtype=np.float32),  # Current player indicator
            ]
        )
        return observation

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Parse action into token_index and stack_index
        token_index = action // 3  # Token index from 0 to 14
        stack_index = action % 3  # Stack index from 0 to 2

        token_value = token_index + 1  # Tokens are numbered from 1 to 15

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10.0
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Perform the move
        self.pool_tokens[token_index] = 0  # Remove token from pool
        self.player_stack_totals[self.current_player][
            stack_index
        ] += token_value  # Update stack total

        # Check for win condition
        if self.player_stack_totals[self.current_player][stack_index] == 15:
            # Current player wins
            self.done = True
            reward = 1.0
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Check if game should end (no tokens left or no legal moves for both players)
        if np.sum(self.pool_tokens) == 0:
            # Determine winner by highest stack sums
            reward = self._determine_winner()
            self.done = True
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Check if both players have no legal moves
        current_player_moves = self.valid_moves()
        # Switch to opponent to check their moves
        self.current_player = 1 - self.current_player
        opponent_moves = self.valid_moves()
        self.current_player = 1 - self.current_player  # Switch back

        if len(current_player_moves) == 0 and len(opponent_moves) == 0:
            # Game ends, determine winner
            reward = self._determine_winner()
            self.done = True
            observation = self._get_obs()
            return observation, reward, True, False, {}

        # Switch current player
        self.current_player = 1 - self.current_player

        # Return observation and default reward
        observation = self._get_obs()
        return observation, 0.0, False, False, {}

    def _determine_winner(self):
        # Determine the winner based on highest stack sums not exceeding 15
        player_totals = []
        for player in range(2):
            valid_stacks = self.player_stack_totals[player][
                self.player_stack_totals[player] <= 15
            ]
            sorted_totals = np.sort(valid_stacks)[::-1]  # Sort in descending order
            player_totals.append(sorted_totals)

        # Compare the highest stacks
        for i in range(3):
            player_stack = (
                player_totals[self.current_player][i]
                if i < len(player_totals[self.current_player])
                else 0
            )
            opponent_stack = (
                player_totals[1 - self.current_player][i]
                if i < len(player_totals[1 - self.current_player])
                else 0
            )
            if player_stack > opponent_stack:
                # Current player wins
                return 1.0
            elif opponent_stack > player_stack:
                # Current player loses
                return -1.0  # Negative reward for losing
            else:
                continue  # Stacks are equal, compare next stack

        # If all stacks are equal, Player B wins
        if self.current_player == 1:
            # Current player (Player B) wins
            return 1.0
        else:
            # Current player (Player A) loses
            return -1.0

    def render(self):
        # Return a string representation of the game state
        player_labels = ["A", "B"]
        lines = []
        lines.append(
            "Tokens in Pool: "
            + str([i + 1 for i in range(15) if self.pool_tokens[i] == 1])
        )
        lines.append(f"Player {player_labels[self.current_player]}'s turn.")
        for player in range(2):
            lines.append(f"Player {player_labels[player]}'s stacks:")
            for stack_index in range(3):
                stack_total = self.player_stack_totals[player][stack_index]
                lines.append(f"  Stack {stack_index+1} Total: {stack_total}")
        return "\n".join(lines)

    def valid_moves(self):
        valid_actions = []
        for token_index in range(15):
            if self.pool_tokens[token_index] == 1:
                token_value = token_index + 1  # Tokens are numbered from 1 to 15
                for stack_index in range(3):
                    if (
                        self.player_stack_totals[self.current_player][stack_index]
                        + token_value
                        <= 15
                    ):
                        action = 3 * token_index + stack_index
                        valid_actions.append(action)
        return valid_actions
