import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space (0-4: place block sizes 1-5, 5: pass)
        self.action_space = spaces.Discrete(6)

        # Define observation space
        # Observation: [current_player_tower_height, opponent_player_tower_height,
        #               current_player_block_counts[0-4], opponent_player_block_counts[0-4]]
        # Block counts for sizes 1-5
        self.max_block_counts = np.array([5, 4, 3, 2, 1], dtype=np.int32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0] * 5 + [0] * 5, dtype=np.int32),
            high=np.array(
                [20, 20]
                + self.max_block_counts.tolist()
                + self.max_block_counts.tolist(),
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tower_heights = np.array(
            [0, 0], dtype=np.int32
        )  # [player1_height, player2_height]
        self.block_counts = np.array(
            [self.max_block_counts.copy(), self.max_block_counts.copy()], dtype=np.int32
        )
        self.current_player = 0  # Player 0 starts
        self.pass_counter = 0
        self.last_player = None  # No moves made yet
        self.done = False

        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        if action == 5:
            # Pass action
            self.pass_counter += 1
            if self.pass_counter == 2:
                # Both players have passed consecutively
                self.done = True
                winner = self._determine_winner()
                reward = 1 if winner == self.current_player else 0
                return self._get_obs(), reward, True, False, {}
            else:
                # Switch players
                self.current_player = 1 - self.current_player
                return self._get_obs(), 0, False, False, {}
        else:
            # Place a block
            block_size = action + 1  # Action 0-4 corresponds to block sizes 1-5
            player_blocks = self.block_counts[self.current_player]
            if player_blocks[action] <= 0:
                # Block not available
                self.done = True
                return self._get_obs(), -10, True, False, {}
            tower_height = self.tower_heights[self.current_player]
            if tower_height + block_size > 20:
                # Tower would exceed 20 units
                self.done = True
                return self._get_obs(), -10, True, False, {}

            # Valid move, update state
            self.tower_heights[self.current_player] += block_size
            self.block_counts[self.current_player][action] -= 1
            self.pass_counter = 0
            self.last_player = self.current_player

            if self.tower_heights[self.current_player] == 20:
                # Current player wins
                self.done = True
                return self._get_obs(), 1, True, False, {}
            else:
                # Check if opponent has any valid moves
                self.current_player = 1 - self.current_player
                return self._get_obs(), 0, False, False, {}

    def render(self):
        # Return a string representing the game state
        s = f"Player 1 Tower Height: {self.tower_heights[0]} units\n"
        s += f"Player 2 Tower Height: {self.tower_heights[1]} units\n"
        s += f"Player {self.current_player + 1}'s turn.\n"
        s += f"Player 1 Remaining Blocks: {dict(zip(range(1,6), self.block_counts[0]))}\n"
        s += f"Player 2 Remaining Blocks: {dict(zip(range(1,6), self.block_counts[1]))}\n"
        return s

    def valid_moves(self):
        valid_actions = []

        player_blocks = self.block_counts[self.current_player]
        tower_height = self.tower_heights[self.current_player]

        for action in range(5):  # Actions 0-4 correspond to block sizes 1-5
            block_size = action + 1
            if player_blocks[action] > 0 and (tower_height + block_size) <= 20:
                valid_actions.append(action)

        if not valid_actions:
            # No valid moves, player must pass
            valid_actions.append(5)  # Pass action
        else:
            # Do not include pass action if there are valid moves
            pass

        return valid_actions

    def _determine_winner(self):
        # Determine winner according to tie-breaker rules
        player_heights = self.tower_heights
        if player_heights[0] > player_heights[1]:
            winner = 0
        elif player_heights[1] > player_heights[0]:
            winner = 1
        else:
            # Heights are equal, check number of blocks used
            player_blocks_used = [
                self.max_block_counts.sum() - self.block_counts[0].sum(),
                self.max_block_counts.sum() - self.block_counts[1].sum(),
            ]
            if player_blocks_used[0] < player_blocks_used[1]:
                winner = 0
            elif player_blocks_used[1] < player_blocks_used[0]:
                winner = 1
            else:
                # Same number of blocks used, player who placed the last block wins
                winner = self.last_player

        return winner

    def _get_obs(self):
        # Observation: [current_player_tower_height, opponent_player_tower_height,
        #               current_player_block_counts[0-4], opponent_player_block_counts[0-4]]
        current_player_blocks = self.block_counts[self.current_player]
        opponent_player_blocks = self.block_counts[1 - self.current_player]
        obs = np.concatenate(
            (
                np.array(
                    [
                        self.tower_heights[self.current_player],
                        self.tower_heights[1 - self.current_player],
                    ]
                ),
                current_player_blocks,
                opponent_player_blocks,
            )
        ).astype(np.int32)
        return obs
