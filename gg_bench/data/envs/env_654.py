import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 20 discrete actions
        # Actions 0-9: Select numbers 1-10 without removing opponent's top number
        # Actions 10-19: Select numbers 1-10 and choose to remove opponent's top number if possible
        self.action_space = spaces.Discrete(20)

        # Observation space:
        # [player_total, opponent_total, player_stack_size, opponent_stack_size,
        #  player_last5_numbers, opponent_last5_numbers]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(14,), dtype=np.int32
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_stacks = [[], []]  # Player 0 and Player 1 stacks
        self.player_totals = [0, 0]  # Player 0 and Player 1 totals
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action to number_selected and remove_opponent flag
        number_selected = (action % 10) + 1  # Map 0-9 to 1-10
        remove_opponent = action >= 10

        # Add number to current player's stack
        self.player_stacks[self.current_player].append(number_selected)
        self.player_totals[self.current_player] += number_selected
        current_total = self.player_totals[self.current_player]

        # Check for bust
        if current_total > 100:
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Check for victory
        if current_total == 100:
            reward = 1
            self.done = True
            # Check for tie
            opponent_total = self.player_totals[1 - self.current_player]
            if opponent_total == 100:
                player_stack_size = len(self.player_stacks[self.current_player])
                opponent_stack_size = len(self.player_stacks[1 - self.current_player])
                if player_stack_size < opponent_stack_size:
                    reward = 1  # Current player wins
                elif player_stack_size > opponent_stack_size:
                    reward = -1  # Current player loses
                else:
                    self.done = False  # Continue play if tie
                    reward = 0
            return self._get_obs(), reward, self.done, False, {}

        # Check for multiples of 10
        if current_total % 10 == 0:
            if remove_opponent:
                if self.player_stacks[1 - self.current_player]:
                    # Remove opponent's top number
                    removed_number = self.player_stacks[1 - self.current_player].pop()
                    self.player_totals[1 - self.current_player] -= removed_number
                # If opponent's stack is empty, no effect
            else:
                pass  # Player chose not to remove opponent's top number
        else:
            if remove_opponent:
                # Invalid action: cannot remove opponent's number when total is not multiple of 10
                reward = -10
                self.done = True
                return self._get_obs(), reward, True, False, {}

        # Switch current player
        self.current_player = 1 - self.current_player
        reward = 0
        return self._get_obs(), reward, False, False, {}

    def render(self):
        lines = []
        for i in range(2):
            lines.append(
                f"Player {i} Stack: {self.player_stacks[i]} (Total: {self.player_totals[i]})"
            )
        return "\n".join(lines)

    def valid_moves(self):
        valid_actions = []
        current_total = self.player_totals[self.current_player]
        for num in range(1, 11):
            potential_total = current_total + num
            if potential_total > 100:
                continue  # Skip numbers that would cause bust
            for remove in [False, True]:
                action = (num - 1) + (10 if remove else 0)
                if potential_total % 10 == 0:
                    valid_actions.append(action)
                else:
                    if not remove:
                        valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Prepare observation array
        obs = np.zeros(14, dtype=np.int32)

        # Current player data
        obs[0] = self.player_totals[self.current_player]
        obs[2] = len(self.player_stacks[self.current_player])

        # Opponent data
        obs[1] = self.player_totals[1 - self.current_player]
        obs[3] = len(self.player_stacks[1 - self.current_player])

        # Last 5 numbers in current player's stack (padded with zeros)
        last5_player = self.player_stacks[self.current_player][-5:]
        obs[4 : 4 + len(last5_player)] = last5_player

        # Last 5 numbers in opponent's stack (padded with zeros)
        last5_opponent = self.player_stacks[1 - self.current_player][-5:]
        obs[9 : 9 + len(last5_opponent)] = last5_opponent

        return obs
