import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 10 actions: numbers 1-9 (indices 0-8) and 'pass' action (index 9)
        self.action_space = spaces.Discrete(10)
        # Observation space:
        # - Indices 0-8: Status of numbers 1-9 (-1: opponent, 0: available, 1: current player)
        # - Index 9: Current player's total sum (0-15)
        # - Index 10: Opponent's total sum (0-15)
        self.observation_space = spaces.Box(
            low=np.array([-1] * 9 + [0, 0]),
            high=np.array([1] * 9 + [15, 15]),
            dtype=np.int8,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Status of numbers 1-9: 0 means available, 1 means taken by current player, -1 by opponent
        self.available_numbers = np.zeros(
            9, dtype=np.int8
        )  # Indices 0-8 represent numbers 1-9
        # Player sums: keys are 1 and -1 for each player
        self.player_sums = {1: 0, -1: 0}
        # Start with player 1
        self.current_player = 1
        # Number of consecutive passes
        self.passes_consecutive = 0
        # Last player who made a valid move
        self.last_valid_player = None
        # Game done flag
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_obs(),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        reward = 0
        info = {}

        # Check if action is valid
        if action == 9:  # Pass action
            # Check if player must pass (cannot select any number without exceeding 15)
            can_select = False
            for idx in range(9):
                if (
                    self.available_numbers[idx] == 0
                    and self.player_sums[self.current_player] + (idx + 1) <= 15
                ):
                    can_select = True
                    break
            if can_select:
                # Passing when a valid move is available is invalid
                reward = -10
                self.done = True
                return self._get_obs(), reward, True, False, info
            else:
                # Valid pass
                self.passes_consecutive += 1
                if self.passes_consecutive >= 2:
                    # Game ends
                    self.done = True
                    winner = self._determine_winner()
                    if winner == self.current_player:
                        reward = 1
                    return self._get_obs(), reward, True, False, info
                else:
                    # Switch player
                    self.current_player *= -1
                    return self._get_obs(), reward, False, False, info
        elif 0 <= action <= 8:  # Selecting number action + 1
            number = action + 1
            if self.available_numbers[action] != 0:
                # Number already taken
                reward = -10
                self.done = True
                return self._get_obs(), reward, True, False, info
            elif self.player_sums[self.current_player] + number > 15:
                # Exceeds sum limit
                reward = -10
                self.done = True
                return self._get_obs(), reward, True, False, info
            else:
                # Valid move
                self.available_numbers[action] = self.current_player
                self.player_sums[self.current_player] += number
                self.passes_consecutive = 0
                self.last_valid_player = self.current_player
                # Check for win condition
                if self.player_sums[self.current_player] == 15:
                    reward = 1
                    self.done = True
                    return self._get_obs(), reward, True, False, info
                # Check if no valid moves remain
                if not self._any_valid_moves():
                    # Game ends
                    self.done = True
                    winner = self._determine_winner()
                    if winner == self.current_player:
                        reward = 1
                    return self._get_obs(), reward, True, False, info
                # Switch player
                self.current_player *= -1
                return self._get_obs(), reward, False, False, info
        else:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, info

    def render(self):
        lines = []
        lines.append(
            "Available Numbers: "
            + " ".join([str(i + 1) for i in range(9) if self.available_numbers[i] == 0])
        )
        lines.append(f"Player 1's Total Sum: {self.player_sums[1]}")
        lines.append(f"Player 2's Total Sum: {self.player_sums[-1]}")
        lines.append(
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        return "\n".join(lines)

    def valid_moves(self):
        valid_actions = []
        # Check pass action
        can_select = False
        for idx in range(9):
            if (
                self.available_numbers[idx] == 0
                and self.player_sums[self.current_player] + (idx + 1) <= 15
            ):
                can_select = True
                break
        if not can_select:
            valid_actions.append(9)  # Pass
        else:
            # Add all valid number selections
            for idx in range(9):
                if (
                    self.available_numbers[idx] == 0
                    and self.player_sums[self.current_player] + (idx + 1) <= 15
                ):
                    valid_actions.append(idx)
        return valid_actions

    def _get_obs(self):
        # Observation includes number statuses and player sums
        obs = np.concatenate(
            (
                self.available_numbers,
                np.array(
                    [
                        self.player_sums[self.current_player],
                        self.player_sums[-self.current_player],
                    ],
                    dtype=np.int8,
                ),
            )
        )
        return obs

    def _any_valid_moves(self):
        # Check if any valid moves remain for both players
        for player in [1, -1]:
            has_move = False
            for idx in range(9):
                if (
                    self.available_numbers[idx] == 0
                    and self.player_sums[player] + (idx + 1) <= 15
                ):
                    has_move = True
                    break
            if has_move:
                return True
        return False

    def _determine_winner(self):
        # Determine winner based on sums and tie-breaker
        sum_player = self.player_sums[self.current_player]
        sum_opponent = self.player_sums[-self.current_player]
        if sum_player < 15 and sum_opponent < 15:
            if sum_player > sum_opponent:
                return self.current_player
            elif sum_player < sum_opponent:
                return -self.current_player
            else:
                # Sums are equal, player who made last valid move wins
                return self.last_valid_player
        elif sum_player < 15:
            return self.current_player
        elif sum_opponent < 15:
            return -self.current_player
        else:
            # Both exceeded 15 (should not happen)
            return None
