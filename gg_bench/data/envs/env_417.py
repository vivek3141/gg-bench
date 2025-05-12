import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 7 possible actions
        self.action_space = spaces.Discrete(7)

        # Define observation space
        # Observation is an array of 14 integers:
        # [own_score, own +1 uses, own +3 uses, own +5 uses, own ×2 uses, own ×3 uses, own reset uses,
        #  opponent_score, opponent +1 uses, opponent +3 uses, opponent +5 uses, opponent ×2 uses, opponent ×3 uses, opponent reset uses]
        low = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,  # Own score and operations
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # Opponent score and operations
            dtype=np.int32,
        )
        high = np.array([50, 3, 2, 1, 2, 1, 1, 50, 3, 2, 1, 2, 1, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize scores
        self.scores = [0, 0]  # [Player 1 score, Player 2 score]

        # Initialize remaining uses of operations for each player
        # Each player's operations: {operation_name: remaining_uses}
        self.operation_counts = [
            {"+1": 3, "+3": 2, "+5": 1, "×2": 2, "×3": 1, "reset": 1},
            {"+1": 3, "+3": 2, "+5": 1, "×2": 2, "×3": 1, "reset": 1},
        ]

        # Current player (0 for Player 1, 1 for Player 2)
        self.current_player = 0
        self.done = False

        return self._get_obs(), {}

    def _get_obs(self):
        # Get observation from the current state
        own_idx = self.current_player
        opp_idx = 1 - self.current_player

        own_score = self.scores[own_idx]
        opp_score = self.scores[opp_idx]

        own_ops = self.operation_counts[own_idx]
        opp_ops = self.operation_counts[opp_idx]

        obs = np.array(
            [
                own_score,
                own_ops["+1"],
                own_ops["+3"],
                own_ops["+5"],
                own_ops["×2"],
                own_ops["×3"],
                own_ops["reset"],
                opp_score,
                opp_ops["+1"],
                opp_ops["+3"],
                opp_ops["+5"],
                opp_ops["×2"],
                opp_ops["×3"],
                opp_ops["reset"],
            ],
            dtype=np.int32,
        )

        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Verify if the action is valid
        if action not in self.valid_moves():
            self.done = True
            return self._get_obs(), -10, True, False, {"reason": "Invalid move"}

        current_player = self.current_player
        opponent_player = 1 - self.current_player

        own_score = self.scores[current_player]
        opp_score = self.scores[opponent_player]
        own_ops = self.operation_counts[current_player]

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Map action to operation
        if action == 0:
            operation = "+1"
            own_score += 1
            own_ops[operation] -= 1
        elif action == 1:
            operation = "+3"
            own_score += 3
            own_ops[operation] -= 1
        elif action == 2:
            operation = "+5"
            own_score += 5
            own_ops[operation] -= 1
        elif action == 3:
            operation = "×2"
            own_score *= 2
            own_ops[operation] -= 1
        elif action == 4:
            operation = "×3"
            own_score *= 3
            own_ops[operation] -= 1
        elif action == 5:
            operation = "reset"
            own_score = 0
            own_ops[operation] -= 1
        elif action == 6:
            operation = "reset"
            opp_score = 0
            own_ops[operation] -= 1
        else:
            self.done = True
            return self._get_obs(), -10, True, False, {"reason": "Invalid action"}

        # Check if the operation causes the score to exceed 50
        if own_score > 50:
            self.done = True
            return self._get_obs(), -10, True, False, {"reason": "Score exceeds 50"}

        # Update scores
        self.scores[current_player] = own_score
        self.scores[opponent_player] = opp_score

        # Check for victory
        if own_score == 50:
            reward = 1
            terminated = True
            self.done = True
            info["reason"] = "Win"
            return self._get_obs(), reward, terminated, False, info

        # Switch to the other player
        self.current_player = opponent_player

        # Check if the next player can make any valid move
        if not self.valid_moves():
            # The next player cannot make a valid move; current player wins
            self.current_player = current_player  # Switch back to the winner
            reward = 1
            terminated = True
            self.done = True
            info["reason"] = "Opponent cannot make a valid move"
            return self._get_obs(), reward, terminated, False, info

        return self._get_obs(), reward, terminated, False, info

    def valid_moves(self):
        if self.done:
            return []

        current_player = self.current_player
        opponent_player = 1 - current_player

        own_score = self.scores[current_player]
        opp_score = self.scores[opponent_player]
        own_ops = self.operation_counts[current_player]

        valid_moves = []

        # Check each operation for validity
        # Addition operations
        if own_ops["+1"] > 0 and own_score + 1 <= 50:
            valid_moves.append(0)
        if own_ops["+3"] > 0 and own_score + 3 <= 50:
            valid_moves.append(1)
        if own_ops["+5"] > 0 and own_score + 5 <= 50:
            valid_moves.append(2)

        # Multiplication operations
        if own_ops["×2"] > 0 and own_score * 2 <= 50:
            valid_moves.append(3)
        if own_ops["×3"] > 0 and own_score * 3 <= 50:
            valid_moves.append(4)

        # Reset operations
        if own_ops["reset"] > 0:
            valid_moves.append(5)  # Reset self
            valid_moves.append(6)  # Reset opponent

        return valid_moves

    def render(self):
        # Generate a string representation of the current state
        current_player = self.current_player
        opponent_player = 1 - current_player
        own_ops = self.operation_counts[current_player]
        opp_ops = self.operation_counts[opponent_player]

        state_str = f"Current turn: Player {current_player + 1}\n"
        state_str += (
            f"Player {current_player + 1} - Score: {self.scores[current_player]}\n"
        )
        state_str += "Remaining operations:\n"
        for op, count in own_ops.items():
            state_str += f"  {op}: {count}\n"

        state_str += (
            f"\nPlayer {opponent_player + 1} - Score: {self.scores[opponent_player]}\n"
        )
        state_str += "Remaining operations:\n"
        for op, count in opp_ops.items():
            state_str += f"  {op}: {count}\n"

        return state_str
