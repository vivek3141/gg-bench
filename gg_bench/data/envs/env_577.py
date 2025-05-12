import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 20 possible actions (weights 1-10 with + or -)
        self.action_space = spaces.Discrete(20)

        # Observation space: [balance, available_weights(10), current_player]
        self.observation_space = spaces.Box(
            low=np.array([-55] + [0] * 10 + [0], dtype=np.int32),
            high=np.array([55] + [1] * 10 + [1], dtype=np.int32),
            shape=(12,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 0
        self.available_weights = {i: True for i in range(1, 11)}
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        return self._get_obs(), self.info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, self.info

        weight, operation = self._decode_action(action)

        # Check if weight is available
        if not self.available_weights.get(weight, False):
            self.done = True
            return self._get_obs(), -10, True, False, self.info

        # Apply operation
        if operation == "+":
            new_balance = self.balance + weight
        else:
            new_balance = self.balance - weight

        # Temporarily remove weight and update balance to check validity
        temp_available_weights = self.available_weights.copy()
        temp_available_weights[weight] = False

        # Check for valid move
        if not self._is_valid_state(new_balance, temp_available_weights):
            self.done = True
            return self._get_obs(), -10, True, False, self.info

        # Update state
        self.balance = new_balance
        self.available_weights[weight] = False

        # Check for win
        if self.balance == 0:
            self.done = True
            return self._get_obs(), 1, True, False, self.info

        # Check if opponent has any valid moves
        opponent_valid_moves = self._get_valid_actions(
            self.balance, temp_available_weights
        )
        if not opponent_valid_moves:
            self.done = True
            return self._get_obs(), 1, True, False, self.info

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1
        return self._get_obs(), 0, False, False, self.info

    def render(self):
        available_weights = [str(i) for i in range(1, 11) if self.available_weights[i]]
        state = f"Current Balance: {self.balance}\n"
        state += f"Available Weights: {', '.join(available_weights)}\n"
        state += f"Current Player: Player {self.current_player}\n"
        return state

    def valid_moves(self):
        return self._get_valid_actions(self.balance, self.available_weights)

    # Helper methods
    def _get_obs(self):
        available = [1 if self.available_weights[i] else 0 for i in range(1, 11)]
        obs = np.array(
            [self.balance] + available + [self.current_player - 1], dtype=np.int32
        )
        return obs

    def _decode_action(self, action):
        weight = (action // 2) + 1
        operation = "+" if action % 2 == 0 else "-"
        return weight, operation

    def _is_valid_state(self, balance, available_weights):
        # If balance is zero, state is valid
        if balance == 0:
            return True

        # Check if opponent has any valid moves
        opponent_valid_moves = self._get_valid_actions(balance, available_weights)
        return bool(opponent_valid_moves)

    def _get_valid_actions(self, balance, available_weights):
        valid_actions = []
        for weight in range(1, 11):
            if available_weights.get(weight, False):
                for op in ["+", "-"]:
                    if op == "+":
                        new_balance = balance + weight
                    else:
                        new_balance = balance - weight

                    temp_available_weights = available_weights.copy()
                    temp_available_weights[weight] = False

                    # If move results in balance zero or opponent can make a move
                    if new_balance == 0 or self._opponent_can_move(
                        new_balance, temp_available_weights
                    ):
                        action = (weight - 1) * 2 + (0 if op == "+" else 1)
                        valid_actions.append(action)
        return valid_actions

    def _opponent_can_move(self, balance, available_weights):
        for weight in range(1, 11):
            if available_weights.get(weight, False):
                for op in ["+", "-"]:
                    if op == "+":
                        new_balance = balance + weight
                    else:
                        new_balance = balance - weight
                    if new_balance == 0:
                        return True
                    if self._has_future_moves(new_balance, available_weights, weight):
                        return True
        return False

    def _has_future_moves(self, balance, available_weights, used_weight):
        temp_available_weights = available_weights.copy()
        temp_available_weights[used_weight] = False
        for weight in range(1, 11):
            if temp_available_weights.get(weight, False):
                return True
        return balance == 0
