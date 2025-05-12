import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0: Add 3, 1: Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # current_number: float (1 to 200)
        # last_function_used: int (-1, 0, or 1) (-1 for None, 0 for Add 3, 1 for Multiply by 2)
        # consecutive_use_count: int (0 to 2)
        # current_player: float (-1 or 1)
        self.observation_space = spaces.Box(
            low=np.array([1, -1, 0, -1], dtype=np.float32),
            high=np.array([200, 1, 2, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_number = 1
        self.current_player = 1  # Player 1 starts, can be 1 or -1

        # For each player, store last_function_used and consecutive_use_count
        self.last_function_used = {
            1: None,
            -1: None,
        }  # None, 0 (Add 3), or 1 (Multiply by 2)
        self.consecutive_use_count = {1: 0, -1: 0}

        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Apply the action
        if action == 0:  # Add 3
            self.current_number += 3
        elif action == 1:  # Multiply by 2
            self.current_number *= 2

        # Update function usage counts
        if self.last_function_used[self.current_player] == action:
            self.consecutive_use_count[self.current_player] += 1
        else:
            self.consecutive_use_count[self.current_player] = 1

        # Update last function used
        self.last_function_used[self.current_player] = action

        # Check for win condition
        if self.current_number >= 100:
            reward = 1
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # Switch to next player
        self.current_player *= -1

        # Return observation
        observation = self._get_observation()
        reward = 0
        return observation, reward, self.done, False, {}

    def render(self):
        function_names = {0: "Add 3", 1: "Multiply by 2", None: "None"}
        last_func = self.last_function_used[self.current_player]
        last_func_name = function_names[last_func]
        consecutive = self.consecutive_use_count[self.current_player]
        player_name = "Player 1" if self.current_player == 1 else "Player 2"

        return (
            f"Current Number: {self.current_number}\n"
            f"Current Player: {player_name}\n"
            f"Last Function Used: {last_func_name}\n"
            f"Consecutive Uses: {consecutive}\n"
        )

    def valid_moves(self):
        valid_actions = []
        for action in [0, 1]:
            if (
                self.last_function_used[self.current_player] == action
                and self.consecutive_use_count[self.current_player] == 2
            ):
                continue  # Can't use the same function more than twice in a row
            valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        last_func = self.last_function_used[self.current_player]
        # Map None to -1 for last_function_used in observation
        last_func_obs = -1 if last_func is None else last_func
        observation = np.array(
            [
                self.current_number,
                float(last_func_obs),  # last function used
                float(self.consecutive_use_count[self.current_player]),
                float(self.current_player),
            ],
            dtype=np.float32,
        )
        return observation
