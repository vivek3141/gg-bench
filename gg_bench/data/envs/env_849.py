import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(10), representing numbers 1 to 10
        self.action_space = spaces.Discrete(10)
        # The observation is a vector [cumulative_total, last_opponent_number]
        # cumulative_total ranges from 0 to at least 60 (to allow for totals exceeding 50)
        # last_opponent_number ranges from 0 to 10 (0 indicates no last number)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([60, 10]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_total = 0
        self.last_opponent_number = 0  # Start with 0, meaning no restriction
        self.current_player = 1  # Current player (1 or 2), agent plays both sides
        self.done = False

        observation = np.array(
            [self.cumulative_total, self.last_opponent_number], dtype=np.int32
        )
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # First, check if the agent has any valid moves
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # Current player has no valid moves and thus loses
            self.done = True
            return self._get_obs(), -1, True, False, {}

        # Check if the action is valid
        if action not in valid_actions:
            # Illegal move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Map action (0-9) to number (1-10)
        number_chosen = action + 1

        # Update cumulative total
        self.cumulative_total += number_chosen

        # Check if cumulative total >= 50
        if self.cumulative_total >= 50:
            # Current player loses
            self.done = True
            return self._get_obs(), -1, True, False, {}

        # Prepare for the next player's turn
        # Update last opponent's number
        self.last_opponent_number = number_chosen
        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player has any valid moves
        valid_actions_next = self.valid_moves()
        if len(valid_actions_next) == 0:
            # Next player has no valid moves and thus loses
            self.done = True
            # Current player wins
            return self._get_obs(), 1, True, False, {}

        # Game continues
        return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        return np.array(
            [self.cumulative_total, self.last_opponent_number], dtype=np.int32
        )

    def render(self):
        # Return a string representation of the state
        render_str = f"Cumulative Total: {self.cumulative_total}\n"
        if self.last_opponent_number == 0:
            render_str += "Last opponent's number: None\n"
        else:
            render_str += f"Last opponent's number: {self.last_opponent_number}\n"
        return render_str

    def valid_moves(self):
        # Returns the list of valid actions (indices from 0 to 9 corresponding to numbers 1-10)
        valid_actions = []
        for action in range(10):
            number = action + 1
            if number == self.last_opponent_number:
                continue
            if self.cumulative_total + number >= 50:
                continue
            valid_actions.append(action)
        return valid_actions
