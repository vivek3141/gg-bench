import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.max_number = 50  # Maximum number for the starting current number
        self.action_space = spaces.Discrete(self.max_number)
        self.observation_space = spaces.Box(
            low=0, high=self.max_number, shape=(5,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 20  # Starting Number can be adjusted
        self.phase = "split"  # 'split' or 'choose'
        self.split_option1 = 0
        self.split_option2 = 0
        self.current_player = 1  # 1 or -1 to switch between players
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        if self.phase == "split":
            if self.current_number == 1:
                # Current player cannot split 1 and loses
                self.done = True
                reward = -1  # Current player loses
                return self._get_observation(), reward, True, False, {}

            # Perform the split
            self.split_option1 = action
            self.split_option2 = self.current_number - action
            self.phase = "choose"
            reward = 0
        elif self.phase == "choose":
            # Action must be 0 or 1
            if action == 0:
                selected_number = self.split_option1
            elif action == 1:
                selected_number = self.split_option2
            else:
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, {}

            self.current_number = selected_number

            if self.current_number == 1:
                # Next player cannot split and loses; current player wins
                self.done = True
                reward = 1  # Current player wins
                return self._get_observation(), reward, True, False, {}

            # Switch to split phase and change current player
            self.phase = "split"
            self.current_player *= -1
            reward = 0

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        if self.phase == "split":
            phase = "Split Phase"
        else:
            phase = "Choose Phase"
        render_str = (
            f"Current Player: {self.current_player}\n"
            f"Current Number: {self.current_number}\n"
            f"Phase: {phase}\n"
            f"Split Options: {self.split_option1}, {self.split_option2}\n"
        )
        return render_str

    def valid_moves(self):
        if self.phase == "split":
            if self.current_number == 1:
                return []
            else:
                # Valid splits are integers from 1 to current_number - 1
                return list(range(1, self.current_number))
        elif self.phase == "choose":
            # Choices are 0 or 1 for the two split options
            return [0, 1]
        else:
            return []

    def _get_observation(self):
        # Observation includes current number, phase, split options, and current player
        observation = np.array(
            [
                self.current_number,
                0 if self.phase == "split" else 1,  # Phase indicator
                self.split_option1,
                self.split_option2,
                self.current_player,
            ],
            dtype=np.int32,
        )
        return observation
