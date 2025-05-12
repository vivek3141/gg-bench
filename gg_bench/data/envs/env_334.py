import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the game state
        self.locks_status = np.full(
            (10,), -1, dtype=np.int8
        )  # All locks are unopened (-1)
        self.locks_feedback = np.zeros((10,), dtype=np.int8)  # Feedback for each lock
        self.key_location = None
        self.current_player = 1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the game state
        self.locks_status = np.full(
            (10,), -1, dtype=np.int8
        )  # All locks are unopened (-1)
        self.locks_feedback = np.zeros(
            (10,), dtype=np.int8
        )  # All feedback is 0 (no feedback)
        self.key_location = self.np_random.integers(
            0, 10
        )  # Randomly select key location from 0 to 9
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0
        terminated = False
        truncated = False
        info = {}

        if action < 0 or action >= 10 or self.locks_status[action] != -1:
            # Invalid action
            reward = -10  # Penalty for invalid move
            terminated = True
            self.done = True
        else:
            # Valid action
            if action == self.key_location:
                # Player opens the lock containing the key
                self.locks_status[action] = 1
                self.done = True
                terminated = True
                reward = 1  # Current player wins
            else:
                # Open the lock, get feedback
                self.locks_status[action] = 0
                if action < self.key_location:
                    # Key is in higher-numbered lock
                    self.locks_feedback[action] = 1
                elif action > self.key_location:
                    # Key is in lower-numbered lock
                    self.locks_feedback[action] = -1
                # Switch players
                self.current_player = 2 if self.current_player == 1 else 1
            # Reward remains 0 for valid move without winning

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        return np.concatenate([self.locks_status, self.locks_feedback])

    def render(self):
        state_str = "Locks Status:\n"
        for i in range(10):
            lock_num = i + 1
            status = self.locks_status[i]
            if status == -1:
                status_str = "Closed"
            elif status == 0:
                status_str = "Opened (Empty)"
            elif status == 1:
                status_str = "Opened (Key Found!)"
            else:
                status_str = "Unknown"
            feedback = self.locks_feedback[i]
            if feedback == 1:
                feedback_str = "Key is in higher-numbered lock."
            elif feedback == -1:
                feedback_str = "Key is in lower-numbered lock."
            else:
                feedback_str = ""
            state_str += f"Lock {lock_num}: {status_str}"
            if feedback_str:
                state_str += f" Feedback: {feedback_str}"
            state_str += "\n"
        return state_str

    def valid_moves(self):
        return [i for i in range(10) if self.locks_status[i] == -1]
