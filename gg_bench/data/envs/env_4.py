import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(7)

        # Observation space: the state of the 4 switches, values -1 ('B') or 1 ('A')
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial switch configuration: -1 or 1 for each switch
        self.switches = np.random.choice([-1, 1], size=4).astype(np.float32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.switches.copy(), {}

    def step(self, action):
        if self.done:
            return self.switches.copy(), 0, True, False, {}

        if action < 0 or action >= 7:
            # Invalid action
            reward = -10
            self.done = True
            return self.switches.copy(), reward, True, False, {}

        # Apply the action
        if action == 0:
            self._flip_switch(0)
        elif action == 1:
            self._flip_switch(1)
        elif action == 2:
            self._flip_switch(2)
        elif action == 3:
            self._flip_switch(3)
        elif action == 4:
            self._flip_switch(0)
            self._flip_switch(1)
        elif action == 5:
            self._flip_switch(1)
            self._flip_switch(2)
        elif action == 6:
            self._flip_switch(2)
            self._flip_switch(3)

        # Check for win condition
        if self.current_player == 1 and np.all(self.switches == 1):
            self.done = True
            reward = 1
        elif self.current_player == -1 and np.all(self.switches == -1):
            self.done = True
            reward = 1
        else:
            reward = 0

        if not self.done:
            # Switch player
            self.current_player *= -1

        return self.switches.copy(), reward, self.done, False, {}

    def _flip_switch(self, index):
        # Flip the state of the switch at position index
        self.switches[index] *= -1

    def render(self):
        # Return a string representation of the switches
        switch_states = ["A" if s == 1 else "B" for s in self.switches]
        return "Switches: " + " ".join(switch_states)

    def valid_moves(self):
        # All actions are valid unless the game is over
        if self.done:
            return []
        else:
            return list(range(7))
