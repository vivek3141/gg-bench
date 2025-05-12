import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space:
        # Actions 0-6: Flip single switch at position 0-6
        # Actions 7-12: Flip two adjacent switches starting at positions 0-5
        self.action_space = spaces.Discrete(13)

        # Observation space: 7 switches, each can be 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly initialize the switches to 0 or 1
        self.switches = np.random.randint(0, 2, size=7, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Define target states for players
        # Player 1 aims for all switches On (1)
        # Player 2 aims for all switches Off (0)
        self.target_state = {1: 1, 2: 0}

        return self.switches.copy(), {}

    def step(self, action):
        if self.done:
            return self.switches.copy(), -10, True, False, {}

        # Validate action
        if action < 0 or action >= 13:
            # Invalid action
            self.done = True
            return self.switches.copy(), -10, True, False, {}

        if 0 <= action <= 6:
            # Flip single switch at position 'action'
            pos = action
            if pos < 0 or pos >= 7:
                # Invalid position
                self.done = True
                return self.switches.copy(), -10, True, False, {}
            # Flip the switch
            self.switches[pos] ^= 1

        elif 7 <= action <= 12:
            # Flip two adjacent switches starting at position 'pos'
            pos = action - 7
            if pos < 0 or pos >= 6:
                # Invalid position
                self.done = True
                return self.switches.copy(), -10, True, False, {}
            # Flip the switches
            self.switches[pos] ^= 1
            self.switches[pos + 1] ^= 1

        else:
            # Invalid action
            self.done = True
            return self.switches.copy(), -10, True, False, {}

        # Check for victory
        if np.all(self.switches == self.target_state[self.current_player]):
            self.done = True
            return self.switches.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.switches.copy(), 0, False, False, {}

    def render(self):
        switch_str = " ".join(str(s) for s in self.switches)
        return f"Switches: {switch_str}\nCurrent Player: {self.current_player}"

    def valid_moves(self):
        # All actions are valid unless the game is over
        if self.done:
            return []
        else:
            return list(range(13))
