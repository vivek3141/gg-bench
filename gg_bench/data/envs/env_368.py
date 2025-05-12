import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 10 discrete actions (switch indices 0 to 9)
        self.action_space = spaces.Discrete(10)

        # Observation space: 10-dimensional vector representing switch states (0 for OFF, 1 for ON)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize all switches to OFF (0)
        self.switches = np.zeros(10, dtype=np.int8)
        # Player 1 starts; can represent players as 1 and -1 if needed
        self.current_player = 1
        # Game not over
        self.done = False
        return self.switches.copy(), {}

    def step(self, action):
        # Check for invalid action: action is out of bounds or switch is already ON
        if self.done or self.switches[action] == 1:
            return self.switches.copy(), -10, True, False, {}

        # Toggle the chosen switch to ON
        self.switches[action] = 1
        selected_switch_number = action + 1  # Switch numbers are 1-indexed

        # Identify and toggle affected switches (excluding the chosen switch)
        for i in range(10):
            if i != action:
                switch_number = i + 1
                if (selected_switch_number % switch_number == 0) or (
                    switch_number % selected_switch_number == 0
                ):
                    # Toggle the switch state
                    self.switches[i] = 1 - self.switches[i]

        # Switch to the other player
        self.current_player *= -1

        # Check if the next player has any valid moves
        if len(self.valid_moves()) == 0:
            # Current player wins because the next player cannot make a move
            self.done = True
            return self.switches.copy(), 1, True, False, {}
        else:
            # Game continues
            return self.switches.copy(), 0, False, False, {}

    def render(self):
        # Create a string representation of the switch states
        display = "Switches:\n"
        for i in range(10):
            switch_number = i + 1
            state = "ON" if self.switches[i] == 1 else "OFF"
            display += f"{switch_number}[{state}] "
        return display

    def valid_moves(self):
        # Return a list of indices for switches that are OFF
        return [i for i in range(10) if self.switches[i] == 0]
