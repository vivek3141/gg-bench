import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # The action space consists of all possible valid actions
        # Single flips: positions 1 to 7 (indices 0 to 6)
        # Double flips: adjacent pairs from positions 1-2 to 6-7 (indices 7 to 12)
        # Total of 13 actions
        self.action_space = spaces.Discrete(13)

        # Observation space is the state of the switches, 0 (off) or 1 (on)
        # The switches are positions 1 to 7 (indices 0 to 6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All switches are initially off (0)
        self.switches = np.zeros(7, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.switches.copy(), {}  # Return observation and info

    def step(self, action):
        # Check for invalid action
        valid_actions = self.valid_moves()
        if action not in valid_actions or self.done:
            self.done = True
            return (
                self.switches.copy(),
                -10,
                True,
                False,
                {},  # Observation, reward, terminated, truncated, info
            )

        # Perform the action
        if action >= 0 and action <= 6:
            # Single flip at position action + 1
            position = action
            self.switches[position] = 1 - self.switches[position]
        elif action >= 7 and action <= 12:
            # Flip two adjacent switches
            position = action - 7
            self.switches[position] = 1 - self.switches[position]
            self.switches[position + 1] = 1 - self.switches[position + 1]

        # Check for win condition
        if np.all(self.switches == 1):
            self.done = True
            return self.switches.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.switches.copy(), 0, False, False, {}

    def render(self):
        # Return a string representing the current state
        state_str = " ".join(map(str, self.switches))
        return f"Current State: {state_str}"

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = list(range(13))  # Initially, all actions are possible
        # Exclude actions that are invalid based on the current state
        # Since any switch can be flipped at any time, and the game rules allow flipping any single switch
        # For double flips, ensure that positions are adjacent within range 0 to 6
        # No additional validation is needed here, all actions are valid unless the game is over
        if self.done:
            return []
        return valid_actions


# Create the environment
env = CustomEnv()

# Reset the environment
obs, info = env.reset()

done = False
while not done:
    # Get the list of valid moves
    valid_actions = env.valid_moves()

    # Choose an action (for example, random valid action)
    action = np.random.choice(valid_actions)

    # Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)

    # Render the current state
    print(env.render())

    # Print the reward
    print(f"Reward: {reward}\n")
