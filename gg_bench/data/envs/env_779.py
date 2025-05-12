import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are integers from 0 to 4, representing numbers 1 to 5
        self.action_space = spaces.Discrete(5)

        # Observations are the points of both players: [agent_points, opponent_points]
        # Points can range from -15 to 15
        self.observation_space = spaces.Box(
            low=-15, high=15, shape=(2,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_points = 15
        self.opponent_points = 15
        self.in_sudden_death = False
        self.done = False
        # Return observation and info
        return np.array([self.agent_points, self.opponent_points], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is over, no further moves are allowed
            return (
                np.array([self.agent_points, self.opponent_points], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        if action not in [0, 1, 2, 3, 4]:
            # Invalid action
            self.done = True
            return (
                np.array([self.agent_points, self.opponent_points], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Agent's selected number
        agent_number = action + 1  # Map 0-4 to 1-5

        # Opponent's action - randomly select a number between 1 and 5
        opponent_action = self.np_random.integers(0, 5)
        opponent_number = opponent_action + 1

        # Apply game rules
        if agent_number != opponent_number:
            if agent_number > opponent_number:
                # Agent subtracts their own number from opponent's points
                self.opponent_points -= agent_number
            else:
                # Opponent subtracts their own number from agent's points
                self.agent_points -= opponent_number
        else:
            # Both players subtract their own number from their own points
            self.agent_points -= agent_number
            self.opponent_points -= opponent_number

        # Check for game over conditions
        if self.agent_points <= 0 and self.opponent_points <= 0:
            if agent_number == opponent_number:
                # Both players selected the same number, sudden death is initiated
                self.in_sudden_death = True
                self.done = False  # Game continues
                reward = 0
            else:
                # The player with the higher number wins
                if agent_number > opponent_number:
                    # Agent wins
                    self.done = True
                    reward = 1
                else:
                    # Opponent wins
                    self.done = True
                    reward = 0
        elif self.agent_points <= 0:
            # Agent loses
            self.done = True
            reward = 0
        elif self.opponent_points <= 0:
            # Agent wins
            self.done = True
            reward = 1
        else:
            # Game continues
            self.done = False
            reward = 0

        # Return observation, reward, terminated, truncated, info
        return (
            np.array([self.agent_points, self.opponent_points], dtype=np.int32),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        state = (
            f"Agent Points: {self.agent_points}\n"
            f"Opponent Points: {self.opponent_points}\n"
        )
        if self.in_sudden_death:
            state += "Sudden Death Mode!\n"
        return state

    def valid_moves(self):
        # All actions are valid unless the game is over
        if self.done:
            return []
        else:
            return [0, 1, 2, 3, 4]
