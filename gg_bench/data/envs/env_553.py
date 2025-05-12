import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: integers from 0 to 7, corresponding to multipliers 2 to 9
        self.action_space = spaces.Discrete(8)

        # Observation space: [cumulative_total, current_player]
        # cumulative_total ranges from 1 to a reasonable upper limit (e.g., 1e6)
        # current_player is 0 (Agent) or 1 (Opponent)
        self.observation_space = spaces.Box(
            low=np.array([1, 0]),
            high=np.array([1e6, 1]),
            shape=(2,),
            dtype=np.int32,
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_total = 1
        self.current_player = 0  # 0 for Agent, 1 for Opponent
        return np.array([self.cumulative_total, self.current_player]), {}

    def step(self, action):
        info = {}
        terminated = False
        reward = -10  # Default reward for a valid move

        if self.current_player == 0:
            # Agent's turn
            multiplier = action + 2  # Map action 0-7 to multiplier 2-9

            if multiplier < 2 or multiplier > 9:
                # Invalid move
                reward = -10
                terminated = True
                return (
                    np.array([self.cumulative_total, self.current_player]),
                    reward,
                    terminated,
                    False,
                    info,
                )

            self.cumulative_total *= multiplier

            if self.cumulative_total > 100:
                # Agent loses
                reward = -10
                terminated = True
                return (
                    np.array([self.cumulative_total, self.current_player]),
                    reward,
                    terminated,
                    False,
                    info,
                )
            else:
                # Valid move, switch to Opponent
                self.current_player = 1
        else:
            # Opponent's turn
            valid_actions = self.valid_moves()
            opponent_action = np.random.choice(valid_actions)
            multiplier = opponent_action + 2
            self.cumulative_total *= multiplier

            if self.cumulative_total > 100:
                # Opponent loses, Agent wins
                reward = 1
                terminated = True
            else:
                # Valid move, switch back to Agent
                self.current_player = 0

        return (
            np.array([self.cumulative_total, self.current_player]),
            reward,
            terminated,
            False,
            info,
        )

    def render(self):
        return f"Current total: {self.cumulative_total}, Player {'Agent' if self.current_player == 0 else 'Opponent'}'s turn."

    def valid_moves(self):
        # Return all possible actions (0 to 7 corresponding to multipliers 2 to 9)
        return list(range(8))
