import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(9) for combinations of actions (1,1) to (3,3)
        self.action_space = spaces.Discrete(9)

        # Observation space: Player 1 HP and Player 2 HP
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_hp = 10
        self.player2_hp = 10
        self.done = False
        observation = np.array([self.player1_hp, self.player2_hp], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return current state with zero reward
            observation = np.array([self.player1_hp, self.player2_hp], dtype=np.int32)
            return observation, 0, True, False, {}

        # Map action to player choices
        action_mapping = {
            0: (1, 1),
            1: (1, 2),
            2: (1, 3),
            3: (2, 1),
            4: (2, 2),
            5: (2, 3),
            6: (3, 1),
            7: (3, 2),
            8: (3, 3),
        }

        if action not in action_mapping:
            # Invalid action
            observation = np.array([self.player1_hp, self.player2_hp], dtype=np.int32)
            return observation, -10, True, False, {}

        p1_choice, p2_choice = action_mapping[action]

        # Resolve actions
        if p1_choice != p2_choice:
            # Different numbers chosen
            if p1_choice > p2_choice:
                # Player 1 attacks Player 2
                self.player2_hp -= p1_choice
            else:
                # Player 2 attacks Player 1
                self.player1_hp -= p2_choice
        else:
            # Same numbers chosen, both players take self-damage
            self.player1_hp -= p1_choice
            self.player2_hp -= p2_choice

        # Check for game end
        self.done = self.player1_hp <= 0 or self.player2_hp <= 0

        # Determine reward
        if self.done:
            if self.player1_hp <= 0 and self.player2_hp <= 0:
                # Both players have HP <= 0, but the rules state there's always a winner
                # In this implementation, we'll declare Player 2 as the winner in this scenario
                reward = -1  # Current player loses
            elif self.player1_hp <= 0:
                reward = -1  # Player 1 loses
            else:
                reward = 1  # Player 1 wins
        else:
            reward = 0  # No reward on non-terminal steps

        # Prepare observation
        self.player1_hp = max(0, self.player1_hp)
        self.player2_hp = max(0, self.player2_hp)
        observation = np.array([self.player1_hp, self.player2_hp], dtype=np.int32)

        return observation, reward, self.done, False, {}

    def render(self):
        state_str = f"Player 1 HP: {self.player1_hp}\nPlayer 2 HP: {self.player2_hp}\n"
        return state_str

    def valid_moves(self):
        # All actions from 0 to 8 are valid
        return list(range(9))
