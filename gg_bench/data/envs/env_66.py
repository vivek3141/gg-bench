import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0-9 correspond to numbers 1-10
        self.action_space = spaces.Discrete(10)

        # Observation: [current player's HP, opponent's HP]
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_HP = 100
        self.player2_HP = 100
        self.current_player = 1
        self.done = False
        # Observation: [current player's HP, opponent's HP]
        observation = np.array([self.player1_HP, self.player2_HP], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if action is valid
        if action not in self.valid_moves():
            self.done = True
            reward = -10
            # Observation remains the same
            if self.current_player == 1:
                observation = np.array(
                    [self.player1_HP, self.player2_HP], dtype=np.int32
                )
            else:
                observation = np.array(
                    [self.player2_HP, self.player1_HP], dtype=np.int32
                )
            return (
                observation,
                reward,
                self.done,
                False,
                {},
            )  # Terminated due to invalid move

        N = action + 1  # Map action 0-9 to numbers 1-10

        # Get current and opponent HP
        if self.current_player == 1:
            current_player_HP = self.player1_HP
            opponent_HP = self.player2_HP
        else:
            current_player_HP = self.player2_HP
            opponent_HP = self.player1_HP

        # Check for critical strike
        if opponent_HP % N == 0:
            damage = 2 * N
        else:
            damage = N

        # Update opponent's HP
        opponent_HP -= damage
        opponent_HP = max(opponent_HP, 0)  # Ensure HP doesn't go below zero

        # Update the HP in the environment
        if self.current_player == 1:
            self.player2_HP = opponent_HP
        else:
            self.player1_HP = opponent_HP

        # Check for victory
        if opponent_HP <= 0:
            reward = 1  # Current player wins
            self.done = True
        else:
            reward = 0
            self.done = False

        # Switch turns
        self.current_player = 2 if self.current_player == 1 else 1

        # Prepare observation for the next player
        if self.current_player == 1:
            observation = np.array([self.player1_HP, self.player2_HP], dtype=np.int32)
        else:
            observation = np.array([self.player2_HP, self.player1_HP], dtype=np.int32)

        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def render(self):
        render_str = (
            f"Player 1's HP: {self.player1_HP} | Player 2's HP: {self.player2_HP}\n"
        )
        render_str += f"It's Player {self.current_player}'s turn."
        return render_str

    def valid_moves(self):
        return list(range(10))  # Valid actions are 0-9 corresponding to numbers 1-10
