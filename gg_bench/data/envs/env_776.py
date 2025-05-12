import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 for Add, 1 for Subtract
        self.action_space = spaces.Discrete(2)

        # Observation space: [current player's score, opponent's score, die roll]
        # Scores range from 0 to 15, die roll from 1 to 6
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1], dtype=np.int32),
            high=np.array([15, 15, 6], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the environment
        self.current_scores = None
        self.current_player = None
        self.done = None
        self.die_roll = None

        # Random number generator
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize scores and player
        self.current_scores = [0, 0]
        self.current_player = 0  # Player 1 starts
        self.done = False

        # Roll the die for the current player
        self.die_roll = self.np_random.integers(1, 7)

        # Check if the current player has any valid moves
        if not self.valid_moves():
            # Current player loses immediately
            self.done = True
            observation = self._get_observation()
            return observation, {}
        else:
            observation = self._get_observation()
            return observation, {}

    def step(self, action):
        if self.done:
            # Game is over
            observation = self._get_observation()
            return observation, 0, True, False, {}

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action results in immediate loss
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Apply the action
        current_score = self.current_scores[self.current_player]
        if action == 0:
            # Add the die roll to the current score
            new_score = current_score + self.die_roll
        else:
            # Subtract the die roll from the current score
            new_score = current_score - self.die_roll

        # Update the current player's score
        self.current_scores[self.current_player] = new_score

        # Check for win/loss conditions
        if new_score == 15:
            # Current player wins
            reward = 1
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}
        elif new_score < 0 or new_score > 15:
            # Current player loses
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}
        else:
            # Switch to next player
            self.current_player = 1 - self.current_player

            # Roll the die for the next player
            self.die_roll = self.np_random.integers(1, 7)

            # Check if next player has any valid moves
            if not self.valid_moves():
                # Next player loses immediately, current player wins
                reward = 1
                self.done = True
                # Switch back to previous player for correct observation
                self.current_player = 1 - self.current_player
                observation = self._get_observation()
                return observation, reward, True, False, {}
            else:
                # Game continues
                reward = 0
                observation = self._get_observation()
                return observation, reward, False, False, {}

    def render(self):
        # Return a string representation of the current state
        render_str = f"Current Player: Player {self.current_player + 1}\n"
        render_str += f"Player 1 Score: {self.current_scores[0]}\n"
        render_str += f"Player 2 Score: {self.current_scores[1]}\n"
        render_str += f"Die Roll: {self.die_roll}\n"
        render_str += f"Valid Moves: {['Add', 'Subtract'] if len(self.valid_moves()) == 2 else ['Add'] if 0 in self.valid_moves() else ['Subtract'] if 1 in self.valid_moves() else []}\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid moves for the current state
        current_score = self.current_scores[self.current_player]
        die_roll = self.die_roll

        valid_moves = []
        if 0 <= current_score + die_roll <= 15:
            valid_moves.append(0)  # Add is valid
        if 0 <= current_score - die_roll <= 15:
            valid_moves.append(1)  # Subtract is valid
        return valid_moves

    def _get_observation(self):
        # Build and return the observation
        observation = np.array(
            [
                self.current_scores[self.current_player],  # Current player's score
                self.current_scores[1 - self.current_player],  # Opponent's score
                self.die_roll,  # Current die roll
            ],
            dtype=np.int32,
        )
        return observation
